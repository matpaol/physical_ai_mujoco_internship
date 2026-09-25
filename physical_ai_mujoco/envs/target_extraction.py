"""Adapter Gymnasium: delega scena, osservazione, esecuzione e regole del task.

state conserva il vettore teacher di 17 valori per oggetto; stereo/both
conservano le immagini grezze. La ricostruzione stereo appartiene alla fase 1B.
"""

from dataclasses import asdict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from physical_ai_mujoco.contracts import (
    STATE_FEATURES_PER_OBJECT,
    ObjectDecision,
    TaskContext,
)
from physical_ai_mujoco.infrastructure.builder import ComponentBuilder
from physical_ai_mujoco.infrastructure.builder import PROJECT_ROOT as PROJECT_ROOT
from physical_ai_mujoco.decide import ObservationEncoder
from physical_ai_mujoco.observe import ExactObserver, SensorObserver
from physical_ai_mujoco.sensors import SimulatedStereoCamera
from physical_ai_mujoco.simulation import SceneSession


class TargetExtractionEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        obs_mode="state",
        render_mode=None,
        env_config_path=None,
        scene_rules_path=None,
        stereo_baseline=None,
        fixed_scene_seed=None,
        capture_camera=None,
        capture_stride=16,
        object_count=None,
        resample_shapes=True,
        highlight_target=False,
        realtime_factor=1.0,
        terminate_on_target=None,
        terminate_on_collapse=None,
        scene_pool_size=0,
        fresh_scene_probability=0.1,
        observer=None,
        executor=None,
        task_rules=None,
        sensor_source=None,
        observation_encoder=None,
    ):
        super().__init__()
        self.render_mode = render_mode
        builder = ComponentBuilder()
        inputs = builder.load(env_config_path, scene_rules_path)
        self._env_config = inputs.env
        configured_mode = inputs.env.get("observation_mode")
        if configured_mode is not None and obs_mode == "state":
            obs_mode = configured_mode
        if obs_mode not in {"state", "sensor", "stereo", "both"}:
            raise ValueError(
                "obs_mode deve essere 'state', 'sensor', 'stereo' o "
                f"'both': {obs_mode!r}"
            )
        self.obs_mode = obs_mode
        self._stereo_config = dict(inputs.env["stereo_camera"])
        if stereo_baseline is not None:
            if stereo_baseline <= 0:
                raise ValueError("La baseline stereo deve essere positiva")
            self._stereo_config["baseline"] = float(stereo_baseline)
        self.task_rules = (
            task_rules
            if task_rules is not None
            else builder.task(inputs.env, terminate_on_target, terminate_on_collapse)
        )
        self.observer = (
            observer
            if observer is not None
            else builder.observer(inputs.env, inputs.objects)
        )
        self.executor = (
            executor if executor is not None else builder.executor(inputs.env)
        )
        self.stereo_camera = SimulatedStereoCamera()
        self.session = SceneSession(
            inputs,
            self.task_rules,
            stereo_baseline=stereo_baseline,
            object_count=object_count,
            render_mode=render_mode,
            fixed_scene_seed=fixed_scene_seed,
            capture_camera=capture_camera,
            capture_stride=capture_stride,
            resample_shapes=resample_shapes,
            highlight_target=highlight_target,
            realtime_factor=realtime_factor,
            scene_pool_size=scene_pool_size,
            fresh_scene_probability=fresh_scene_probability,
        )
        self.object_count = self.session.object_count
        sensor_settings = inputs.env.get("sensor_observation", {})
        maximum_tracks = int(
            sensor_settings.get("max_tracks", max(self.object_count, 2 * self.object_count))
        )
        self.observation_encoder = (
            observation_encoder
            if observation_encoder is not None
            else ObservationEncoder(maximum_tracks)
        )
        self.sensor_source = sensor_source
        if isinstance(self.observer, SensorObserver) and self.sensor_source is None:
            self.sensor_source = builder.sensor_source(
                inputs.env, lambda: self.simulator
            )
        if self.obs_mode == "sensor" and not isinstance(self.observer, SensorObserver):
            raise ValueError("obs_mode='sensor' richiede un SensorObserver")
        self._latest_decision_observation = None
        self._latest_encoded = None
        self._last_selected_object_id = None
        self._sensor_action_max_distance = float(
            sensor_settings.get("action_match_max_distance", 0.15)
        )
        action_count = (
            self.observation_encoder.max_objects
            if self.obs_mode == "sensor"
            else self.object_count
        )
        self.action_space = spaces.Discrete(action_count)
        self.observation_space = self._build_observation_space()

    @property
    def simulator(self):
        return self.session.simulator

    @property
    def target_id(self):
        return self.session.target_id

    @property
    def object_ids(self):
        return self.session.object_ids

    @property
    def task(self):
        return dict(self.task_rules.config)

    @property
    def step_callback(self):
        return self.session.viewer.step_callback

    @step_callback.setter
    def step_callback(self, callback):
        self.session.viewer.step_callback = callback

    # Compatibility for existing analysis scripts. New code uses public APIs.
    @property
    def _object_ids(self):
        return self.object_ids

    @property
    def _scene_rules(self):
        return self.session.scene_rules

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        result = self.session.reset(self.np_random)
        self.observer.reset(result.scene_seed)
        if self.sensor_source is not None and hasattr(self.sensor_source, "reset"):
            self.sensor_source.reset(result.scene_seed + 50_000_101)
        self.observation_encoder.reset(result.scene_seed)
        self._latest_decision_observation = None
        self._latest_encoded = None
        self._last_selected_object_id = None
        self.task_rules.reset(self.session.current_positions(), self.target_id)
        observation = self._observation()
        info = self._build_info(0.0, False, result.settling)
        info.update(
            scene_seed=result.scene_seed,
            from_pool=result.from_pool,
            release_log=list(self.session.release_log),
        )
        return observation, info

    def _task_context(self, *, deployable=False):
        selection = self.session.scene_rules.get("object_selection", {})
        return TaskContext(
            target_id=None if deployable else self.target_id,
            ground_height=float(self.simulator.scene.ground.pose.position[2]),
            target_type_id=selection.get("target_type_id"),
        )

    def decision_observation(self, *, refresh=False):
        if self.simulator is None:
            raise RuntimeError("Chiama reset() prima di osservare")
        if (
            isinstance(self.observer, SensorObserver)
            and not refresh
            and self._latest_decision_observation is not None
        ):
            return self._latest_decision_observation
        if isinstance(self.observer, SensorObserver):
            if self.sensor_source is None:
                raise RuntimeError("SensorObserver privo di SensorSource")
            result = self.observer.observe(
                self.sensor_source.capture(), self._task_context(deployable=True)
            )
        else:
            result = self.observer.observe(
                self.simulator, self._task_context(deployable=False)
            )
        self._latest_decision_observation = result
        return result

    def action_index(self, decision: ObjectDecision):
        if self.obs_mode == "sensor" and self._latest_encoded is not None:
            return self._latest_encoded.slot_ids.index(decision.object_id)
        return self.object_ids.index(decision.object_id)

    def _sensor_action_object_id(self, action: int) -> str | None:
        encoded = self._latest_encoded
        observation = self._latest_decision_observation
        if encoded is None or observation is None or not encoded.action_mask[action]:
            return None
        track_id = encoded.slot_ids[action]
        perceived = next(
            (item for item in observation.scene.objects if item.object_id == track_id),
            None,
        )
        if perceived is None or perceived.position is None:
            return None
        candidates = []
        estimate = np.asarray(perceived.position, dtype=float)
        for object_id in self.object_ids:
            if not self.simulator.is_present(object_id):
                continue
            position = np.asarray(
                self.simulator.get_object_state(object_id).position, dtype=float
            )
            candidates.append((float(np.linalg.norm(position - estimate)), object_id))
        if not candidates:
            return None
        distance, object_id = min(candidates)
        return object_id if distance <= self._sensor_action_max_distance else None

    def step(self, action):
        if self.simulator is None:
            raise RuntimeError("Chiama reset() prima di step()")
        action = int(action)
        if not 0 <= action < self.action_space.n:
            raise ValueError(f"Azione fuori range: {action}")
        selected_id = (
            self._sensor_action_object_id(action)
            if self.obs_mode == "sensor"
            else self.object_ids[action]
        )
        self._last_selected_object_id = selected_id
        if selected_id is None:
            observation = self._observation()
            info = self._build_info(0.0, True)
            info["execution"] = None
            return (
                observation,
                self.task_rules.invalid_reward(),
                False,
                False,
                info,
            )
        execution = self.executor.execute(
            ObjectDecision(selected_id), self.simulator
        )
        if not execution.removed:
            # L'esecutore ideale conosce solo l'errore already_removed.
            # Gli esiti motori aggiuntivi saranno definiti nella fase 2.
            observation = self._observation()
            info = self._build_info(0.0, True)
            info["execution"] = execution
            return (
                observation,
                self.task_rules.invalid_reward(),
                False,
                False,
                info,
            )
        settling = self.session.settle()
        result = self.task_rules.evaluate(execution, self.session.current_positions())
        observation = self._observation()
        info = self._build_info(result.disturbance, False, settling)
        info.update(
            {
                k: v
                for k, v in asdict(result).items()
                if k not in {"reward", "terminated", "disturbance"}
            }
        )
        info["execution"] = execution
        if self.render_mode == "human":
            self.render()
        return observation, result.reward, result.terminated, False, info

    def action_masks(self):
        if self.simulator is None:
            return np.ones(self.action_space.n, dtype=bool)
        if self.obs_mode == "sensor":
            if self._latest_encoded is None:
                return np.zeros(self.action_space.n, dtype=bool)
            return self._latest_encoded.action_mask.copy()
        return np.asarray(
            [self.simulator.is_present(i) for i in self.object_ids], dtype=bool
        )

    def _observation(self):
        if self.obs_mode == "state":
            return self._state_observation()
        if self.obs_mode == "sensor":
            decision_observation = self.decision_observation(refresh=True)
            self._latest_encoded = self.observation_encoder.encode(
                decision_observation
            )
            return self._latest_encoded.vector
        frame = self.stereo_camera.capture(self.simulator)
        result = {"rgb_left": frame.rgb_left, "rgb_right": frame.rgb_right}
        if self.obs_mode == "both":
            result["state"] = self._state_observation()
        return result

    def privileged_state(self):
        """Simulation-only truth of the current scene, for teachers and evaluation.

        It does not depend on the configured observer: even with the real
        perception pipeline the teacher can read the simulator.
        """
        if self.simulator is None:
            raise RuntimeError("Call reset() before privileged_state()")
        return ExactObserver().privileged_state(self.simulator, self.target_id)

    def _state_observation(self):
        # Legacy teacher path: not the deployable Observation given to DECIDE.
        return self.privileged_state().as_vector()

    def _build_info(self, disturbance, invalid_action, settling=None):
        return dict(
            action_mask=self.action_masks(),
            target_id=self.target_id,
            target_index=self.object_ids.index(self.target_id)
            if self.target_id in self.object_ids
            else -1,
            present_objects=self.simulator.present_objects(),
            disturbance_step=disturbance,
            disturbance_total=self.task_rules.total_disturbance,
            invalid_action=invalid_action,
            selected_object_id=self._last_selected_object_id,
            settled=True if settling is None else settling.settled,
            settling_duration=0.0 if settling is None else settling.duration,
            frames=[] if settling is None else settling.frames,
        )

    def snapshot(self):
        """Fotografia dell'episodio corrente; il wrapper conserva il proprio contatore."""
        return dict(
            simulation=self.simulator.snapshot(), task=self.task_rules.snapshot()
        )

    def restore(self, snapshot):
        self.simulator.restore(snapshot["simulation"])
        self.task_rules.restore(snapshot["task"])
        self.session.target_id = self.task_rules.target_id

    def render(self):
        if self.render_mode == "human" and self.session.viewer.renderer is not None:
            return self.session.viewer.renderer.render("human")
        if self.render_mode == "rgb_array":
            return self.simulator.render_camera("cam_left")

    def close(self):
        self.session.close()

    def _build_observation_space(self) -> spaces.Space:
        state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.object_count * STATE_FEATURES_PER_OBJECT,),
            dtype=np.float32,
        )
        image_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                int(self._stereo_config["height"]),
                int(self._stereo_config["width"]),
                3,
            ),
            dtype=np.uint8,
        )

        if self.obs_mode == "state":
            return state_space
        if self.obs_mode == "sensor":
            return spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.observation_encoder.vector_size,),
                dtype=np.float32,
            )
        if self.obs_mode == "stereo":
            return spaces.Dict({"rgb_left": image_space, "rgb_right": image_space})
        return spaces.Dict(
            {
                "state": state_space,
                "rgb_left": image_space,
                "rgb_right": image_space,
            }
        )
