"""Adapter Gymnasium: delega scena, osservazione, esecuzione e regole del task.

state conserva il vettore teacher di 17 valori per oggetto; stereo/both
conservano le immagini grezze. La ricostruzione stereo appartiene alla fase 1B.
"""

from dataclasses import asdict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from physical_ai_mujoco.contracts import STATE_FEATURES_PER_OBJECT, ObjectDecision
from physical_ai_mujoco.infrastructure.builder import ComponentBuilder
from physical_ai_mujoco.infrastructure.builder import PROJECT_ROOT as PROJECT_ROOT
from physical_ai_mujoco.observe import StereoCapture
from physical_ai_mujoco.simulation import SceneSession


class TargetExtractionEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        obs_mode="state",
        render_mode=None,
        env_config_path=None,
        scene_rules_path=None,
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
    ):
        super().__init__()
        if obs_mode not in {"state", "stereo", "both"}:
            raise ValueError(
                f"obs_mode deve essere 'state', 'stereo' o 'both': {obs_mode!r}"
            )
        self.obs_mode = obs_mode
        self.render_mode = render_mode
        builder = ComponentBuilder()
        inputs = builder.load(env_config_path, scene_rules_path)
        self._env_config = inputs.env
        self._stereo_config = inputs.env["stereo_camera"]
        self.task_rules = (
            task_rules
            if task_rules is not None
            else builder.task(inputs.env, terminate_on_target, terminate_on_collapse)
        )
        self.observer = (
            observer if observer is not None else builder.observer(inputs.env)
        )
        self.executor = (
            executor if executor is not None else builder.executor(inputs.env)
        )
        self.stereo_capture = StereoCapture()
        self.session = SceneSession(
            inputs,
            self.task_rules,
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
        self.action_space = spaces.Discrete(self.object_count)
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
        self.task_rules.reset(self.session.current_positions(), self.target_id)
        info = self._build_info(0.0, False, result.settling)
        info.update(
            scene_seed=result.scene_seed,
            from_pool=result.from_pool,
            release_log=list(self.session.release_log),
        )
        return self._observation(), info

    def decision_observation(self):
        if self.simulator is None:
            raise RuntimeError("Chiama reset() prima di osservare")
        return self.observer.observe(self.simulator, self.target_id)

    def action_index(self, decision: ObjectDecision):
        return self.object_ids.index(decision.object_id)

    def step(self, action):
        if self.simulator is None:
            raise RuntimeError("Chiama reset() prima di step()")
        action = int(action)
        if not 0 <= action < self.object_count:
            raise ValueError(f"Azione fuori range: {action}")
        execution = self.executor.execute(
            ObjectDecision(self.object_ids[action]), self.simulator
        )
        if not execution.removed:
            # L'esecutore ideale conosce solo l'errore already_removed.
            # Gli esiti motori aggiuntivi saranno definiti nella fase 2.
            return (
                self._observation(),
                self.task_rules.invalid_reward(),
                False,
                False,
                self._build_info(0.0, True),
            )
        settling = self.session.settle()
        result = self.task_rules.evaluate(execution, self.session.current_positions())
        info = self._build_info(result.disturbance, False, settling)
        info.update(
            {
                k: v
                for k, v in asdict(result).items()
                if k not in {"reward", "terminated", "disturbance"}
            }
        )
        if self.render_mode == "human":
            self.render()
        return self._observation(), result.reward, result.terminated, False, info

    def action_masks(self):
        if self.simulator is None:
            return np.ones(self.object_count, dtype=bool)
        return np.asarray(
            [self.simulator.is_present(i) for i in self.object_ids], dtype=bool
        )

    def _observation(self):
        if self.obs_mode == "state":
            return self._state_observation()
        result = self.stereo_capture.capture(self.simulator)
        if self.obs_mode == "both":
            result["state"] = self._state_observation()
        return result

    def _state_observation(self):
        return self.decision_observation().as_vector()

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
        if self.obs_mode == "stereo":
            return spaces.Dict({"rgb_left": image_space, "rgb_right": image_space})
        return spaces.Dict(
            {
                "state": state_space,
                "rgb_left": image_space,
                "rgb_right": image_space,
            }
        )
