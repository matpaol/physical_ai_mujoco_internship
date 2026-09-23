"""Composizione all'avvio; nessun framework di dependency injection."""

from dataclasses import dataclass
from pathlib import Path
import json
from physical_ai_mujoco.scene.dataset_loader import (
    load_object_dataset,
    load_ground_dataset,
    load_scene_rules,
    load_simulation_config,
)
from physical_ai_mujoco.observe import (
    CADMatcher,
    DegradedObserver,
    ExactObserver,
    LidarGeometryEstimator,
    OracleObserver,
    SensorObserver,
)
from physical_ai_mujoco.sensors import (
    ImageDisturbance,
    LearnedDetector,
    SimulatedSensorSource,
    resolve_detector_weights,
)
from physical_ai_mujoco.sensors.lidar import LidarConfig, LidarNoise
from physical_ai_mujoco.execute import IdealRemovalExecutor
from physical_ai_mujoco.task import TargetExtractionTask

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ProjectInputs:
    env: dict
    scene_rules: dict
    objects: dict
    grounds: dict
    simulation: dict


class ComponentBuilder:
    def load(self, env_config_path=None, scene_rules_path=None):
        env = json.loads(
            Path(
                env_config_path or PROJECT_ROOT / "configs/phase_0b/env.json"
            ).read_text(encoding="utf-8")
        )
        from physical_ai_mujoco.infrastructure.experiment import selected_profile

        profile = selected_profile()
        if profile is not None:
            if not profile.available:
                raise ValueError(profile.description)
            for key, value in profile.env_overrides.items():
                if isinstance(value, dict):
                    env[key] = {**env.get(key, {}), **value}
                else:
                    env[key] = value
        datasets = env.get("datasets", {})
        return ProjectInputs(
            env,
            load_scene_rules(
                scene_rules_path or PROJECT_ROOT / "configs/phase_0b/scene_rules.json"
            ),
            load_object_dataset(
                PROJECT_ROOT
                / datasets.get(
                    "objects", "datasets/object_dataset/geometric_objects.json"
                )
            ),
            load_ground_dataset(
                PROJECT_ROOT
                / datasets.get("grounds", "datasets/ground_dataset/basic_grounds.json")
            ),
            load_simulation_config(
                PROJECT_ROOT
                / datasets.get("simulation", "configs/phase_0a/simulation.json")
            ),
        )

    def observer(self, env, objects=None):
        mode = env.get("components", {}).get("observer", "exact")
        if mode == "exact":
            return ExactObserver()
        if mode == "oracle":
            return OracleObserver()
        if mode == "degraded":
            settings = env.get("degraded_observation", {})
            return DegradedObserver(
                position_sigma=settings.get("position_sigma", 0.01),
                drop_probability=settings.get("drop_probability", 0.1),
            )
        if mode == "sensor_learned":
            settings = env.get("sensor_observation", {})
            detector = LearnedDetector(
                weights_path=resolve_detector_weights(settings.get("detector_weights")),
                confidence_threshold=float(
                    settings.get("detector_confidence_threshold", 0.25)
                ),
                class_names=tuple(
                    settings.get("class_names", ("obstacle", "pfm_1_target"))
                ),
            )
            cad_matchers = {}
            if objects is not None and settings.get("cad_matching", True):
                target_types = set(settings.get("cad_target_type_ids", ("pfm_1_target",)))
                for definition in objects.get("object_types", ()):
                    if (
                        definition.get("id") in target_types
                        and definition.get("shape") == "mesh"
                    ):
                        cad_matchers[definition["id"]] = CADMatcher.from_stl(
                            definition["mesh_file"],
                            tuple(definition["mesh_scale"]),
                        )
            return SensorObserver(
                detector,
                geometry_estimator=LidarGeometryEstimator(
                    mesh_target_type_ids=tuple(cad_matchers),
                    cad_matchers=cad_matchers,
                ),
            )
        raise ValueError(f"Observer non disponibile: {mode}")

    def sensor_source(self, env, simulator_provider, *, seed=0):
        settings = env.get("sensor_observation", {})
        lidar_path = settings.get(
            "lidar_config", "configs/sensors/livox_avia.json"
        )
        disturbance = settings.get("disturbance", {})
        image = None
        lidar_noise = None
        if disturbance:
            image = ImageDisturbance(
                contrast_range=tuple(disturbance.get("contrast_range", (1.0, 1.0))),
                brightness_range=tuple(
                    disturbance.get("brightness_range", (0.0, 0.0))
                ),
                gamma_range=tuple(disturbance.get("gamma_range", (1.0, 1.0))),
                noise_sigma_range=tuple(
                    disturbance.get("noise_sigma_range", (0.0, 0.0))
                ),
                blur_probability=float(disturbance.get("blur_probability", 0.0)),
            )
            lidar_noise = LidarNoise(
                distance_sigma=float(disturbance.get("lidar_distance_sigma", 0.0)),
                angle_sigma_deg=float(disturbance.get("lidar_angle_sigma_deg", 0.0)),
                dropout_probability=float(
                    disturbance.get("lidar_drop_probability", 0.0)
                ),
            )
        return SimulatedSensorSource(
            simulator_provider,
            lidar_config=LidarConfig.from_file(PROJECT_ROOT / lidar_path),
            image_disturbance=image,
            lidar_noise=lidar_noise,
            seed=seed,
        )

    def executor(self, env):
        mode = env.get("components", {}).get("executor", "ideal_removal")
        if mode != "ideal_removal":
            raise ValueError(f"Executor non disponibile: {mode}")
        return IdealRemovalExecutor()

    def task(self, env, terminate_on_target=None, terminate_on_collapse=None):
        return TargetExtractionTask(
            env["task"], terminate_on_target, terminate_on_collapse
        )
