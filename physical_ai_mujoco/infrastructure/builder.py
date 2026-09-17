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
from physical_ai_mujoco.observe import ExactObserver
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
            ).read_text()
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

    def observer(self, env):
        mode = env.get("components", {}).get("observer", "exact")
        if mode != "exact":
            raise ValueError(f"Observer non disponibile: {mode}")
        return ExactObserver()

    def executor(self, env):
        mode = env.get("components", {}).get("executor", "ideal_removal")
        if mode != "ideal_removal":
            raise ValueError(f"Executor non disponibile: {mode}")
        return IdealRemovalExecutor()

    def task(self, env, terminate_on_target=None, terminate_on_collapse=None):
        return TargetExtractionTask(
            env["task"], terminate_on_target, terminate_on_collapse
        )
