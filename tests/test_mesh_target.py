from pathlib import Path

import gymnasium as gym
import pytest

import physical_ai_mujoco.envs  # noqa: F401
from physical_ai_mujoco.infrastructure.builder import ComponentBuilder
from physical_ai_mujoco.simulation.mujoco_builder import build_mjcf


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pfm_mesh_is_the_unique_configured_target():
    inputs = ComponentBuilder().load()
    selection = inputs.scene_rules["object_selection"]
    assert selection["target_type_id"] == "pfm_1_target"
    assert selection["required_type_ids"].count("pfm_1_target") == 1

    definition = next(
        item for item in inputs.objects["object_types"] if item["id"] == "pfm_1_target"
    )
    assert definition["shape"] == "mesh"
    assert Path(definition["mesh_file"]).is_file()
    assert definition["mesh_scale"] == [0.001, 0.001, 0.001]


@pytest.mark.parametrize("object_count", [1, 2, 3, 8])
def test_every_scene_contains_one_pfm_and_selects_it_as_target(object_count):
    env = gym.make(
        "TargetExtraction-v0",
        object_count=object_count,
        disable_env_checker=True,
    )
    try:
        _, info = env.reset(seed=7)
        target = next(
            item for item in env.unwrapped.simulator.scene.objects
            if item.instance_id == info["target_id"]
        )
        assert target.type_id == "pfm_1_target"
        assert target.shape == "mesh"
        assert sum(
            item.type_id == "pfm_1_target"
            for item in env.unwrapped.simulator.scene.objects
        ) == 1
    finally:
        env.close()


def test_pfm_mesh_is_compiled_as_a_mujoco_mesh():
    import mujoco

    env = gym.make("TargetExtraction-v0", object_count=3, disable_env_checker=True)
    try:
        env.reset(seed=11)
        scene = env.unwrapped.simulator.scene
        target = next(item for item in scene.objects if item.type_id == "pfm_1_target")
        xml = build_mjcf(scene)
        assert f'name="{target.instance_id}_mesh"' in xml
        assert f'mesh="{target.instance_id}_mesh"' in xml
        assert target.mass == pytest.approx(0.075)
        body_id = mujoco.mj_name2id(
            env.unwrapped.simulator.model,
            mujoco.mjtObj.mjOBJ_BODY,
            target.instance_id,
        )
        assert env.unwrapped.simulator.model.body_mass[body_id] == pytest.approx(0.075)
    finally:
        env.close()
