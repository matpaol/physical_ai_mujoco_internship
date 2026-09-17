from pathlib import Path

import numpy as np
import pytest

from physical_ai_mujoco.scene.dataset_loader import (
    load_ground_dataset,
    load_object_dataset,
    load_scene_rules,
    load_simulation_config,
    validate_references,
)
from physical_ai_mujoco.scene.scene_generator import generate_scene


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs():
    object_dataset = load_object_dataset(
        PROJECT_ROOT / "datasets/object_dataset/geometric_objects.json"
    )
    ground_dataset = load_ground_dataset(
        PROJECT_ROOT / "datasets/ground_dataset/basic_grounds.json"
    )
    scene_rules = load_scene_rules(
        PROJECT_ROOT / "configs/phase_0a/scene_rules.json"
    )
    simulation_config = load_simulation_config(
        PROJECT_ROOT / "configs/phase_0a/simulation.json"
    )
    return object_dataset, ground_dataset, scene_rules, simulation_config


def test_input_references(inputs):
    object_dataset, ground_dataset, scene_rules, _ = inputs
    validate_references(object_dataset, ground_dataset, scene_rules)


def test_same_seed_generates_same_scene(inputs):
    first = generate_scene(*inputs, seed=42)
    second = generate_scene(*inputs, seed=42)
    assert first == second


def test_different_seeds_generate_different_scenes(inputs):
    first = generate_scene(*inputs, seed=42)
    second = generate_scene(*inputs, seed=43)
    assert first != second


def test_scene_contains_required_types(inputs):
    scene = generate_scene(*inputs, seed=42)
    type_ids = [item.type_id for item in scene.objects]
    assert type_ids == ["box", "cylinder"]


def test_generated_values_are_physical(inputs):
    scene = generate_scene(*inputs, seed=42)
    for item in scene.objects:
        assert item.mass > 0
        assert item.density > 0
        assert all(value > 0 for value in item.size.values())
        assert np.isclose(np.linalg.norm(item.pose.quaternion), 1.0)


def test_mujoco_model_compiles_and_settles(inputs):
    pytest.importorskip("mujoco")
    from physical_ai_mujoco.simulation.settling import run_until_settled
    from physical_ai_mujoco.simulation.simulator import Simulator

    scene = generate_scene(*inputs, seed=42)
    simulator = Simulator(scene)
    try:
        result = run_until_settled(simulator)
    finally:
        simulator.close()

    assert result.settled
    assert set(result.object_states) == {"object_000", "object_001"}
    for state in result.object_states.values():
        assert state.position[2] > 0

