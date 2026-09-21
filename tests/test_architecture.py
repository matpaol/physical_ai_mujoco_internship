"""Contratti, sostituibilita e confini della prima migrazione."""

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from physical_ai_mujoco.contracts import (
    ObjectObservation,
    Observation,
    PrivilegedState,
    SceneObject,
    SceneState,
    PhysicalRelationState,
    ObjectUncertainty,
    UncertaintyState,
    ObjectDecision,
    ExecutionOutcome,
)
from physical_ai_mujoco.decide import Decider, HighestObjectDecider, RandomDecider, PPODecider
from physical_ai_mujoco.execute import Executor
from physical_ai_mujoco.observe import ExactObserver
from physical_ai_mujoco.task import TargetExtractionTask

ROOT = Path(__file__).resolve().parents[1]


def sample_observation():
    def obj(name, z, present=True, target=False):
        return SceneObject(
            name, "target" if target else "obstacle", "box",
            (0.0, 0.0, z) if present else None,
            (1.0, 0.0, 0.0, 0.0) if present else None,
            "box", (0.1, 0.1, 0.1), present, 1.0 if present else None,
        )

    objects = (
        obj("buried", 0.1, target=True),
        obj("highest", 0.4),
        obj("removed", 10.0, False),
    )
    return Observation(
        SceneState(objects, "buried", 0.0, None, "world", 0.0),
        PhysicalRelationState(tuple(o.object_id for o in objects), (), "test", True, "world", 0.0),
        UncertaintyState(tuple(ObjectUncertainty(o.object_id, o.present, 1.0, 1.0) for o in objects), False, "test", "world", 0.0),
    )


def test_deciders_use_data_and_preserve_ties_and_random_sequence():
    observation = sample_observation()
    assert HighestObjectDecider().decide(observation).object_id == "highest"
    tied = replace(observation, scene=replace(
        observation.scene,
        objects=tuple(replace(o, position=(0, 0, 0.1)) for o in observation.objects),
    ))
    assert HighestObjectDecider().decide(tied).object_id == "buried"
    expected = np.random.default_rng(18)
    policy = RandomDecider(np.random.default_rng(18))
    for _ in range(20):
        decision = policy.decide(observation)
        assert tuple(o.object_id for o in observation.objects).index(decision.object_id) == int(expected.choice([0, 1]))
    with pytest.raises(ValueError):
        HighestObjectDecider().decide(replace(observation, scene=replace(observation.scene, objects=())))


def test_contract_and_policy_modules_do_not_import_backend_or_environment():
    for folder in ("contracts", "decide", "task", "observe"):
        for path in (ROOT / "physical_ai_mujoco" / folder).glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                imports = []
                if isinstance(node, ast.Import):
                    imports = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imports = [node.module or ""]
                for module in imports:
                    assert not any(
                        module == x or module.startswith(x + ".")
                        for x in (
                            "mujoco",
                            "gymnasium",
                            "rclpy",
                            "physical_ai_mujoco.simulation",
                            "physical_ai_mujoco.envs",
                            "scripts",
                        )
                    ), (path, module)


def test_sensor_implementations_do_not_import_mujoco():
    for path in (ROOT / "physical_ai_mujoco" / "sensors").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(name == "mujoco" or name.startswith("mujoco.") for name in names), path


def test_vision_training_changes_appearance_only_through_simulator_api():
    """Il visore non tocca MuJoCo: usa Simulator.apply_visual_conditions."""
    for path in (ROOT / "physical_ai_mujoco" / "vision_training").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(name == "mujoco" or name.startswith("mujoco.") for name in names), path


def test_visual_conditions_are_plain_data():
    """Le condizioni visive si possono campionare senza caricare MuJoCo."""
    path = ROOT / "physical_ai_mujoco" / "simulation" / "visual_conditions.py"
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            assert not any(name.startswith("mujoco") for name in names)


def test_shared_contracts_do_not_depend_on_sensor_implementations():
    for path in (ROOT / "physical_ai_mujoco" / "contracts").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(
                name == "physical_ai_mujoco.sensors"
                or name.startswith("physical_ai_mujoco.sensors.")
                for name in names
            ), (path, names)


def make_task(**overrides):
    return TargetExtractionTask(
        dict(
            removal_cost=0.05,
            disturbance_penalty=4.0,
            disturbance_threshold=0.05,
            target_reward=1.0,
            failure_penalty=1.0,
            terminate_on_target=overrides.get("terminate_on_target", True),
            terminate_on_collapse=overrides.get("terminate_on_collapse", False),
        )
    )


def test_task_threshold_and_one_time_target_reward():
    task = make_task(terminate_on_target=False)
    task.reset({"target": np.zeros(3), "other": np.zeros(3)}, "target")
    first = task.evaluate(
        ExecutionOutcome("target", True), {"other": np.array([0.05, 0, 0])}
    )
    assert first.reward == pytest.approx(0.75)
    assert not first.collapsed  # threshold is strictly greater, not >=
    assert not first.terminated
    last = task.evaluate(ExecutionOutcome("other", True), {})
    assert last.reward == -0.05
    assert last.terminated
    assert not last.target_just_removed


def test_task_snapshot_restores_collapse_history_and_reference():
    task = make_task()
    positions = {"target": np.zeros(3), "other": np.zeros(3)}
    task.reset(positions, "target")
    snapshot = task.snapshot()
    result = task.evaluate(
        ExecutionOutcome("third", True),
        {"target": np.zeros(3), "other": np.array([0.06, 0, 0])},
    )
    assert result.collapsed_ever
    task.restore(snapshot)
    result = task.evaluate(ExecutionOutcome("target", True), {"other": np.zeros(3)})
    assert result.is_success
    assert task.total_disturbance == 0


def test_env_delegates_to_replaceable_objects_and_keeps_public_action_mapping():
    import gymnasium as gym
    import physical_ai_mujoco.envs  # noqa: F401

    class CountingObserver(ExactObserver):
        calls = 0

        def observe(self, simulator, target_id):
            self.calls += 1
            return super().observe(simulator, target_id)

    class RejectingExecutor(Executor):
        calls = 0

        def execute(self, decision, simulator):
            self.calls += 1
            return ExecutionOutcome(decision.object_id, False, "already_removed")

    observer = CountingObserver()
    executor = RejectingExecutor()
    env = gym.make(
        "TargetExtraction-v0", object_count=2, observer=observer, executor=executor
    )
    try:
        old, info = env.reset(seed=2)
        action = env.unwrapped.action_index(ObjectDecision(info["target_id"]))
        obs, reward, terminated, truncated, info = env.step(action)
        env.unwrapped.decision_observation()
        assert executor.calls == 1
        assert observer.calls >= 1
        np.testing.assert_array_equal(old, obs)
        assert info["invalid_action"] and reward < 0 and not terminated
    finally:
        env.close()


def test_episode_snapshot_restores_visibility_and_task():
    import gymnasium as gym
    import physical_ai_mujoco.envs  # noqa: F401

    env = gym.make("TargetExtraction-v0", object_count=3, terminate_on_target=False)
    try:
        before, info = env.reset(seed=2)
        snapshot = env.unwrapped.snapshot()
        simulator = env.unwrapped.simulator
        colors = simulator.model.geom_rgba.copy()
        env.step(0)
        env.unwrapped.restore(snapshot)
        np.testing.assert_array_equal(colors, simulator.model.geom_rgba)
        np.testing.assert_array_equal(
            before, env.unwrapped._state_observation()
        )
        assert env.unwrapped.action_masks().all()
        assert env.unwrapped.task_rules.total_disturbance == 0
    finally:
        env.close()


def test_ppo_normalizes_and_does_not_mask_invalid_actions():
    class Space:
        shape = (51,)

    class Model:
        observation_space = Space()

        def predict(self, vector, deterministic):
            assert deterministic
            np.testing.assert_array_equal(vector, privileged.as_vector() * 2)
            return 2, None

    class Normalizer:
        def normalize_obs(self, vector):
            return vector * 2

    privileged = PrivilegedState(tuple(
        ObjectObservation(o.object_id, o.position or (0, 0, 0),
                          (1, 0, 0, 0), (0, 0, 0), (0, 0, 0),
                          1.0, 0.5, o.present, o.is_target)
        for o in sample_observation().objects
    ))
    policy = PPODecider(Model(), Normalizer())
    assert policy.predict_index(privileged.as_vector()) == 2
    assert not isinstance(policy, Decider)
    with pytest.raises(TypeError, match="teacher"):
        policy.decide(sample_observation())
    with pytest.raises(ValueError, match="numero di oggetti"):
        policy.predict_index(np.zeros(17))


def test_experiment_profile_configures_components_without_phase_branches(
    tmp_path, monkeypatch
):
    import json
    from physical_ai_mujoco.infrastructure.experiment import ExperimentProfile
    from physical_ai_mujoco.infrastructure.builder import ComponentBuilder

    path = tmp_path / "profile.json"
    data = dict(
        name="Nome arbitrario, nessuna fase nel codice",
        available=True,
        description="test",
        default_decider="random",
        operations=["1"],
        env_overrides={
            "components": {"observer": "exact", "executor": "ideal_removal"}
        },
    )
    path.write_text(json.dumps(data))
    monkeypatch.setenv("PHYSICAL_AI_EXPERIMENT", str(path))
    builder = ComponentBuilder()
    inputs = builder.load()
    assert isinstance(builder.observer(inputs.env), ExactObserver)
    data["available"] = False
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="test"):
        ExperimentProfile.load(path).activate()
    with pytest.raises(ValueError, match="test"):
        builder.load()


def test_pool_restores_visible_objects_after_removal():
    import gymnasium as gym
    import physical_ai_mujoco.envs  # noqa: F401

    env = gym.make(
        "TargetExtraction-v0",
        object_count=2,
        scene_pool_size=1,
        fresh_scene_probability=0,
    )
    try:
        env.reset(seed=0)
        expected = env.unwrapped.simulator.model.geom_rgba.copy()
        env.step(0)
        _, info = env.reset(seed=1)
        assert info["from_pool"]
        np.testing.assert_array_equal(expected, env.unwrapped.simulator.model.geom_rgba)
    finally:
        env.close()
