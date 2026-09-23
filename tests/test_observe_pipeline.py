"""Contratti e confine fra OSSERVA, DECIDE e teacher privilegiato."""

from dataclasses import asdict, fields, replace
from pathlib import Path
import sys
from types import SimpleNamespace

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from physical_ai_mujoco.observe import DegradedObserver, ExactObserver
from physical_ai_mujoco.observe import StereoObserver
from physical_ai_mujoco.contracts import (
    Observation, ObservationInvariantError, SceneObject, SensorBundle,
    ObjectUncertainty, PerceptualObject, PerceptualState, PhysicalRelation,
    SensorEvidence, TaskContext, validate_observation,
)
from physical_ai_mujoco.observe.pipeline import ObservationBuilder, SceneUnderstanding


LOWER_CONTEXT = TaskContext("lower", 0.0)


def fake_simulator(mass=1.0):
    objects = (
        SimpleNamespace(
            instance_id="lower", type_id="box", shape="box",
            size={"x": 0.1, "y": 0.1, "z": 0.1}, mass=mass,
            friction=(0.5, 0.01, 0.01), center_of_mass=(0.0, 0.0, 0.0),
        ),
        SimpleNamespace(
            instance_id="upper", type_id="box", shape="box",
            size={"x": 0.1, "y": 0.1, "z": 0.1}, mass=2.0,
            friction=(0.7, 0.02, 0.003), center_of_mass=(0.01, 0.0, -0.02),
        ),
    )
    states = {
        "lower": SimpleNamespace(position=(0.0, 0.0, 0.05), quaternion=(1.0, 0.0, 0.0, 0.0), linear_velocity=(0, 0, 0), angular_velocity=(0, 0, 0)),
        "upper": SimpleNamespace(position=(0.0, 0.0, 0.15), quaternion=(1.0, 0.0, 0.0, 0.0), linear_velocity=(0, 0, 0), angular_velocity=(0, 0, 0)),
    }
    return SimpleNamespace(
        scene=SimpleNamespace(objects=objects, ground=SimpleNamespace(pose=SimpleNamespace(position=(0, 0, 0)))),
        time=3.0,
        is_present=lambda name: True,
        get_object_state=lambda name: states[name],
        support_graph=lambda: {"upper": ["lower"], "lower": ["terreno"]},
    )


def test_exact_pipeline_is_structured_and_keeps_privileged_values_separate():
    observer = ExactObserver()
    first = observer.observe(fake_simulator(mass=1.0), LOWER_CONTEXT)
    second = observer.observe(fake_simulator(mass=8.0), LOWER_CONTEXT)
    assert first == second
    assert first.scene.target_id == "lower"
    assert first.relations.relations[0].source_id == "lower"
    assert first.relations.relations[0].target_id == "upper"
    assert first.relations.relations[0].relation_type == "candidate_support"
    assert first.uncertainty.unknown_space is False
    assert "mass" not in str(asdict(first))
    assert "friction" not in str(asdict(first))
    assert not hasattr(first, "as_vector")
    privileged = observer.privileged_state(fake_simulator(mass=8.0), "lower")
    assert privileged.objects[0].mass == 8.0
    assert privileged.as_vector().shape == (34,)


def test_privileged_state_exists_only_on_exact_observer():
    assert hasattr(ExactObserver(), "privileged_state")
    assert not hasattr(DegradedObserver(), "privileged_state")
    assert not hasattr(StereoObserver(), "privileged_state")


def test_all_observers_require_the_same_source_context_contract():
    with pytest.raises(TypeError, match="TaskContext"):
        ExactObserver().observe(fake_simulator(), "lower")


def test_target_selection_prefers_classification_confidence():
    objects = (
        PerceptualObject(
            "wrong", "pfm_1_target", (0, 0, 0), None, "box", (1, 1, 1),
            True, 0.8, classification_confidence=0.2,
        ),
        PerceptualObject(
            "right", "pfm_1_target", (1, 0, 0), None, "box", (1, 1, 1),
            True, 0.4, classification_confidence=0.9,
        ),
    )
    scene = SceneUnderstanding().build(
        PerceptualState(objects, "world", 0.0),
        SensorEvidence("world", 0.0),
        TaskContext(None, 0.0, target_type_id="pfm_1_target"),
    )
    assert scene.target_id == "right"


def test_operational_observation_has_only_allowed_fields():
    observation = ExactObserver().observe(fake_simulator(), LOWER_CONTEXT)
    assert {field.name for field in fields(Observation)} == {
        "scene", "relations", "uncertainty"
    }
    assert {field.name for field in fields(SceneObject)} == {
        "object_id", "role", "type_id", "position", "quaternion", "shape",
        "size", "detected", "perception_quality",
    }
    assert all(
        not any(hasattr(item, forbidden) for forbidden in (
            "mass", "friction", "linear_velocity", "angular_velocity",
        ))
        for item in observation.scene.objects
    )


def test_observe_never_calls_privileged_state(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("observe() ha letto privileged_state")

    monkeypatch.setattr(ExactObserver, "privileged_state", forbidden)
    ExactObserver().observe(fake_simulator(), LOWER_CONTEXT)
    DegradedObserver(position_sigma=0, drop_probability=0).observe(
        fake_simulator(), LOWER_CONTEXT
    )
    blank = np.zeros((4, 4, 3), dtype=np.uint8)
    bundle = SensorBundle(
        blank, blank.copy(), np.eye(3), np.eye(4), 0.1, 0.0, "world", {}, {},
    )
    StereoObserver().observe(bundle, TaskContext("lower", 0.0))


def test_contract_validator_checks_output_without_rebuilding():
    observation = ExactObserver().observe(fake_simulator(), LOWER_CONTEXT)
    validate_observation(observation)
    malformed = replace(
        observation,
        relations=replace(observation.relations, frame="camera"),
    )
    with pytest.raises(ObservationInvariantError, match="Frame"):
        validate_observation(malformed)


def test_contract_validator_rejects_each_structural_violation():
    observation = ExactObserver().observe(fake_simulator(), LOWER_CONTEXT)
    scene, relations, uncertainty = (
        observation.scene, observation.relations, observation.uncertainty
    )
    cases = (
        (replace(observation, scene=replace(scene, objects=(scene.objects[0], scene.objects[0]))), "duplicati"),
        (replace(observation, relations=replace(relations, object_ids=("lower",))), "grafo"),
        (replace(observation, uncertainty=replace(uncertainty, objects=(uncertainty.objects[0],))), "incertezza"),
        (replace(observation, uncertainty=replace(
            uncertainty, objects=uncertainty.objects + (ObjectUncertainty("old", True, None, None),),
        )), "visibile"),
        (replace(observation, relations=replace(relations, timestamp=99.0)), "Timestamp"),
        (replace(observation, relations=replace(
            relations, relations=relations.relations + (
                PhysicalRelation("lower", "missing", "candidate_support", 0.5),
            ),
        )), "assente"),
    )
    for malformed, message in cases:
        with pytest.raises(ObservationInvariantError, match=message):
            validate_observation(malformed)


def test_degraded_source_reproduces_dropouts_and_preserves_unknown():
    observer = DegradedObserver(position_sigma=0.01, drop_probability=0.5)
    observer.reset(42)
    first = observer.observe(fake_simulator(), LOWER_CONTEXT)
    observer.reset(42)
    second = observer.observe(fake_simulator(), LOWER_CONTEXT)
    assert first == second
    assert first.uncertainty.unknown_space == (len(first.objects) < 2)
    assert set(o.object_id for o in first.objects) <= {"lower", "upper"}
    assert all(item.pose_quality is not None for item in first.uncertainty.objects)


def test_degraded_uncertainty_remembers_previously_seen_ids_without_pose():
    observer = DegradedObserver(position_sigma=0, drop_probability=0)
    observer.reset(7)
    first = observer.observe(fake_simulator(), LOWER_CONTEXT)
    assert len(first.objects) == 2
    observer.extractor.drop_probability = 1.0
    second = observer.observe(fake_simulator(), LOWER_CONTEXT)
    assert second.objects == ()
    assert set(second.uncertainty.unseen_object_ids) == {"lower", "upper"}
    assert second.uncertainty.unknown_space
    assert all(item.pose_quality is None for item in second.uncertainty.objects)


def test_observation_builder_rejects_misaligned_products():
    obs = ExactObserver().observe(fake_simulator(), LOWER_CONTEXT)
    builder = ObservationBuilder()
    with pytest.raises(ValueError, match="Frame"):
        builder.build(obs.scene, replace(obs.relations, frame="camera"), obs.uncertainty)
    with pytest.raises(ValueError, match="ID"):
        builder.build(obs.scene, replace(obs.relations, object_ids=("lower",)), obs.uncertainty)
    with pytest.raises(ValueError, match="Timestamp"):
        builder.build(obs.scene, obs.relations, replace(obs.uncertainty, timestamp=4.0))


def test_stereo_bundle_triangulates_without_simulator_state():
    left = np.zeros((100, 100, 3), dtype=np.uint8)
    right = left.copy()
    left_mask = np.zeros((100, 100), dtype=bool)
    right_mask = left_mask.copy()
    left_mask[49:52, 59:62] = True
    right_mask[49:52, 49:52] = True
    bundle = SensorBundle(
        left, right,
        np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]]),
        np.eye(4), 0.1, 2.0, "world",
        {"track-1": left_mask}, {"track-1": right_mask},
    )
    result = StereoObserver().observe(bundle, TaskContext("track-1", 0.0))
    assert result.scene.target_id == "track-1"
    assert result.objects[0].position == pytest.approx((0.1, 0.0, 1.0))
    assert result.objects[0].is_target
    assert result.relations.available is False  # geometria non ancora stimata
    assert result.uncertainty.unknown_space
    assert result.uncertainty.objects[0].pose_quality is None


def test_environment_exposes_operational_and_teacher_channels_separately():
    import gymnasium as gym
    import physical_ai_mujoco.envs  # noqa: F401

    env = gym.make("TargetExtraction-v0", object_count=2, disable_env_checker=True)
    try:
        vector, _ = env.reset(seed=7)
        operational = env.unwrapped.decision_observation()
        assert vector.shape == (34,)
        assert len(operational.objects) == 2
        assert not hasattr(operational, "as_vector")
        assert all(o.position is not None for o in operational.objects)
        assert np.isfinite(vector).all()
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# --------------------------------------------------------- MuJoCo contact oracle

def _fake_oracle_simulator(supports):
    simulator = fake_simulator()
    simulator.support_graph = lambda: supports
    return simulator


def test_privileged_state_exposes_physical_truth_without_changing_teacher_vector():
    privileged = ExactObserver().privileged_state(fake_simulator(), "lower")
    upper = privileged.objects[1]
    assert (upper.type_id, upper.shape, upper.size) == ("box", "box", (0.1, 0.1, 0.1))
    assert upper.center_of_mass == (0.01, 0.0, -0.02)
    assert (upper.friction, upper.torsional_friction, upper.rolling_friction) == (0.7, 0.02, 0.003)
    # The legacy teacher encoding keeps 17 values per object.
    assert privileged.as_vector().shape == (34,)
    assert privileged.as_vector()[17 + 14] == pytest.approx(0.7)


def test_privileged_state_carries_contact_supports_between_present_objects():
    # Ground and objects that are no longer in the scene are not supports.
    supports = {"lower": ["terreno", "upper", "parked"], "upper": ["terreno"]}
    privileged = ExactObserver().privileged_state(_fake_oracle_simulator(supports), "lower")
    assert privileged.contact_supports == (("upper", "lower"),)


def test_oracle_relations_come_from_contacts_not_geometry():
    from physical_ai_mujoco.observe import OracleObserver

    # The geometry says "upper" rests on "lower"; the contacts say the opposite.
    supports = {"lower": ["terreno", "upper"], "upper": ["terreno"]}
    observation = OracleObserver().observe(_fake_oracle_simulator(supports), LOWER_CONTEXT)
    validate_observation(observation)
    edges = [(r.source_id, r.target_id, r.relation_type, r.relation_score)
             for r in observation.relations.relations]
    assert edges == [("upper", "lower", "support", 1.0)]
    assert observation.relations.estimator == "mujoco_contacts_oracle"
    assert observation.relations.available is True
    assert observation.relations.dependency_graph == {"lower": (), "upper": ("lower",)}


def test_oracle_changes_only_the_relations_of_the_exact_observation():
    from physical_ai_mujoco.observe import OracleObserver

    simulator = _fake_oracle_simulator({"upper": ["lower"], "lower": ["terreno"]})
    oracle = OracleObserver().observe(simulator, LOWER_CONTEXT)
    exact = ExactObserver().observe(simulator, LOWER_CONTEXT)
    assert oracle.scene == exact.scene
    assert oracle.uncertainty == exact.uncertainty
    assert oracle.relations.estimator != exact.relations.estimator


def test_oracle_estimator_requires_the_privileged_branch():
    from physical_ai_mujoco.observe.pipeline import OracleRelationEstimator

    scene = ExactObserver().observe(fake_simulator(), LOWER_CONTEXT).scene
    with pytest.raises(TypeError, match="PrivilegedState"):
        OracleRelationEstimator().estimate(scene, SensorEvidence("world", 3.0))


def test_oracle_graph_matches_mujoco_support_graph_on_real_scene():
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("mujoco")
    import physical_ai_mujoco.envs  # noqa: F401  (registers the environment)
    from physical_ai_mujoco.observe import OracleObserver

    env = gym.make("TargetExtraction-v0", disable_env_checker=True,
                   observer=OracleObserver(), obs_mode="state")
    try:
        env.reset(seed=3)
        base = env.unwrapped
        observation = base.decision_observation(refresh=True)
        validate_observation(observation)
        present = set(base.simulator.present_objects())
        expected = {
            (lower, upper)
            for upper, lowers in base.simulator.support_graph().items()
            if upper in present
            for lower in lowers
            if lower in present
        }
        got = {(r.source_id, r.target_id) for r in observation.relations.relations}
        assert got == expected
        assert expected, "the test pile must contain at least one object-on-object support"

        # After a removal the graph follows the new scene.
        removed = next(iter(present - {base.target_id}))
        env.step(base.object_ids.index(removed))
        after = base.decision_observation(refresh=True)
        assert removed not in {obj.object_id for obj in after.scene.objects}
        assert all(removed not in (r.source_id, r.target_id) for r in after.relations.relations)
    finally:
        env.close()


def test_oracle_profile_builds_oracle_observer():
    import json
    from physical_ai_mujoco.infrastructure.builder import ComponentBuilder
    from physical_ai_mujoco.observe import OracleObserver

    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "configs/experiments/oracolo.json").read_text(encoding="utf-8"))
    assert isinstance(ComponentBuilder().observer(profile["env_overrides"]), OracleObserver)
