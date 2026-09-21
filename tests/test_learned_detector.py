"""Adapter del detector appreso e metriche visive del benchmark.

Generazione del dataset e training sono testati in `test_vision_training.py`.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from physical_ai_mujoco.contracts import (
    ObjectUncertainty,
    Observation,
    PhysicalRelationState,
    SceneObject,
    SceneState,
    StereoFrame,
    UncertaintyState,
)
from physical_ai_mujoco.evaluation.observe_benchmark import (
    _interactive_profiles,
    _resolve_detector_weights,
    _target_exposure_curve,
    measure_observation,
)
from physical_ai_mujoco.sensors import LearnedDetector, SegmentationPrediction
from physical_ai_mujoco.sensors import ImageDisturbance, apply_image_disturbance


def _frame() -> StereoFrame:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    return StereoFrame(
        image,
        image.copy(),
        np.array([[20.0, 0, 15], [0, 20.0, 10], [0, 0, 1]]),
        np.eye(4),
        0.1,
        0.0,
        "world",
    )


def test_image_disturbance_is_seeded_and_preserves_monochrome_shape():
    image = np.full((20, 30), 120, dtype=np.uint8)
    policy = ImageDisturbance(blur_probability=1.0)
    first = apply_image_disturbance(image, policy, np.random.default_rng(9))
    second = apply_image_disturbance(image, policy, np.random.default_rng(9))
    np.testing.assert_array_equal(first, second)
    assert first.shape == image.shape and first.dtype == np.uint8
    assert not np.array_equal(first, image)
    with pytest.raises(ValueError, match="blur_kernel"):
        ImageDisturbance(blur_kernel=2)


class _Backend:
    def predict(self, image):
        obstacle = np.zeros(image.shape, dtype=bool)
        obstacle[4:9, 20:25] = True
        target = np.zeros(image.shape, dtype=bool)
        target[5:12, 3:10] = True
        rejected = np.zeros(image.shape, dtype=bool)
        rejected[1:3, 1:3] = True
        return (
            SegmentationPrediction(obstacle, 0, 0.8),
            SegmentationPrediction(target, 1, 0.95),
            SegmentationPrediction(rejected, 1, 0.1),
        )


def test_learned_detector_obeys_same_contract_without_torch_or_simulator():
    detections = LearnedDetector(backend=_Backend(), confidence_threshold=0.25).detect(_frame())
    assert detections.source == "learned_left_segmentation"
    assert len(detections.detections) == 2
    assert set(detections.type_ids.values()) == {"obstacle", "pfm_1_target"}
    assert max(detections.class_confidences.values()) == pytest.approx(0.95)
    assert detections.right_masks == {}


def test_learned_detector_reports_missing_weights_before_optional_import(tmp_path):
    with pytest.raises(FileNotFoundError, match="Pesi detector non trovati"):
        LearnedDetector(weights_path=tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="weights_path"):
        LearnedDetector()


def test_benchmark_matches_learned_track_ids_only_inside_evaluation():
    perceived = SceneObject(
        "learned_000", "target", "pfm_1_target",
        (0.01, 0.0, 0.1), None, "mesh", (0.06, 0.12, 0.02), True, 0.8,
    )
    observation = Observation(
        SceneState((perceived,), "learned_000", 0.0, None, "world", 1.0),
        PhysicalRelationState(("learned_000",), (), "test", False, "world", 1.0),
        UncertaintyState(
            (ObjectUncertainty("learned_000", True, 0.8, 0.8),),
            True, "test", "world", 1.0,
        ),
    )
    truth_description = SimpleNamespace(
        instance_id="object_007",
        type_id="pfm_1_target",
        shape="mesh",
        size={"x": 0.06, "y": 0.12, "z": 0.02},
    )
    simulator = SimpleNamespace(
        present_objects=lambda: ("object_007",),
        get_object_state=lambda _object_id: SimpleNamespace(position=(0.0, 0.0, 0.1)),
        scene=SimpleNamespace(objects=(truth_description,)),
        support_graph=lambda: {},
    )
    metrics = measure_observation(observation, simulator)
    assert metrics["matched_objects"] == 1
    assert metrics["target_detected"]
    assert metrics["observation_to_truth"] == {"learned_000": "object_007"}
    assert metrics["mean_position_error_m"] == pytest.approx(0.01)
    assert metrics["mean_size_ratio"] == pytest.approx(1.0)
    assert metrics["mean_volume_size_ratio"] == pytest.approx(1.0)
    assert metrics["target_size_ratio_mean"] == pytest.approx(1.0)
    assert metrics["target_volume_size_ratio"] == pytest.approx(1.0)


def test_target_matching_accepts_selected_duplicate_near_real_target():
    objects = (
        SceneObject(
            "learned_closest", "obstacle", "pfm_1_target",
            (0.0, 0.0, 0.1), None, "mesh", None, True, 0.7,
        ),
        SceneObject(
            "learned_selected", "target", "pfm_1_target",
            (0.02, 0.0, 0.1), None, "mesh", None, True, 0.9,
        ),
    )
    observation = Observation(
        SceneState(objects, "learned_selected", 0.0, None, "world", 1.0),
        PhysicalRelationState(tuple(item.object_id for item in objects), (), "test", False, "world", 1.0),
        UncertaintyState(
            tuple(ObjectUncertainty(item.object_id, True, 0.8, 0.8) for item in objects),
            True, "test", "world", 1.0,
        ),
    )
    truth = SimpleNamespace(
        instance_id="object_007", type_id="pfm_1_target", shape="mesh",
        size={"x": 0.06, "y": 0.12, "z": 0.02},
    )
    simulator = SimpleNamespace(
        present_objects=lambda: ("object_007",),
        get_object_state=lambda _object_id: SimpleNamespace(position=(0.0, 0.0, 0.1)),
        scene=SimpleNamespace(objects=(truth,)),
        support_graph=lambda: {},
    )
    metrics = measure_observation(observation, simulator)
    assert metrics["target_detected"]
    assert metrics["matched_target_truth_id"] == "object_007"
    assert metrics["object_precision"] == pytest.approx(0.5)


def test_exposure_curve_separates_fully_hidden_target():
    records = [
        {"metrics": {"target_camera_visible_fraction": 0.0, "target_visual_recognized": False}},
        {"metrics": {"target_camera_visible_fraction": 0.08, "target_visual_recognized": True}},
        {"metrics": {"target_camera_visible_fraction": 0.35, "target_visual_recognized": False}},
    ]
    curve = _target_exposure_curve(records)
    assert curve[0]["fully_hidden"] and curve[0]["scene_count"] == 1
    assert curve[1]["target_recall"] == 1.0
    assert curve[4]["target_recall"] == 0.0


def test_interactive_profiles_ignore_scene_rule_files(tmp_path):
    (tmp_path / "clean.json").write_text(
        '{"name":"Clean","scene_count":1,"object_count":3,"seed":42,'
        '"degraded":{},"stereo":{}}'
    )
    (tmp_path / "immersed_scene_rules.json").write_text(
        '{"schema_version":2,"ground_selection":{"type_id":"flat_standard"}}'
    )
    (tmp_path / "broken.json").write_text("{not-json")

    profiles = _interactive_profiles(tmp_path)

    assert [(path.name, payload["name"]) for path, payload in profiles] == [
        ("clean.json", "Clean")
    ]


def test_detector_weights_resolve_canonical_then_latest_version(tmp_path):
    older = tmp_path / "pfm_1_seg_20260101.pt"
    newer = tmp_path / "pfm_1_seg_immersed_v2.pt"
    older.write_bytes(b"older")
    newer.write_bytes(b"newer")
    older.touch()
    newer.touch()
    older_mtime = older.stat().st_mtime - 10
    older.touch()
    import os
    os.utime(older, (older_mtime, older_mtime))

    assert _resolve_detector_weights(None, tmp_path) == newer

    canonical = tmp_path / "pfm_1_seg.pt"
    canonical.write_bytes(b"canonical")
    assert _resolve_detector_weights(None, tmp_path) == canonical
    assert _resolve_detector_weights(newer, tmp_path) == newer
    with pytest.raises(FileNotFoundError, match="Pesi detector non trovati"):
        _resolve_detector_weights(tmp_path / "missing.pt", tmp_path)
