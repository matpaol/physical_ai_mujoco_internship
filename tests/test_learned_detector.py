"""Dataset sintetico e adapter del detector appreso."""

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
from physical_ai_mujoco.experiments.synthetic_dataset import (
    DatasetGenerationConfig,
    _prepare_directory,
    _object_count_for_scene,
    _validation_indices,
    _write_sample,
    mask_to_yolo_segments,
)
from physical_ai_mujoco.experiments.train_detector import DetectorTrainingConfig
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


def test_mask_export_is_normalized_and_dataset_split_is_by_scene():
    mask = np.zeros((10, 20), dtype=bool)
    mask[2:8, 4:16] = True
    segments = mask_to_yolo_segments(mask)
    assert len(segments) == 1
    assert len(segments[0]) >= 8
    assert all(0 <= value <= 1 for value in segments[0])

    config = DatasetGenerationConfig(scene_count=10, validation_fraction=0.2, seed=7)
    first = _validation_indices(config)
    assert first == _validation_indices(config)
    assert len(first) == 2
    assert first < set(range(config.scene_count))
    variable = DatasetGenerationConfig(scene_count=5, object_count=(2, 6), seed=7)
    counts = [_object_count_for_scene(variable, index) for index in range(5)]
    assert counts == [_object_count_for_scene(variable, index) for index in range(5)]
    assert all(2 <= value <= 6 for value in counts)
    with pytest.raises(ValueError, match="target_immersion_range"):
        DatasetGenerationConfig(target_immersion_range=(0.8, 0.2))


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


def test_lossless_masks_json_and_yolo_labels_are_written_together(tmp_path):
    destination = tmp_path / "dataset"
    _prepare_directory(destination)
    target = np.zeros((12, 16), dtype=bool)
    target[2:8, 3:10] = True
    obstacle = np.zeros_like(target)
    obstacle[7:11, 10:15] = True
    annotation = _write_sample(
        destination,
        "train",
        "sample",
        np.zeros(target.shape, dtype=np.uint8),
        {"target": target, "other": obstacle},
        {"target": "pfm_1_target", "other": "slab"},
        "pfm_1_target",
        3,
    )
    assert (destination / "images/train/sample.png").is_file()
    assert all((destination / item["mask"]).is_file() for item in annotation["instances"])
    labels = (destination / "labels/train/sample.txt").read_text().splitlines()
    assert {line.split()[0] for line in labels} == {"0", "1"}
    with pytest.raises(FileExistsError, match="non e' vuota"):
        _prepare_directory(destination)


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
    with pytest.raises(FileNotFoundError, match="Dataset YAML"):
        DetectorTrainingConfig(
            dataset_yaml=tmp_path / "missing.yaml",
            destination=tmp_path / "weights.pt",
        ).validate()


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
