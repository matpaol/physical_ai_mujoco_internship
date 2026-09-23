"""La sorgente simulata fornisce maschere visibili senza usare pose in OSSERVA."""

from dataclasses import replace
import json
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from physical_ai_mujoco.contracts import TaskContext
from physical_ai_mujoco.evaluation.observe_benchmark import export_graphs, main, run_benchmark
from physical_ai_mujoco.observe import ExactObserver, StereoObserver
from physical_ai_mujoco.sensors import BundleBuilder, DetectionNoise, OracleDetector, SimulatedStereoCamera


def _environment(baseline=None):
    import gymnasium as gym
    import physical_ai_mujoco.envs  # noqa: F401

    return gym.make(
        "TargetExtraction-v0", object_count=3, disable_env_checker=True,
        stereo_baseline=baseline,
    )


def test_segmentation_masks_feed_stereo_observer_and_exclude_ground():
    env = _environment()
    try:
        env.reset(seed=5)
        simulator = env.unwrapped.simulator
        frame = SimulatedStereoCamera().capture(simulator)
        assert not hasattr(frame, "left_masks")
        assert not hasattr(frame, "type_ids")
        bundle = BundleBuilder(OracleDetector(simulator)).build(frame)
        assert bundle.rgb_left.shape == bundle.rgb_right.shape == (240, 320, 3)
        for image in (bundle.rgb_left, bundle.rgb_right):
            np.testing.assert_array_equal(image[:, :, 0], image[:, :, 1])
            np.testing.assert_array_equal(image[:, :, 1], image[:, :, 2])
        assert bundle.intrinsics.shape == (3, 3)
        assert bundle.world_from_left.shape == (4, 4)
        assert set(bundle.left_masks) <= set(simulator.present_objects())
        assert set(bundle.right_masks) <= set(simulator.present_objects())
        assert bundle.left_masks and bundle.right_masks
        for masks in (bundle.left_masks, bundle.right_masks):
            occupied = np.zeros(bundle.rgb_left.shape[:2], dtype=int)
            for mask in masks.values():
                assert mask.shape == occupied.shape and mask.any()
                occupied += mask
            assert occupied.max() == 1
        observation = StereoObserver().observe(
            bundle, TaskContext(env.unwrapped.target_id, 0.0)
        )
        assert observation.objects
        assert set(item.object_id for item in observation.objects) <= set(simulator.present_objects())
        assert all(np.isfinite(item.position).all() for item in observation.objects)
        assert not observation.relations.available
    finally:
        env.close()


def test_baseline_override_moves_rendered_cameras_and_calibration():
    import mujoco

    baseline = 0.23
    env = _environment(baseline)
    try:
        env.reset(seed=5)
        simulator = env.unwrapped.simulator
        bundle = BundleBuilder(OracleDetector(simulator)).build(SimulatedStereoCamera().capture(simulator))
        left_id = mujoco.mj_name2id(
            simulator.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_left"
        )
        right_id = mujoco.mj_name2id(
            simulator.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_right"
        )
        actual_distance = np.linalg.norm(
            simulator.data.cam_xpos[left_id] - simulator.data.cam_xpos[right_id]
        )
        np.testing.assert_allclose(actual_distance, baseline, atol=1e-9)
        assert bundle.baseline == baseline
    finally:
        env.close()


def test_stereo_degradation_is_seeded_and_does_not_change_rgb():
    env = _environment()
    try:
        env.reset(seed=5)
        builder = BundleBuilder(
            OracleDetector(env.unwrapped.simulator),
            DetectionNoise(centroid_noise_px=2.0, drop_probability=0.2),
        )
        frame = SimulatedStereoCamera().capture(env.unwrapped.simulator)
        builder.reset(123)
        first = builder.build(frame)
        builder.reset(123)
        second = builder.build(frame)
        np.testing.assert_array_equal(first.rgb_left, second.rgb_left)
        np.testing.assert_array_equal(first.rgb_right, second.rgb_right)
        assert first.left_masks.keys() == second.left_masks.keys()
        assert first.right_masks.keys() == second.right_masks.keys()
        for key in first.left_masks:
            np.testing.assert_array_equal(first.left_masks[key], second.left_masks[key])
        for key in first.right_masks:
            np.testing.assert_array_equal(first.right_masks[key], second.right_masks[key])

        empty = BundleBuilder(OracleDetector(env.unwrapped.simulator), DetectionNoise(drop_probability=1.0))
        bundle = empty.build(frame)
        assert bundle.left_masks == bundle.right_masks == {}
        observation = StereoObserver().observe(
            bundle, TaskContext(env.unwrapped.target_id, 0.0)
        )
        assert observation.objects == ()
        assert not observation.relations.available
        assert observation.uncertainty.unknown_space
    finally:
        env.close()


def test_benchmark_compares_all_sources_without_decide_or_execute(tmp_path):
    config = {
        "scene_count": 1,
        "object_count": 3,
        "seed": 5,
        "degraded": {"position_sigma": 0.0, "drop_probability": 0.0},
        "stereo": {"centroid_noise_px": 0.0, "drop_probability": 0.0},
    }
    report = run_benchmark(config)
    assert [item["mode"] for item in report["scenes"]] == ["exact", "degraded", "stereo"]
    assert len({item["scene_seed"] for item in report["scenes"]}) == 1
    assert report["summary"]["exact"]["metrics"]["mean_position_error_m"]["mean"] == 0
    assert report["summary"]["degraded"]["metrics"]["mean_position_error_m"]["mean"] == 0
    assert report["summary"]["stereo"]["relation_unavailable_scenes"] == 1
    assert all(item["metrics"]["invariants_ok"] for item in report["scenes"])
    assert "segmentation oracle" in report["stereo_identity_source"]
    files = export_graphs(report, tmp_path)
    assert any(path.suffix == ".dot" for path in files)
    exact_graph = (tmp_path / "scene_000_exact.dot").read_text(encoding="utf-8")
    stereo_graph = (tmp_path / "scene_000_stereo.dot").read_text(encoding="utf-8")
    assert "Contatti MuJoCo" in exact_graph
    assert "Relazioni non disponibili" in stereo_graph


def test_benchmark_random_object_count_is_shared_and_reproducible():
    config = {
        "scene_count": 3,
        "object_count": [2, 4],
        "seed": 17,
        "degraded": {"position_sigma": 0.0, "drop_probability": 0.0},
        "stereo": {"centroid_noise_px": 0.0, "drop_probability": 0.0},
    }
    first = run_benchmark(config, ("exact", "stereo"))
    second = run_benchmark(config, ("exact", "stereo"))
    for scene_index in range(3):
        records = [item for item in first["scenes"] if item["scene_index"] == scene_index]
        assert len(records) == 2
        count = records[0]["object_count"]
        assert 2 <= count <= 4
        assert records[1]["object_count"] == count
        assert records[0]["scene_seed"] == records[1]["scene_seed"]
        assert records[0]["metrics"]["true_objects"] == count
        assert records[0]["metrics"]["object_uncertainty"]
    assert [item["object_count"] for item in first["scenes"]] == [
        item["object_count"] for item in second["scenes"]
    ]
    assert [item["scene_seed"] for item in first["scenes"]] == [
        item["scene_seed"] for item in second["scenes"]
    ]
    assert all(item["metrics"]["invariants_ok"] for item in first["scenes"])


def test_benchmark_records_invariant_failure_and_continues(monkeypatch, tmp_path):
    original = ExactObserver.observe
    calls = 0

    def first_result_has_wrong_frame(self, source, target_id):
        nonlocal calls
        calls += 1
        observation = original(self, source, target_id)
        if calls == 1:
            return replace(
                observation,
                relations=replace(observation.relations, frame="camera"),
            )
        return observation

    monkeypatch.setattr(ExactObserver, "observe", first_result_has_wrong_frame)
    config = {
        "scene_count": 2, "object_count": 2, "seed": 5,
        "degraded": {"position_sigma": 0.0, "drop_probability": 0.0},
        "stereo": {"centroid_noise_px": 0.0, "drop_probability": 0.0},
    }
    report = run_benchmark(config, ("exact",))
    failed, valid = report["scenes"]
    assert failed["metrics"]["invariants_ok"] is False
    assert failed["metrics"]["invariant_violations"] == ["Frame non allineati"]
    assert "object_recall" not in failed["metrics"]
    assert valid["metrics"]["invariants_ok"] is True
    assert report["summary"]["exact"]["invariant_violations"] == 1
    assert report["summary"]["exact"]["metrics"]["object_recall"]["scene_count"] == 1
    files = export_graphs(report, tmp_path)
    assert len([path for path in files if path.suffix == ".dot"]) == 2
    assert "Observation non valida" in (tmp_path / "scene_000_exact.dot").read_text(encoding="utf-8")


def test_benchmark_does_not_reclassify_other_errors_as_invariants(monkeypatch):
    def broken_source(*_args, **_kwargs):
        raise ValueError("errore di acquisizione")

    monkeypatch.setattr(ExactObserver, "observe", broken_source)
    config = {
        "scene_count": 1, "object_count": 2, "seed": 5,
        "degraded": {"position_sigma": 0.0, "drop_probability": 0.0},
        "stereo": {"centroid_noise_px": 0.0, "drop_probability": 0.0},
    }
    with pytest.raises(ValueError, match="errore di acquisizione"):
        run_benchmark(config, ("exact",))


def test_cli_saves_invalid_report_and_returns_nonzero(monkeypatch, tmp_path):
    original = ExactObserver.observe

    def wrong_frame(self, source, target_id):
        observation = original(self, source, target_id)
        return replace(
            observation,
            relations=replace(observation.relations, frame="camera"),
        )

    monkeypatch.setattr(ExactObserver, "observe", wrong_frame)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "scene_count": 1, "object_count": 2, "seed": 5,
        "degraded": {"position_sigma": 0.0, "drop_probability": 0.0},
        "stereo": {"centroid_noise_px": 0.0, "drop_probability": 0.0},
    }), encoding="utf-8")
    output_path = tmp_path / "report.json"
    assert main([
        "--config", str(config_path), "--mode", "exact", "--headless",
        "--output", str(output_path),
    ]) == 1
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["scenes"][0]["metrics"]["invariant_violations"] == ["Frame non allineati"]
    assert saved["summary"]["exact"]["invariant_violations"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
