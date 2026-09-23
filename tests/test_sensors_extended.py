"""Contratti e integrazione dei sensori simulati con OSSERVA."""

import json
from pathlib import Path
import sys

# Con `python tests/test_*.py`, importa il progetto che contiene questo test.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from physical_ai_mujoco.contracts import TaskContext
from physical_ai_mujoco.evaluation.observe_benchmark import run_benchmark
from physical_ai_mujoco.observe import StereoObserver
from physical_ai_mujoco.sensors import (
    BundleBuilder, CalibrationNoise, DepthNoise, DetectionNoise, Detector,
    OracleDetector, SimulatedStereoCamera, StereoDetections,
    StereoRig,
)
from physical_ai_mujoco.sensors.calibration import estimate_extrinsics, perturb_rig
from physical_ai_mujoco.sensors.cloud import estimate_geometry, mask_to_point_cloud
from physical_ai_mujoco.sensors.capture import StereoFrame
from physical_ai_mujoco.evaluation.sensor_calibration import calibration_curve, compare_lidar_depth
from physical_ai_mujoco.sensors.lidar import (
    LidarConfig, LidarNoise, apply_lidar_noise, scan,
)
from physical_ai_mujoco.sensors.noise import apply_depth_noise, apply_detection_noise


def _rig():
    return StereoRig(np.array([[100., 0, 20], [0, 100, 10], [0, 0, 1]]),
                     np.eye(4), 0.2, 20, 40)


def test_rig_projects_and_unprojects_both_eyes(tmp_path):
    rig = _rig()
    point = np.array([0.1, 0.04, 2.0])
    for eye in ("left", "right"):
        u, v, z = rig.project(point, eye)
        np.testing.assert_allclose(rig.unproject(u, v, z, eye), point, atol=1e-12)
    assert np.linalg.norm(rig.world_from_eye("right")[:3, 3] -
                          rig.world_from_eye("left")[:3, 3]) == pytest.approx(0.2)
    path = tmp_path / "rig.json"
    path.write_text(json.dumps({"intrinsics": rig.intrinsics.tolist(),
                                "world_from_left": rig.world_from_left.tolist(),
                                "baseline": 0.2, "height": 20, "width": 40}), encoding="utf-8")
    assert StereoRig.from_calibration_file(path).project(point) == pytest.approx(rig.project(point))
    with pytest.raises(ValueError, match="dietro"):
        rig.project((0, 0, -1))


def test_mask_to_cloud_known_pixel_and_invalid_depth():
    depth = np.full((3, 3), np.nan)
    depth[1, 1] = 2.0
    mask = np.ones((3, 3), dtype=bool)
    k = np.array([[2., 0, 1], [0, 2, 1], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, 3] = (1, 0, 0)
    np.testing.assert_allclose(mask_to_point_cloud(depth, k, pose, mask), [[1, 0, 2]])
    assert mask_to_point_cloud(depth, k, pose, np.zeros_like(mask)).shape == (0, 3)
    with pytest.raises(ValueError, match="Dimensioni"):
        mask_to_point_cloud(depth, k, pose, np.zeros((2, 2)))


def test_geometry_on_full_known_box_and_bad_input():
    corners = np.array([(x, y, z) for x in (-1., 1.) for y in (-.5, .5)
                        for z in (-.25, .25)])
    geometry = estimate_geometry(corners, shape_hint="box")
    np.testing.assert_allclose(geometry.position, (0, 0, 0))
    np.testing.assert_allclose(sorted(geometry.size), (.5, 1., 2.), atol=1e-12)
    assert geometry.point_count == 8
    with pytest.raises(ValueError, match="sei"):
        estimate_geometry(corners[:3])


def test_geometry_shape_classification_on_complete_synthetic_surfaces():
    rng = np.random.default_rng(0)
    sphere = rng.normal(size=(1000, 3))
    sphere /= np.linalg.norm(sphere, axis=1)[:, None]
    assert estimate_geometry(sphere).shape == "sphere"
    angles = rng.uniform(0, 2*np.pi, 1000)
    cylinder = np.column_stack((np.cos(angles), np.sin(angles), rng.uniform(-1, 1, 1000)))
    assert estimate_geometry(cylinder).shape == "cylinder"
    assert estimate_geometry(rng.uniform(-1, 1, (1000, 3))).shape == "box"


def test_frame_rejects_nonmonochrome_and_bad_depth():
    good = np.zeros((3, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="monocromatica"):
        StereoFrame(good, np.tile(np.array([1, 2, 3], dtype=np.uint8), (3, 4, 1)),
                    np.eye(3), np.eye(4), .1, 0., "world")
    with pytest.raises(ValueError, match="Depth"):
        StereoFrame(good, good, np.eye(3), np.eye(4), .1, 0., "world",
                    depth_left=np.ones((2, 4)))


class _FakeDetector(Detector):
    def detect(self, frame):
        mask = np.zeros(frame.gray_left.shape, dtype=bool)
        mask[1:4, 1:4] = True
        return StereoDetections({"track_a": mask}, {"track_a": mask}, None, "fake")


def test_detector_is_substitutable_without_simulator():
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    frame = StereoFrame(image, image, _rig().intrinsics, np.eye(4), .2, 1., "world")
    bundle = BundleBuilder(_FakeDetector(), seed=3).build(frame)
    assert bundle.detection_source == "fake"
    assert list(bundle.left_masks) == ["track_a"]
    assert bundle.type_ids is None


def test_detector_bad_mask_raises_at_boundary():
    class BadDetector(Detector):
        def detect(self, frame):
            return StereoDetections({"bad": np.ones((2, 2))}, {}, None, "fake")
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    frame = StereoFrame(image, image, _rig().intrinsics, np.eye(4), .2, 1., "world")
    with pytest.raises(ValueError, match="non allineata"):
        BundleBuilder(BadDetector()).build(frame)


def test_detection_and_depth_noise_seeded_and_validated():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    policy = DetectionNoise(edge_erosion_px=1)
    first = apply_detection_noise({"a": mask}, policy, np.random.default_rng(4))
    second = apply_detection_noise({"a": mask}, policy, np.random.default_rng(4))
    np.testing.assert_array_equal(first["a"], second["a"])
    assert first["a"].sum() < mask.sum()
    assert apply_detection_noise({"a": mask}, DetectionNoise(drop_probability=1),
                                 np.random.default_rng(1)) == {}
    image = np.ones((20, 20))
    noisy = apply_depth_noise(image, DepthNoise(distance_sigma=.01, dropout_probability=.1),
                              np.random.default_rng(5))
    assert np.isnan(noisy).any()
    assert np.nanmean(noisy) == pytest.approx(1., abs=.01)
    with pytest.raises(ValueError):
        DetectionNoise(drop_probability=1.1)


def test_calibration_perturbation_and_correspondence_estimate():
    rig = _rig()
    policy = CalibrationNoise(translation_sigma=.01, rotation_sigma_deg=.5,
                              baseline_sigma=.001)
    a = perturb_rig(rig, policy, np.random.default_rng(5))
    b = perturb_rig(rig, policy, np.random.default_rng(5))
    np.testing.assert_array_equal(a.world_from_left, b.world_from_left)
    assert a.baseline == b.baseline
    source = np.random.default_rng(2).normal(size=(30, 3))
    target = source + np.array([.2, -.1, .3])
    target[0] += 10  # corrispondenza sbagliata
    pose = estimate_extrinsics(source, target, trim_fraction=.1)
    np.testing.assert_allclose(pose[:3, 3], (.2, -.1, .3), atol=1e-6)
    with pytest.raises(ValueError, match="degeneri"):
        estimate_extrinsics(np.zeros((4, 3)), np.ones((4, 3)))


class _WallSimulator:
    time = 1.25

    def raycast(self, origin, direction):
        return 2 / direction[0] if direction[0] > 0 else None


def test_lidar_pattern_hits_noise_and_json():
    frame = scan(_WallSimulator(), LidarConfig(n_rays=8, n_planes=2,
                 fov_horizontal_deg=180, fov_vertical_deg=20, max_range=5))
    assert len(frame.ranges) == 16
    assert 0 < frame.valid.sum() < 16
    assert np.isnan(frame.points[~frame.valid]).all()
    json.dumps(frame.to_dict(), allow_nan=False)
    policy = LidarNoise(distance_sigma=.01, angle_sigma_deg=.1,
                        dropout_probability=.2, range_dependent=False)
    a = apply_lidar_noise(frame, policy, np.random.default_rng(5))
    b = apply_lidar_noise(frame, policy, np.random.default_rng(5))
    np.testing.assert_array_equal(a.valid, b.valid)
    np.testing.assert_allclose(a.ranges, b.ranges)
    with pytest.raises(ValueError, match="Pattern"):
        LidarConfig(n_rays=0)


def _environment():
    import gymnasium as gym
    import physical_ai_mujoco.envs  # noqa: F401
    return gym.make("TargetExtraction-v0", object_count=3, disable_env_checker=True)


def test_depth_rig_detector_and_observer_on_mujoco_scene():
    env = _environment()
    try:
        env.reset(seed=5)
        simulator = env.unwrapped.simulator
        rig = StereoRig.from_scene_description(simulator.scene.stereo_camera)
        k, pose, baseline = simulator.stereo_calibration()
        np.testing.assert_allclose(rig.intrinsics, k)
        np.testing.assert_allclose(rig.world_from_left, pose, atol=1e-8)
        assert rig.baseline == baseline
        frame = SimulatedStereoCamera(with_depth=True).capture(simulator)
        assert frame.gray_left.shape == frame.depth_left.shape == (240, 320)
        assert frame.gray_left.dtype == np.uint8
        assert np.isfinite(frame.depth_left).any()
        assert frame.depth_left[120, 160] > 0
        detector = OracleDetector(simulator)
        bundle = BundleBuilder(detector, seed=4).build(frame)
        assert bundle.detection_source == "mujoco_oracle"
        assert set(bundle.left_masks) <= set(simulator.present_objects())
        context = TaskContext(env.unwrapped.target_id, 0.0)
        observation = StereoObserver(reconstruction_mode="depth").observe(bundle, context)
        assert observation.objects and observation.relations.available
        assert all(obj.size is not None for obj in observation.objects)
        stereo = StereoObserver(reconstruction_mode="stereo").observe(bundle, context)
        assert not stereo.relations.available
        lidar = scan(simulator, LidarConfig(n_rays=24, n_planes=2,
                     fov_vertical_deg=20, origin=(0, -.5, .1)))
        assert lidar.valid.any()
        assert lidar.timestamp == simulator.time
    finally:
        env.close()


def test_observer_mode_is_explicit_and_depth_mode_is_strict():
    with pytest.raises(ValueError, match="reconstruction_mode"):
        StereoObserver(reconstruction_mode="automatic")
    env = _environment()
    try:
        env.reset(seed=5)
        simulator = env.unwrapped.simulator
        frame = SimulatedStereoCamera(with_depth=False).capture(simulator)
        bundle = BundleBuilder(OracleDetector(simulator)).build(frame)
        context = TaskContext(env.unwrapped.target_id, 0.0)
        with pytest.raises(ValueError, match="richiede depth"):
            StereoObserver(reconstruction_mode="depth").observe(bundle, context)
    finally:
        env.close()


def test_rgbd_benchmark_reports_size_and_supports():
    config = {"scene_count": 1, "object_count": 3, "seed": 5,
              "degraded": {"position_sigma": 0, "drop_probability": 0},
              "stereo": {"centroid_noise_px": 0, "drop_probability": 0}}
    report = run_benchmark(config, ("rgbd",))
    metrics = report["scenes"][0]["metrics"]
    assert metrics["invariants_ok"]
    assert metrics["mean_size_error_m"] is not None
    assert metrics["relations_available"]
    assert report["rgbd_depth_source"].startswith("MuJoCo")


def test_calibration_curve_and_lidar_depth_diagnostic():
    config = {"scene_count": 1, "object_count": 3, "seed": 5,
              "degraded": {"position_sigma": 0, "drop_probability": 0},
              "stereo": {"centroid_noise_px": 0, "drop_probability": 0}}
    curve = calibration_curve(config, (0., .01), (0.,))
    assert len(curve["records"]) == 2
    assert curve["records"][0]["position_error_m"] is not None
    assert all(record["invariant_violations"] == 0 for record in curve["records"])
    env = _environment()
    try:
        env.reset(seed=5)
        sim = env.unwrapped.simulator
        frame = SimulatedStereoCamera(with_depth=True).capture(sim)
        rig = StereoRig(frame.intrinsics, frame.world_from_left, frame.baseline,
                        *frame.depth_left.shape)
        lidar = scan(sim, LidarConfig(n_rays=90, n_planes=5,
                     fov_horizontal_deg=120, fov_vertical_deg=45,
                     origin=(0, -.55, .25),
                     rotation=((0, -1, 0), (1, 0, 0), (0, 0, 1)), max_range=2))
        comparison = compare_lidar_depth(lidar, frame.depth_left, rig)
        assert comparison["matches"] > 0
        assert comparison["median_abs_depth_error_m"] < .05
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
