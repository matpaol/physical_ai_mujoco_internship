"""Regressioni per tracking, encoding, CAD e disturbi runtime."""

from types import SimpleNamespace

import numpy as np
import pytest

from physical_ai_mujoco.contracts import (
    ObjectUncertainty,
    Observation,
    PhysicalRelation,
    PhysicalRelationState,
    SceneObject,
    SceneState,
    UncertaintyState,
)
from physical_ai_mujoco.observe import CADMatcher, ObservationEncoder
from physical_ai_mujoco.observe import SensorObserver
from physical_ai_mujoco.sensors import (
    ImageDisturbance,
    ObjectTracker,
    SimulatedSensorSource,
    StereoDetections,
)
from physical_ai_mujoco.sensors.lidar import LidarConfig, LidarNoise


def _mask(x0, x1):
    result = np.zeros((30, 40), dtype=bool)
    result[8:16, x0:x1] = True
    return result


def test_tracker_ids_survive_detector_reordering_and_short_occlusion():
    tracker = ObjectTracker()
    first = tracker.update(
        StereoDetections(
            {"temporary_a": _mask(3, 9), "temporary_b": _mask(25, 31)},
            {},
            {"temporary_a": "obstacle", "temporary_b": "pfm_1_target"},
            "test",
        )
    )
    ids_by_class = {item.class_id: item.track_id for item in first.detections}
    tracker.update(StereoDetections({}, {}, None, "test"))
    reordered = tracker.update(
        StereoDetections(
            {"new_first": _mask(24, 30), "new_second": _mask(4, 10)},
            {},
            {"new_first": "pfm_1_target", "new_second": "obstacle"},
            "test",
        )
    )
    assert {item.class_id: item.track_id for item in reordered.detections} == ids_by_class


def _observation(object_ids):
    objects = tuple(
        SceneObject(
            object_id,
            "target" if index == 0 else "obstacle",
            "pfm_1_target" if index == 0 else "obstacle",
            (float(index), 0.0, 0.1),
            (1.0, 0.0, 0.0, 0.0),
            "mesh" if index == 0 else "box",
            (0.02, 0.06, 0.12),
            True,
            0.8,
        )
        for index, object_id in enumerate(object_ids)
    )
    relations = (
        (PhysicalRelation(object_ids[0], object_ids[1], "candidate_support", 0.7),)
        if len(object_ids) > 1
        else ()
    )
    return Observation(
        SceneState(objects, object_ids[0] if object_ids else None, 0.0, None, "world", 1.0),
        PhysicalRelationState(tuple(object_ids), relations, "test", True, "world", 1.0),
        UncertaintyState(
            tuple(ObjectUncertainty(item.object_id, True, 0.8, 0.7) for item in objects),
            False,
            "test",
            "world",
            1.0,
        ),
    )


def test_observation_encoder_has_fixed_shape_stable_slots_and_action_mask():
    encoder = ObservationEncoder(4)
    first = encoder.encode(_observation(("target", "rock")))
    second = encoder.encode(_observation(("rock",)))
    assert first.vector.shape == second.vector.shape == (encoder.vector_size,)
    assert first.vector.dtype == np.float32
    assert first.slot_ids[:2] == second.slot_ids[:2] == ("target", "rock")
    np.testing.assert_array_equal(first.action_mask[:2], (True, True))
    np.testing.assert_array_equal(second.action_mask[:2], (False, True))


def test_encoder_recycles_stale_slots_and_handles_spurious_detections():
    encoder = ObservationEncoder(2)
    first = encoder.encode(_observation(("target", "rock")))
    assert first.action_mask.sum() == 2
    noisy = encoder.encode(_observation(("rock", "spurious")))
    assert noisy.vector.shape == first.vector.shape
    assert set(noisy.slot_ids) == {"rock", "spurious"}
    assert noisy.action_mask.sum() == 2
    flooded = encoder.encode(_observation(("rock", "spurious", "another")))
    assert flooded.vector.shape == first.vector.shape
    assert flooded.vector[-1] == 1.0  # capacity overflow is explicitly uncertain


def test_noisy_detector_drop_and_false_positive_stay_inside_sensor_boundary():
    from physical_ai_mujoco.contracts import TaskContext
    from physical_ai_mujoco.sensors.noise import DetectionNoise, apply_detection_noise
    from tests.test_sensor_observer_boundary import _MaskDetector, _packet

    class NoisyDetector(_MaskDetector):
        def detect(self, frame):
            clean = super().detect(frame)
            masks = apply_detection_noise(
                clean.left_masks,
                DetectionNoise(drop_probability=1.0, false_positive_probability=1.0),
                np.random.default_rng(3),
            )
            return StereoDetections(masks, {}, {key: "obstacle" for key in masks}, "noise_test")

    observation = SensorObserver(NoisyDetector()).observe(
        _packet(), TaskContext(None, -1.0, target_type_id="pfm_1_target")
    )
    encoded = ObservationEncoder(2).encode(observation)
    assert encoded.vector.shape == (ObservationEncoder(2).vector_size,)
    assert observation.scene.target_id is None
    assert not any(item.is_target for item in observation.objects)


def test_sensor_policy_adapter_consumes_sensor_vector_only():
    from physical_ai_mujoco.infrastructure.policy_adapter import SensorPolicyAdapter

    class Model:
        observation_space = SimpleNamespace(shape=(3,))

        def predict(self, vector, deterministic):
            np.testing.assert_array_equal(vector, (2.0, 4.0, 6.0))
            assert deterministic
            return 1, None

    class Normalizer:
        def normalize_obs(self, vector):
            return vector * 2

    env = SimpleNamespace(unwrapped=SimpleNamespace(obs_mode="sensor"))
    adapter = SensorPolicyAdapter(Model(), Normalizer())
    assert adapter(env, {}, None, np.asarray((1.0, 2.0, 3.0))) == 1
    env.unwrapped.obs_mode = "state"
    with pytest.raises(ValueError, match="obs_mode"):
        adapter(env, {}, None, np.asarray((1.0, 2.0, 3.0)))


def test_cad_matcher_restores_full_size_from_partial_surface():
    rng = np.random.default_rng(7)
    model = np.column_stack(
        (
            rng.uniform(-0.03, 0.03, 1800),
            rng.uniform(-0.06, 0.06, 1800),
            rng.choice((-0.01, 0.01), 1800),
        )
    )
    matcher = CADMatcher(model, max_iterations=25)
    angle = 0.35
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    translation = np.asarray((0.1, -0.2, 0.3))
    transformed = model @ rotation.T + translation
    partial = transformed[transformed[:, 0] > 0.095]
    match = matcher.match(partial)
    assert sorted(match.geometry.size) == pytest.approx(
        sorted(matcher.nominal_size), abs=1e-12
    )
    assert np.linalg.norm(np.asarray(match.geometry.position) - translation) < 0.025
    assert match.rmse < 0.005


class _SensorStub:
    time = 3.0

    def render_stereo(self):
        image = np.zeros((10, 12, 3), dtype=np.uint8)
        return image, image.copy()

    def stereo_calibration(self):
        return np.asarray(((10.0, 0, 6), (0, 10.0, 5), (0, 0, 1))), np.eye(4), 0.1

    def raycast(self, origin, direction):
        return 1.0


def test_simulated_source_applies_seeded_mono_and_lidar_disturbances():
    source = SimulatedSensorSource(
        _SensorStub(),
        lidar_config=LidarConfig(n_rays=8, fov_horizontal_deg=90, max_range=2),
        image_disturbance=ImageDisturbance(
            contrast_range=(1.0, 1.0),
            brightness_range=(10.0, 10.0),
            gamma_range=(1.0, 1.0),
            noise_sigma_range=(0.0, 0.0),
            blur_probability=0.0,
        ),
        lidar_noise=LidarNoise(dropout_probability=1.0),
        seed=11,
    )
    packet = source.capture()
    assert np.all(packet.stereo.gray_left == 10)
    np.testing.assert_array_equal(
        packet.stereo.rgb_left[:, :, 0], packet.stereo.rgb_left[:, :, 2]
    )
    assert not packet.lidar.valid.any()


def test_gym_sensor_mode_exposes_fixed_vector_and_rejects_unmatched_action():
    import gymnasium as gym
    import physical_ai_mujoco.envs  # noqa: F401
    from tests.harness.fake_sensor_source import FakeSensorSource
    from tests.test_sensor_observer_boundary import _MaskDetector, _packet

    env = gym.make(
        "TargetExtraction-v0",
        disable_env_checker=True,
        object_count=2,
        obs_mode="sensor",
        observer=SensorObserver(_MaskDetector()),
        sensor_source=FakeSensorSource(_packet()),
    )
    try:
        vector, info = env.reset(seed=3)
        assert env.observation_space.contains(vector)
        assert vector.shape == env.observation_space.shape
        action = int(np.flatnonzero(info["action_mask"])[0])
        _, reward, terminated, truncated, result = env.step(action)
        assert result["invalid_action"]
        assert reward < 0
        assert not terminated and not truncated
    finally:
        env.close()
