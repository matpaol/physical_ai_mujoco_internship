"""Verifica il confine deployable stereo + LiDAR -> Observation."""

import numpy as np
import pytest

from physical_ai_mujoco.contracts import (
    LidarFrame,
    SensorCalibration,
    StereoFrame,
    SynchronizedSensorPacket,
    TaskContext,
)
from physical_ai_mujoco.observe import SensorObserver
from physical_ai_mujoco.sensors import Detector, SimulatedSensorSource, StereoDetections
from physical_ai_mujoco.sensors.lidar import LidarConfig
from tests.harness.fake_sensor_source import FakeSensorSource


def _packet() -> SynchronizedSensorPacket:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    intrinsics = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    pose = np.eye(4)
    stereo = StereoFrame(image, image.copy(), intrinsics, pose, 0.1, 2.0, "world")
    points = np.array(
        [
            (x, y, z)
            for x in (-0.04, 0.04)
            for y in (-0.03, 0.03)
            for z in (0.98, 1.02)
        ],
        dtype=float,
    )
    ranges = np.linalg.norm(points, axis=1)
    directions = points / ranges[:, None]
    lidar = LidarFrame(
        points, ranges, directions, np.ones(len(points), dtype=bool),
        2.0, "world", (0.0, 0.0, 0.0),
    )
    calibration = SensorCalibration(
        intrinsics, pose, np.eye(4), 0.1, 100, 100
    )
    return SynchronizedSensorPacket(stereo, lidar, calibration, 0.0)


class _MaskDetector(Detector):
    def detect(self, frame):
        mask = np.zeros(frame.gray_left.shape, dtype=bool)
        mask[40:61, 40:61] = True
        return StereoDetections(
            {"track-pfm": mask},
            {},
            {"track-pfm": "pfm_1_target"},
            "fake_segmenter",
            {"track-pfm": 0.9},
        )


def test_sensor_observer_builds_target_without_simulator_or_target_instance_id():
    source = FakeSensorSource(_packet())
    observer = SensorObserver(_MaskDetector())
    context = TaskContext(
        target_id=None,
        ground_height=-1.0,
        bounds=(-1.0, 1.0, -1.0, 1.0),
        target_type_id="pfm_1_target",
    )

    observation = observer.observe(source.capture(), context)

    assert source.capture_count == 1
    assert observation.scene.target_id == "track-pfm"
    assert len(observation.objects) == 1
    target = observation.objects[0]
    assert target.is_target
    assert target.type_id == "pfm_1_target"
    assert target.shape == "mesh"
    assert target.position == pytest.approx((0.0, 0.0, 1.0))
    assert sorted(target.size) == pytest.approx((0.04, 0.06, 0.08))
    assert target.perception_quality is not None
    assert observation.relations.available


def test_sensor_observer_does_not_invent_geometry_without_enough_lidar_points():
    packet = _packet()
    sparse = packet.lidar.valid.copy()
    sparse[5:] = False
    ranges = packet.lidar.ranges.copy()
    ranges[~sparse] = np.inf
    points = packet.lidar.points.copy()
    points[~sparse] = np.nan
    packet = SynchronizedSensorPacket(
        packet.stereo,
        LidarFrame(
            points, ranges, packet.lidar.directions, sparse,
            packet.lidar.timestamp, packet.lidar.frame, packet.lidar.origin,
        ),
        packet.calibration,
        0.0,
    )
    observation = SensorObserver(_MaskDetector()).observe(
        packet,
        TaskContext(None, -1.0, target_type_id="pfm_1_target"),
    )
    assert observation.objects == ()
    assert observation.scene.target_id is None
    assert not observation.relations.available


class _SensorStub:
    time = 3.0

    def render_stereo(self):
        image = np.zeros((10, 12, 3), dtype=np.uint8)
        return image, image.copy()

    def stereo_calibration(self):
        return (
            np.array([[10.0, 0, 6], [0, 10.0, 5], [0, 0, 1]]),
            np.eye(4),
            0.1,
        )

    def raycast(self, origin, direction):
        return None


def test_simulated_source_returns_only_raw_synchronized_measurements():
    source = SimulatedSensorSource(
        _SensorStub(),
        lidar_config=LidarConfig(
            n_rays=4,
            n_planes=1,
            fov_horizontal_deg=90,
            max_range=2,
        ),
    )
    packet = source.capture()
    assert packet.synchronization_error == 0
    assert packet.stereo.gray_left.shape == (10, 12)
    assert packet.lidar.valid.sum() == 0
    assert not hasattr(packet, "type_ids")
    assert not hasattr(packet, "instance_masks")


def test_livox_configuration_is_loaded_without_hardcoded_source_parameters():
    from pathlib import Path

    path = Path(__file__).parents[1] / "configs/sensors/livox_avia.json"
    config = LidarConfig.from_file(path)
    assert config.fov_horizontal_deg == pytest.approx(70.4)
    assert config.fov_vertical_deg == pytest.approx(77.2)
    assert config.n_rays * config.n_planes == 23040


def test_packet_rejects_inconsistent_timestamps():
    packet = _packet()
    with pytest.raises(ValueError, match="sincronizzazione"):
        SynchronizedSensorPacket(
            packet.stereo, packet.lidar, packet.calibration, 0.01
        )
