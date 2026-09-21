"""Sorgenti sostituibili che acquisiscono stereo e LiDAR sincronizzati."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from physical_ai_mujoco.contracts import (
    SensorCalibration,
    SynchronizedSensorPacket,
)

from .capture import SimulatedStereoCamera
from .disturbance import ImageDisturbance, apply_image_disturbance
from .lidar import LidarConfig, LidarNoise, apply_lidar_noise, scan


class SensorSource(ABC):
    """API pubblica comune a simulazione, registrazioni e sensori reali."""

    @abstractmethod
    def capture(self) -> SynchronizedSensorPacket:
        raise NotImplementedError


class SimulatedSensorSource(SensorSource):
    """Acquisisce misure da MuJoCo senza aggiungere etichette privilegiate."""

    def __init__(
        self,
        simulator_or_provider,
        *,
        stereo_camera: SimulatedStereoCamera | None = None,
        lidar_config: LidarConfig,
        image_disturbance: ImageDisturbance | None = None,
        lidar_noise: LidarNoise | None = None,
        seed: int = 0,
    ):
        self._simulator_or_provider = simulator_or_provider
        self.stereo_camera = stereo_camera or SimulatedStereoCamera(with_depth=False)
        self.lidar_config = lidar_config
        self.image_disturbance = image_disturbance
        self.lidar_noise = lidar_noise
        self.reset(seed)

    def reset(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def _disturb_stereo(self, stereo):
        if self.image_disturbance is None:
            return stereo
        left = apply_image_disturbance(
            stereo.gray_left, self.image_disturbance, self._rng
        )
        right = apply_image_disturbance(
            stereo.gray_right, self.image_disturbance, self._rng
        )
        return replace(
            stereo,
            rgb_left=np.repeat(left[:, :, None], 3, axis=2),
            rgb_right=np.repeat(right[:, :, None], 3, axis=2),
        )

    def _simulator(self):
        candidate = self._simulator_or_provider
        simulator = candidate() if isinstance(candidate, Callable) else candidate
        if simulator is None:
            raise RuntimeError("Il simulatore non e' ancora disponibile")
        return simulator

    def capture(self) -> SynchronizedSensorPacket:
        simulator = self._simulator()
        stereo = self._disturb_stereo(self.stereo_camera.capture(simulator))
        lidar = scan(simulator, self.lidar_config)
        if self.lidar_noise is not None:
            lidar = apply_lidar_noise(lidar, self.lidar_noise, self._rng)
        world_from_lidar = np.eye(4)
        world_from_lidar[:3, :3] = np.asarray(self.lidar_config.rotation, dtype=float)
        world_from_lidar[:3, 3] = np.asarray(self.lidar_config.origin, dtype=float)
        height, width = stereo.gray_left.shape
        calibration = SensorCalibration(
            intrinsics=stereo.intrinsics,
            world_from_left=stereo.world_from_left,
            world_from_lidar=world_from_lidar,
            baseline=stereo.baseline,
            image_height=height,
            image_width=width,
        )
        return SynchronizedSensorPacket(
            stereo=stereo,
            lidar=lidar,
            calibration=calibration,
            synchronization_error=abs(stereo.timestamp - lidar.timestamp),
        )
