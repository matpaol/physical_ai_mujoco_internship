"""Contratti condivisi per acquisizione e riconoscimento sensoriale.

Le implementazioni simulate e reali producono questi tipi; nessun contratto
qui dipende da MuJoCo, ROS 2 o da un particolare detector.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StereoFrame:
    """Coppia stereo monocromatica esposta come HxWx3 per compatibilita'."""

    rgb_left: np.ndarray
    rgb_right: np.ndarray
    intrinsics: np.ndarray
    world_from_left: np.ndarray
    baseline: float
    timestamp: float
    frame: str
    depth_left: np.ndarray | None = None
    depth_right: np.ndarray | None = None

    def __post_init__(self):
        left = np.asarray(self.rgb_left)
        right = np.asarray(self.rgb_right)
        if (
            left.shape != right.shape
            or left.ndim != 3
            or left.shape[2] != 3
            or left.dtype != np.uint8
            or right.dtype != np.uint8
        ):
            raise ValueError("Le immagini devono essere HxWx3 uint8 della stessa dimensione")
        if not np.array_equal(left[:, :, 0], left[:, :, 1]) or not np.array_equal(
            left[:, :, 1], left[:, :, 2]
        ):
            raise ValueError("La camera monocromatica richiede tre canali identici")
        if not np.array_equal(right[:, :, 0], right[:, :, 1]) or not np.array_equal(
            right[:, :, 1], right[:, :, 2]
        ):
            raise ValueError("La camera monocromatica richiede tre canali identici")
        for depth in (self.depth_left, self.depth_right):
            if depth is not None and np.asarray(depth).shape != left.shape[:2]:
                raise ValueError("Depth non allineata al frame")
        if not np.isfinite(self.timestamp):
            raise ValueError("Timestamp del frame non valido")

    @property
    def gray_left(self) -> np.ndarray:
        return self.rgb_left[:, :, 0]

    @property
    def gray_right(self) -> np.ndarray:
        return self.rgb_right[:, :, 0]


@dataclass(frozen=True)
class LidarFrame:
    """Una scansione LiDAR espressa nel frame canonico dichiarato."""

    points: np.ndarray
    ranges: np.ndarray
    directions: np.ndarray
    valid: np.ndarray
    timestamp: float
    frame: str
    origin: tuple[float, float, float]

    def __post_init__(self):
        n = len(self.ranges)
        if (
            self.points.shape != (n, 3)
            or self.directions.shape != (n, 3)
            or self.valid.shape != (n,)
        ):
            raise ValueError("Dimensioni del frame LiDAR incoerenti")
        if not np.isfinite(self.timestamp) or not np.isfinite(self.origin).all():
            raise ValueError("Timestamp o origine LiDAR non validi")
        if not np.isfinite(self.directions).all() or not np.allclose(
            np.linalg.norm(self.directions, axis=1), 1.0, atol=1e-5
        ):
            raise ValueError("Direzioni LiDAR non unitarie")
        if np.any(self.valid & (~np.isfinite(self.ranges) | (self.ranges < 0))):
            raise ValueError("Ritorni LiDAR validi con range non valido")

    @property
    def valid_points(self) -> np.ndarray:
        return self.points[self.valid]

    def to_dict(self) -> dict:
        return {
            "points": [
                point.tolist() if valid else None
                for point, valid in zip(self.points, self.valid)
            ],
            "ranges": [
                float(value) if valid else None
                for value, valid in zip(self.ranges, self.valid)
            ],
            "directions": self.directions.tolist(),
            "valid": self.valid.tolist(),
            "timestamp": self.timestamp,
            "frame": self.frame,
            "origin": list(self.origin),
        }


@dataclass(frozen=True)
class SensorCalibration:
    """Geometria del rig, senza informazioni sul task o sulla scena."""

    intrinsics: np.ndarray
    world_from_left: np.ndarray
    world_from_lidar: np.ndarray
    baseline: float
    image_height: int
    image_width: int
    distortion: tuple[float, ...] = ()

    def __post_init__(self):
        k = np.asarray(self.intrinsics, dtype=float)
        left = np.asarray(self.world_from_left, dtype=float)
        lidar = np.asarray(self.world_from_lidar, dtype=float)
        if k.shape != (3, 3) or left.shape != (4, 4) or lidar.shape != (4, 4):
            raise ValueError("Dimensioni della calibrazione non valide")
        if not np.isfinite(k).all() or not np.isfinite(left).all() or not np.isfinite(lidar).all():
            raise ValueError("La calibrazione deve contenere valori finiti")
        if k[0, 0] <= 0 or k[1, 1] <= 0 or self.baseline <= 0:
            raise ValueError("Focale e baseline devono essere positive")
        if self.image_height < 1 or self.image_width < 1:
            raise ValueError("Dimensioni immagine non valide")
        for transform in (left, lidar):
            rotation = transform[:3, :3]
            if (
                not np.allclose(transform[3], (0, 0, 0, 1))
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
            ):
                raise ValueError("Le extrinsics devono essere trasformazioni rigide")
        if not np.isfinite(self.distortion).all():
            raise ValueError("Coefficienti di distorsione non validi")
        object.__setattr__(self, "intrinsics", k.copy())
        object.__setattr__(self, "world_from_left", left.copy())
        object.__setattr__(self, "world_from_lidar", lidar.copy())


@dataclass(frozen=True)
class Detection:
    """Istanza segmentata nelle viste, indipendente dal detector usato."""

    track_id: str
    left_mask: np.ndarray
    right_mask: np.ndarray | None
    class_id: str | None
    class_confidence: float | None

    def __post_init__(self):
        left = np.asarray(self.left_mask, dtype=bool)
        right = None if self.right_mask is None else np.asarray(self.right_mask, dtype=bool)
        if left.ndim != 2 or (right is not None and right.shape != left.shape):
            raise ValueError("Maschere della detection non valide")
        if not self.track_id:
            raise ValueError("track_id non puo' essere vuoto")
        if self.class_confidence is not None and not 0 <= self.class_confidence <= 1:
            raise ValueError("class_confidence deve essere compresa tra 0 e 1")
        object.__setattr__(self, "left_mask", left)
        object.__setattr__(self, "right_mask", right)


@dataclass(frozen=True)
class SynchronizedSensorPacket:
    """Misure grezze sincronizzate prodotte da qualsiasi SensorSource."""

    stereo: StereoFrame
    lidar: LidarFrame
    calibration: SensorCalibration
    synchronization_error: float

    def __post_init__(self):
        if not np.isfinite(self.synchronization_error) or self.synchronization_error < 0:
            raise ValueError("Errore di sincronizzazione non valido")
        measured = abs(self.stereo.timestamp - self.lidar.timestamp)
        if not np.isclose(self.synchronization_error, measured, atol=1e-9):
            raise ValueError("Errore di sincronizzazione incoerente con i timestamp")
        calibration = self.calibration
        height, width = self.stereo.gray_left.shape
        if (height, width) != (calibration.image_height, calibration.image_width):
            raise ValueError("Calibrazione e dimensioni delle immagini non coincidono")
        if not np.allclose(self.stereo.intrinsics, calibration.intrinsics):
            raise ValueError("Intrinsics del frame e del rig non coincidono")
        if not np.allclose(self.stereo.world_from_left, calibration.world_from_left):
            raise ValueError("Extrinsics camera del frame e del rig non coincidono")
        if not np.isclose(self.stereo.baseline, calibration.baseline):
            raise ValueError("Baseline del frame e del rig non coincidono")
        if self.stereo.frame != self.lidar.frame:
            raise ValueError("Stereo e LiDAR devono essere espressi nello stesso frame")

    @property
    def timestamp(self) -> float:
        return max(self.stereo.timestamp, self.lidar.timestamp)

    @property
    def frame(self) -> str:
        return self.stereo.frame
