"""Associazione fra segmentazioni 2D e ritorni LiDAR calibrati."""

from dataclasses import dataclass

import numpy as np

from physical_ai_mujoco.contracts import Detection, SensorCalibration


@dataclass(frozen=True)
class LocalizedDetection:
    detection: Detection
    points: np.ndarray
    camera_depths: np.ndarray


class LidarProjector:
    """Proietta punti nel piano immagine e conserva quelli dentro ogni mask."""

    def project(
        self,
        points: np.ndarray,
        detections: tuple[Detection, ...],
        calibration: SensorCalibration,
    ) -> tuple[LocalizedDetection, ...]:
        cloud = np.asarray(points, dtype=float)
        if cloud.ndim != 2 or cloud.shape[1:] != (3,):
            raise ValueError("La point cloud deve avere forma Nx3")
        if not len(cloud):
            return tuple(
                LocalizedDetection(item, np.empty((0, 3)), np.empty((0,)))
                for item in detections
            )
        if not np.isfinite(cloud).all():
            raise ValueError("La point cloud contiene valori non finiti")

        world_from_left = calibration.world_from_left
        left_from_world = np.linalg.inv(world_from_left)
        homogeneous = np.column_stack((cloud, np.ones(len(cloud))))
        camera = (left_from_world @ homogeneous.T).T[:, :3]
        in_front = camera[:, 2] > 0
        k = calibration.intrinsics
        u = np.full(len(cloud), -1, dtype=int)
        v = np.full(len(cloud), -1, dtype=int)
        u[in_front] = np.rint(
            k[0, 0] * camera[in_front, 0] / camera[in_front, 2] + k[0, 2]
        ).astype(int)
        v[in_front] = np.rint(
            k[1, 1] * camera[in_front, 1] / camera[in_front, 2] + k[1, 2]
        ).astype(int)
        visible = (
            in_front
            & (u >= 0)
            & (u < calibration.image_width)
            & (v >= 0)
            & (v < calibration.image_height)
        )

        localized = []
        for detection in detections:
            mask = np.asarray(detection.left_mask, dtype=bool)
            if mask.shape != (calibration.image_height, calibration.image_width):
                raise ValueError(f"Maschera non allineata per {detection.track_id}")
            selected = np.zeros(len(cloud), dtype=bool)
            indices = np.flatnonzero(visible)
            selected[indices] = mask[v[indices], u[indices]]
            localized.append(
                LocalizedDetection(
                    detection=detection,
                    points=cloud[selected],
                    camera_depths=camera[selected, 2],
                )
            )
        return tuple(localized)
