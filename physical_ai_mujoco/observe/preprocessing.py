"""Filtri geometrici applicati alle misure LiDAR prima della fusione."""

import numpy as np

from physical_ai_mujoco.contracts import LidarFrame, TaskContext


class LidarPreprocessor:
    def __init__(self, ground_margin: float = 0.002):
        if ground_margin < 0:
            raise ValueError("ground_margin deve essere non negativo")
        self.ground_margin = float(ground_margin)

    def process(self, frame: LidarFrame, context: TaskContext) -> np.ndarray:
        points = np.asarray(frame.valid_points, dtype=float)
        if not len(points):
            return np.empty((0, 3), dtype=float)
        keep = np.isfinite(points).all(axis=1)
        keep &= points[:, 2] > context.ground_height + self.ground_margin
        if context.bounds is not None:
            x_min, x_max, y_min, y_max = context.bounds
            keep &= (
                (points[:, 0] >= x_min)
                & (points[:, 0] <= x_max)
                & (points[:, 1] >= y_min)
                & (points[:, 1] <= y_max)
            )
        return points[keep]
