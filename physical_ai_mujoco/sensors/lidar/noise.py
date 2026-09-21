"""Errore di range, direzione e dropout sulle misure LiDAR."""

from dataclasses import dataclass, replace

import numpy as np

from .frame import LidarFrame


@dataclass(frozen=True)
class LidarNoise:
    distance_sigma: float = 0.0
    angle_sigma_deg: float = 0.0
    dropout_probability: float = 0.0
    range_dependent: bool = True

    def __post_init__(self):
        if self.distance_sigma < 0 or self.angle_sigma_deg < 0 or not 0 <= self.dropout_probability <= 1:
            raise ValueError("Disturbo LiDAR non valido")


def apply_lidar_noise(frame: LidarFrame, noise: LidarNoise,
                      rng: np.random.Generator) -> LidarFrame:
    valid = frame.valid.copy()
    if noise.dropout_probability:
        valid &= rng.random(len(valid)) >= noise.dropout_probability
    ranges = frame.ranges.copy()
    if noise.distance_sigma:
        sigma = (
            noise.distance_sigma * ranges[valid]
            if noise.range_dependent
            else np.full(valid.sum(), noise.distance_sigma)
        )
        ranges[valid] += rng.normal(0.0, 1.0, valid.sum()) * sigma
    valid &= ranges > 0
    ranges[~valid] = np.inf
    directions = frame.directions.copy()
    if noise.angle_sigma_deg:
        directions[valid] += rng.normal(0, np.deg2rad(noise.angle_sigma_deg),
                                        (valid.sum(), 3))
        directions[valid] /= np.linalg.norm(directions[valid], axis=1)[:, None]
    points = np.full_like(frame.points, np.nan)
    points[valid] = np.asarray(frame.origin) + directions[valid]*ranges[valid, None]
    return replace(frame, points=points, ranges=ranges, directions=directions, valid=valid)
