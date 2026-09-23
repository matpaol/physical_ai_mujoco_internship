"""Pattern di scansione 2D/3D applicato all'API pubblica del simulatore."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .frame import LidarFrame


@dataclass(frozen=True)
class LidarConfig:
    n_rays: int = 720
    n_planes: int = 1
    fov_horizontal_deg: float = 360.0
    fov_vertical_deg: float = 0.0
    max_range: float = 10.0
    origin: tuple[float, float, float] = (0.0, 0.0, 0.2)
    rotation: tuple[tuple[float, float, float], ...] = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

    def __post_init__(self):
        if self.n_rays < 1 or self.n_planes < 1 or not 0 < self.fov_horizontal_deg <= 360:
            raise ValueError("Pattern LiDAR non valido")
        if not 0 <= self.fov_vertical_deg < 180 or not np.isfinite(self.max_range) or self.max_range <= 0:
            raise ValueError("FOV verticale o portata LiDAR non validi")
        rotation = np.asarray(self.rotation, dtype=float)
        if (rotation.shape != (3, 3) or
                not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or
                not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)):
            raise ValueError("Rotazione LiDAR non valida")
        if not np.isfinite(self.origin).all():
            raise ValueError("Origine LiDAR non valida")

    @classmethod
    def from_file(cls, path: str | Path) -> "LidarConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        scan = data.get("simulation_sampling", data)
        mounting = data.get("mounting", {})
        return cls(
            n_rays=int(scan["n_rays"]),
            n_planes=int(scan["n_planes"]),
            fov_horizontal_deg=float(scan["fov_horizontal_deg"]),
            fov_vertical_deg=float(scan["fov_vertical_deg"]),
            max_range=float(scan["max_range"]),
            origin=tuple(float(value) for value in mounting.get("origin", (0, 0, 0.2))),
            rotation=tuple(
                tuple(float(value) for value in row)
                for row in mounting.get(
                    "rotation", ((1, 0, 0), (0, 1, 0), (0, 0, 1))
                )
            ),
        )


def scan(simulator, config: LidarConfig) -> LidarFrame:
    """Nessun hit e' rappresentato da range inf, punto NaN e valid=False."""
    azimuth = np.linspace(-config.fov_horizontal_deg/2, config.fov_horizontal_deg/2,
                          config.n_rays, endpoint=config.fov_horizontal_deg < 360)
    elevation = (np.array([0.0]) if config.n_planes == 1 else
                 np.linspace(-config.fov_vertical_deg/2, config.fov_vertical_deg/2,
                             config.n_planes))
    radians_h = np.deg2rad(np.tile(azimuth, len(elevation)))
    radians_v = np.deg2rad(np.repeat(elevation, len(azimuth)))
    local = np.column_stack((np.cos(radians_v)*np.cos(radians_h),
                             np.cos(radians_v)*np.sin(radians_h), np.sin(radians_v)))
    directions = local @ np.asarray(config.rotation, dtype=float).T
    origin = np.asarray(config.origin, dtype=float)
    ranges = np.full(len(directions), np.inf)
    for index, direction in enumerate(directions):
        hit = simulator.raycast(origin, direction)
        if hit is not None and 0 <= hit <= config.max_range:
            ranges[index] = hit
    valid = np.isfinite(ranges)
    points = np.full_like(directions, np.nan)
    points[valid] = origin + directions[valid] * ranges[valid, None]
    return LidarFrame(points, ranges, directions, valid, simulator.time, "world", tuple(origin))
