"""Geometria calibrata della coppia stereo, senza dipendenze da MuJoCo."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StereoRig:
    intrinsics: np.ndarray
    world_from_left: np.ndarray
    baseline: float
    height: int
    width: int

    def __post_init__(self) -> None:
        k = np.asarray(self.intrinsics, dtype=float)
        pose = np.asarray(self.world_from_left, dtype=float)
        if k.shape != (3, 3) or pose.shape != (4, 4):
            raise ValueError("K deve essere 3x3 e world_from_left 4x4")
        if not np.isfinite(k).all() or not np.isfinite(pose).all():
            raise ValueError("La calibrazione deve essere finita")
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], (0, 0, 1)):
            raise ValueError("Intrinsics non validi")
        rotation = pose[:3, :3]
        if not np.allclose(pose[3], (0, 0, 0, 1)) or not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1e-5
        ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError("world_from_left non e' una trasformazione rigida")
        if not np.isfinite(self.baseline) or self.baseline <= 0:
            raise ValueError("La baseline deve essere positiva")
        if self.height < 1 or self.width < 1:
            raise ValueError("Dimensioni immagine non valide")
        object.__setattr__(self, "intrinsics", k.copy())
        object.__setattr__(self, "world_from_left", pose.copy())

    def world_from_eye(self, eye: str = "left") -> np.ndarray:
        if eye not in ("left", "right"):
            raise ValueError(f"Occhio sconosciuto: {eye}")
        pose = self.world_from_left.copy()
        if eye == "right":
            pose[:3, 3] += pose[:3, 0] * self.baseline
        return pose

    def project(self, point_3d, eye: str = "left") -> tuple[float, float, float]:
        point = np.asarray(point_3d, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Punto 3D non valido")
        pose = self.world_from_eye(eye)
        local = pose[:3, :3].T @ (point - pose[:3, 3])
        if local[2] <= 0:
            raise ValueError("Il punto e' dietro la camera")
        u = self.intrinsics[0, 0] * local[0] / local[2] + self.intrinsics[0, 2]
        v = self.intrinsics[1, 1] * local[1] / local[2] + self.intrinsics[1, 2]
        return float(u), float(v), float(local[2])

    def unproject(self, u: float, v: float, depth: float, eye: str = "left") -> np.ndarray:
        if not np.isfinite((u, v, depth)).all() or depth <= 0:
            raise ValueError("Pixel o profondita' non validi")
        local = np.array([
            (u - self.intrinsics[0, 2]) * depth / self.intrinsics[0, 0],
            (v - self.intrinsics[1, 2]) * depth / self.intrinsics[1, 1],
            depth,
        ])
        pose = self.world_from_eye(eye)
        return pose[:3, :3] @ local + pose[:3, 3]

    @classmethod
    def from_scene_description(cls, description):
        focal = description.focal_length_px()
        eye = np.asarray(description.position, dtype=float)
        target = np.asarray(description.target, dtype=float)
        forward = target - eye
        if np.linalg.norm(forward) == 0:
            raise ValueError("Posizione e target della camera coincidono")
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(forward, world_up))) > 0.999:
            world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        pose = np.eye(4)
        pose[:3, :3] = np.column_stack((right, -up, forward))
        pose[:3, 3] = eye - right * description.baseline / 2
        k = np.array([[focal, 0, description.width / 2],
                      [0, focal, description.height / 2], [0, 0, 1]])
        return cls(k, pose, description.baseline, description.height, description.width)

    @classmethod
    def from_calibration_file(cls, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(np.asarray(data["intrinsics"]), np.asarray(data["world_from_left"]),
                   float(data["baseline"]), int(data["height"]), int(data["width"]))
