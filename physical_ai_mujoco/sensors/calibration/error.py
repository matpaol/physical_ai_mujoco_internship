"""Errore residuo applicato a trasformazioni rigide conosciute."""

import numpy as np

from ..noise import CalibrationNoise
from ..rig import StereoRig


def _small_rotation(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle == 0:
        return np.eye(3)
    axis = vector / angle
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def perturb_transform(pose: np.ndarray, noise: CalibrationNoise,
                      rng: np.random.Generator) -> np.ndarray:
    result = np.asarray(pose, dtype=float).copy()
    if result.shape != (4, 4):
        raise ValueError("La trasformazione deve essere 4x4")
    result[:3, :3] = _small_rotation(
        np.deg2rad(rng.normal(0, noise.rotation_sigma_deg, 3))
    ) @ result[:3, :3]
    result[:3, 3] += rng.normal(0, noise.translation_sigma, 3)
    return result


def perturb_rig(rig: StereoRig, noise: CalibrationNoise,
                rng: np.random.Generator) -> StereoRig:
    baseline = rig.baseline + rng.normal(0, noise.baseline_sigma)
    if baseline <= 0:
        raise ValueError("La baseline perturbata non e' positiva")
    return StereoRig(rig.intrinsics, perturb_transform(rig.world_from_left, noise, rng),
                     baseline, rig.height, rig.width)


def perturb_world_from_lidar(world_from_lidar: np.ndarray, noise: CalibrationNoise,
                             rng: np.random.Generator) -> np.ndarray:
    return perturb_transform(world_from_lidar, noise, rng)
