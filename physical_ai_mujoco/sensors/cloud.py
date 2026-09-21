"""Ricostruzione di punti 3D e geometria approssimata da superfici visibili."""

from dataclasses import dataclass

import numpy as np


def mask_to_point_cloud(depth: np.ndarray, intrinsics: np.ndarray,
                        world_from_camera: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = np.asarray(depth, dtype=float)
    region = np.asarray(mask, dtype=bool)
    k = np.asarray(intrinsics, dtype=float)
    pose = np.asarray(world_from_camera, dtype=float)
    if image.ndim != 2 or region.shape != image.shape or k.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError("Dimensioni depth, maschera o calibrazione non valide")
    if k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("Focale non valida")
    rows, cols = np.nonzero(region & np.isfinite(image) & (image > 0))
    if not len(rows):
        return np.empty((0, 3), dtype=float)
    z = image[rows, cols]
    camera = np.column_stack(((cols-k[0, 2])*z/k[0, 0],
                              (rows-k[1, 2])*z/k[1, 1], z))
    return camera @ pose[:3, :3].T + pose[:3, 3]


@dataclass(frozen=True)
class ObjectGeometry:
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float] | None
    shape: str
    size: tuple[float, float, float]
    confidence: float
    point_count: int


def _quaternion_from_rotation(rotation: np.ndarray) -> tuple[float, float, float, float]:
    # Convenzione MuJoCo: w, x, y, z.
    import cv2

    vector, _ = cv2.Rodrigues(rotation)
    angle = float(np.linalg.norm(vector))
    if angle == 0:
        return (1.0, 0.0, 0.0, 0.0)
    xyz = np.sin(angle/2) * vector[:, 0] / angle
    return (float(np.cos(angle/2)), *(float(v) for v in xyz))


def estimate_geometry(points: np.ndarray, *, shape_hint: str | None = None) -> ObjectGeometry:
    """Stima dall'inviluppo *visibile*: occlusioni possono sottostimare la size."""
    cloud = np.asarray(points, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) < 6 or not np.isfinite(cloud).all():
        raise ValueError("Servono almeno sei punti 3D finiti")
    if shape_hint is not None and shape_hint not in ("box", "cylinder", "sphere"):
        raise ValueError("shape_hint non supportato")
    center = (cloud.min(axis=0) + cloud.max(axis=0)) / 2
    covariance = np.cov((cloud-center).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axes = eigenvectors[:, ::-1]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    local = (cloud-center) @ axes
    extents = np.ptp(local, axis=0)
    radial = np.linalg.norm(cloud-cloud.mean(axis=0), axis=1)
    radial_cv = float(np.std(radial) / np.mean(radial)) if np.mean(radial) else 1.0
    planar_radial = np.linalg.norm(local[:, :2], axis=1)
    planar_cv = float(np.std(planar_radial) / np.mean(planar_radial)) if np.mean(planar_radial) else 1.0
    shape = shape_hint or ("sphere" if radial_cv < 0.10 else
                           "cylinder" if planar_cv < 0.10 else "box")
    if shape == "sphere":
        diameter = 2 * float(np.median(radial))
        size = (diameter,) * 3
        quaternion = None
    elif shape == "cylinder":
        diameter = 2 * float(np.median(planar_radial))
        size = (diameter, diameter, float(extents[2]))
        quaternion = _quaternion_from_rotation(axes)
    else:
        size = tuple(float(value) for value in extents)
        quaternion = _quaternion_from_rotation(axes)
    # Indicatore empirico di copertura, non probabilita' calibrata.
    coverage = float(np.clip(eigenvalues[0] / max(eigenvalues[-1], 1e-12), 0, 1))
    confidence = float(min(1.0, len(cloud)/100) * coverage)
    return ObjectGeometry(tuple(float(value) for value in center), quaternion,
                          shape, size, confidence, len(cloud))
