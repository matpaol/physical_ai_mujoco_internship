"""Allineamento rigido di un CAD noto a una nuvola LiDAR parziale."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import cv2
import numpy as np

from physical_ai_mujoco.sensors.cloud import ObjectGeometry


def _rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    vector, _ = cv2.Rodrigues(np.asarray(rotation, dtype=float))
    angle = float(np.linalg.norm(vector))
    if angle <= 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    xyz = np.sin(angle / 2.0) * vector[:, 0] / angle
    return (float(np.cos(angle / 2.0)), *(float(value) for value in xyz))


def _principal_axes(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    axes = vectors.T
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return axes


def _kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - source_center @ rotation.T
    return rotation, translation


def _nearest_indices(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.empty(len(source), dtype=int)
    distances = np.empty(len(source), dtype=float)
    for start in range(0, len(source), 256):
        chunk = source[start:start + 256]
        squared = np.sum((chunk[:, None, :] - target[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(squared, axis=1)
        indices[start:start + len(chunk)] = nearest
        distances[start:start + len(chunk)] = np.sqrt(
            squared[np.arange(len(chunk)), nearest]
        )
    return indices, distances


@dataclass(frozen=True)
class CADMatch:
    geometry: ObjectGeometry
    rmse: float
    iterations: int


class CADMatcher:
    """ICP parziale con inizializzazioni PCA e trimming degli outlier.

    La distanza e calcolata dai punti osservati verso il CAD, quindi le
    superfici CAD non visibili non vengono penalizzate. Il risultato usa
    dimensioni nominali complete, non l'inviluppo della sola parte visibile.
    """

    def __init__(
        self,
        model_points: np.ndarray,
        *,
        max_model_points: int = 1200,
        max_observed_points: int = 600,
        max_iterations: int = 12,
        trim_fraction: float = 0.8,
        convergence_tolerance: float = 1e-5,
        max_normalized_rmse: float = 0.15,
    ):
        points = np.asarray(model_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 20:
            raise ValueError("Il CAD richiede almeno venti punti 3D")
        if not np.isfinite(points).all():
            raise ValueError("Il CAD contiene punti non finiti")
        if not 0.25 <= trim_fraction <= 1.0:
            raise ValueError("trim_fraction deve essere in [0.25, 1]")
        if max_iterations < 1 or max_model_points < 20 or max_observed_points < 6:
            raise ValueError("Parametri CADMatcher non validi")
        if not 0 < max_normalized_rmse < 1:
            raise ValueError("max_normalized_rmse deve essere in (0, 1)")
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        self.nominal_size = maximum - minimum
        centered = points - (minimum + maximum) / 2.0
        if len(centered) > max_model_points:
            indices = np.linspace(0, len(centered) - 1, max_model_points, dtype=int)
            centered = centered[indices]
        self.model_points = centered
        self.max_observed_points = int(max_observed_points)
        self.max_iterations = int(max_iterations)
        self.trim_fraction = float(trim_fraction)
        self.convergence_tolerance = float(convergence_tolerance)
        self.max_normalized_rmse = float(max_normalized_rmse)
        self._model_axes = _principal_axes(centered)

    @classmethod
    def from_stl(
        cls,
        path: str | Path,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        **kwargs,
    ) -> "CADMatcher":
        data = Path(path).read_bytes()
        if len(data) < 84:
            raise ValueError(f"STL non valido: {path}")
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * triangle_count != len(data):
            raise ValueError("CADMatcher supporta STL binari validi")
        triangles = np.empty((triangle_count, 3, 3), dtype=np.float32)
        for index in range(triangle_count):
            offset = 84 + index * 50 + 12
            triangles[index] = np.frombuffer(
                data, dtype="<f4", count=9, offset=offset
            ).reshape(3, 3)
        points = triangles.reshape(-1, 3).astype(float)
        points *= np.asarray(scale, dtype=float)
        quantized = np.round(points / 1e-5).astype(np.int64)
        _, unique_indices = np.unique(quantized, axis=0, return_index=True)
        return cls(points[np.sort(unique_indices)], **kwargs)

    def _initial_rotations(self, observed: np.ndarray) -> list[np.ndarray]:
        observed_axes = _principal_axes(observed)
        rotations = []
        for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
            signed = self._model_axes @ np.diag(signs)
            rotation = observed_axes @ signed.T
            if np.linalg.det(rotation) > 0:
                rotations.append(rotation)
        return rotations

    def match(self, observed_points: np.ndarray) -> CADMatch:
        observed = np.asarray(observed_points, dtype=float)
        if observed.ndim != 2 or observed.shape[1] != 3 or len(observed) < 6:
            raise ValueError("Servono almeno sei punti osservati per il CAD matching")
        observed = observed[np.isfinite(observed).all(axis=1)]
        if len(observed) < 6:
            raise ValueError("Punti CAD osservati insufficienti o non finiti")
        if len(observed) > self.max_observed_points:
            indices = np.linspace(0, len(observed) - 1, self.max_observed_points, dtype=int)
            observed = observed[indices]

        best = None
        model_center = self.model_points.mean(axis=0)
        observed_center = observed.mean(axis=0)
        for initial_rotation in self._initial_rotations(observed):
            rotation = initial_rotation
            translation = observed_center - model_center @ rotation.T
            previous_rmse = np.inf
            completed = 0
            for iteration in range(self.max_iterations):
                transformed = self.model_points @ rotation.T + translation
                nearest, distances = _nearest_indices(observed, transformed)
                keep_count = max(6, int(np.ceil(len(observed) * self.trim_fraction)))
                keep = np.argsort(distances)[:keep_count]
                source = self.model_points[nearest[keep]]
                target = observed[keep]
                rotation, translation = _kabsch(source, target)
                transformed = self.model_points @ rotation.T + translation
                _, updated_distances = _nearest_indices(observed, transformed)
                rmse = float(np.sqrt(np.mean(np.sort(updated_distances)[:keep_count] ** 2)))
                completed = iteration + 1
                if abs(previous_rmse - rmse) <= self.convergence_tolerance:
                    break
                previous_rmse = rmse
            if best is None or rmse < best[0]:
                best = (rmse, rotation, translation, completed)

        rmse, rotation, translation, iterations = best
        diagonal = max(float(np.linalg.norm(self.nominal_size)), 1e-9)
        if rmse / diagonal > self.max_normalized_rmse:
            raise ValueError(
                "Allineamento CAD rifiutato: errore incompatibile con il modello"
            )
        fit_quality = float(np.exp(-rmse / (0.05 * diagonal)))
        point_quality = min(1.0, len(observed) / 100.0)
        geometry = ObjectGeometry(
            tuple(float(value) for value in translation),
            _rotation_to_quaternion(rotation),
            "mesh",
            tuple(float(value) for value in self.nominal_size),
            fit_quality * point_quality,
            len(observed),
        )
        return CADMatch(geometry, rmse, iterations)
