"""Stima lidar_from_camera da punti 3D corrispondenti, senza ground truth."""

import numpy as np


def _fit_rigid(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source-source_center).T @ (target-target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = target_center - rotation @ source_center
    return pose


def estimate_extrinsics(lidar_points: np.ndarray, camera_points: np.ndarray,
                        *, trim_fraction: float = 0.0) -> np.ndarray:
    """Restituisce camera_from_lidar. Le righe devono essere corrispondenze.

    ``trim_fraction`` scarta gli accoppiamenti con residuo piu' alto e rifitta.
    Senza corrispondenze note il problema non e' risolvibile con questa API.
    """
    source = np.asarray(lidar_points, dtype=float)
    target = np.asarray(camera_points, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Punti LiDAR e camera devono essere Nx3 corrispondenti")
    if len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Servono almeno tre corrispondenze finite")
    if not 0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction deve essere in [0, 0.5)")
    if np.linalg.matrix_rank(source-source.mean(axis=0)) < 2:
        raise ValueError("Corrispondenze degeneri")
    pose = _fit_rigid(source, target)
    if trim_fraction and len(source) > 3:
        # Una prima LS puo' essere trascinata dagli outlier: inizializza con
        # triplette e scegli il modello col minore residuo mediano.
        rng = np.random.default_rng(0)
        best_score = float("inf")
        for _ in range(min(200, len(source)*20)):
            sample = rng.choice(len(source), 3, replace=False)
            if np.linalg.matrix_rank(source[sample]-source[sample].mean(axis=0)) < 2:
                continue
            candidate = _fit_rigid(source[sample], target[sample])
            residual = np.linalg.norm(source @ candidate[:3, :3].T + candidate[:3, 3] - target, axis=1)
            score = float(np.median(residual))
            if score < best_score:
                best_score, pose = score, candidate
        residuals = np.linalg.norm(source @ pose[:3, :3].T + pose[:3, 3] - target, axis=1)
        keep = np.argsort(residuals)[:max(3, int(len(source)*(1-trim_fraction)))]
        if np.linalg.matrix_rank(source[keep]-source[keep].mean(axis=0)) < 2:
            raise ValueError("Corrispondenze non degeneri insufficienti dopo il trimming")
        pose = _fit_rigid(source[keep], target[keep])
    return pose
