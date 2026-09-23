"""Diagnostica CAD riproducibile; non sostituisce un test LiDAR reale."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from physical_ai_mujoco.observe.cad import CADMatcher


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STL = ROOT / "datasets/object_dataset/PFM-1_body copia.STL"


def _rotations() -> tuple[np.ndarray, ...]:
    result = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            matrix = np.eye(3)[list(permutation)] * np.asarray(signs)[:, None]
            if np.linalg.det(matrix) > 0:
                result.append(matrix)
    return tuple(result)


def validate_cad(path: Path = DEFAULT_STL, seed: int = 7) -> dict:
    """Misura fit su superfici parziali e 24 pose note del CAD STL.

    La frazione e una selezione dei punti CAD in quota, non l'esposizione
    ottica o LiDAR in una scena MuJoCo. I risultati sono quindi diagnostici.
    """
    matcher = CADMatcher.from_stl(
        path, scale=(0.001, 0.001, 0.001),
        max_model_points=400, max_observed_points=120, max_iterations=8,
    )
    points = matcher.model_points
    rng = np.random.default_rng(seed)
    translation = np.asarray((0.35, -0.12, 0.08))
    rows = []
    for pose_index, rotation in enumerate(_rotations()):
        transformed = points @ rotation.T + translation
        height = transformed[:, 2]
        for exposure in (0.1, 0.3, 0.5, 0.7, 0.9):
            threshold = float(np.quantile(height, 1.0 - exposure))
            visible = transformed[height >= threshold]
            if len(visible) > 120:
                visible = visible[rng.choice(len(visible), 120, replace=False)]
            row = {"pose": pose_index, "exposure_fraction": exposure,
                   "observed_points": len(visible)}
            try:
                match = matcher.match(visible)
                row.update(
                    accepted=True,
                    translation_error_m=float(np.linalg.norm(
                        np.asarray(match.geometry.position) - translation)),
                    raw_rotation_error_deg=float(np.degrees(2 * np.arccos(
                        np.clip(abs(np.dot(
                            np.asarray(match.geometry.quaternion),
                            _quaternion_from_rotation(rotation),
                        )), 0.0, 1.0)))),
                    size_ratio=[float(a / b) for a, b in zip(
                        match.geometry.size, matcher.nominal_size)],
                    rmse_m=match.rmse,
                )
            except ValueError as error:
                row.update(accepted=False, error=str(error))
            rows.append(row)
    return {
        "kind": "synthetic_cad_partial_surface_diagnostic",
        "stl": str(path), "seed": seed,
        "limitations": [
            "Partial points are selected from the CAD by world height, not captured by LiDAR.",
            "size_ratio is 1 by construction for any accepted CAD match and cannot validate pose.",
            "Rotation error is not scored because physical symmetries need an explicit equivalence definition.",
        ],
        "rows": rows,
    }


def _quaternion_from_rotation(rotation: np.ndarray) -> np.ndarray:
    import cv2

    vector, _ = cv2.Rodrigues(rotation)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.asarray((1.0, 0.0, 0.0, 0.0))
    return np.r_[np.cos(angle / 2), np.sin(angle / 2) * vector[:, 0] / angle]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl", type=Path, default=DEFAULT_STL)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = validate_cad(args.stl, args.seed)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for fraction in (0.1, 0.3, 0.5, 0.7, 0.9):
        rows = [row for row in report["rows"] if row["exposure_fraction"] == fraction]
        accepted = [row for row in rows if row["accepted"]]
        median = float(np.median([row["translation_error_m"] for row in accepted])) if accepted else float("nan")
        print(f"Esposizione proxy {fraction:.0%}: fit {len(accepted)}/24, errore centro mediano {median:.3f} m")
    if args.output:
        print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
