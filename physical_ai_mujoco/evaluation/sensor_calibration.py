"""Diagnostica indipendente dei sensori e sensibilita' alla calibrazione."""

import argparse
import json
from pathlib import Path

import numpy as np

from physical_ai_mujoco.evaluation.observe_benchmark import run_benchmark


def compare_lidar_depth(lidar_frame, depth_left, rig) -> dict:
    """Confronta hit LiDAR e depth camera con posa nota, senza ID MuJoCo."""
    depth = np.asarray(depth_left, dtype=float)
    if depth.shape != (rig.height, rig.width):
        raise ValueError("Depth e rig non hanno le stesse dimensioni")
    residuals = []
    for point in lidar_frame.valid_points:
        try:
            u, v, z = rig.project(point)
        except ValueError:
            continue
        col, row = int(round(u)), int(round(v))
        if not (0 <= row < rig.height and 0 <= col < rig.width):
            continue
        measured = depth[row, col]
        if np.isfinite(measured) and measured > 0:
            residuals.append(float(abs(measured-z)))
    return {"matches": len(residuals),
            "median_abs_depth_error_m": float(np.median(residuals)) if residuals else None,
            "mean_abs_depth_error_m": float(np.mean(residuals)) if residuals else None}


def calibration_curve(config: dict, translation_levels: tuple[float, ...],
                      rotation_levels_deg: tuple[float, ...]) -> dict:
    """Stesse scene/seed per ogni livello; risultati, non soglie imposte."""
    if not translation_levels or not rotation_levels_deg:
        raise ValueError("La curva richiede almeno un livello per ogni asse")
    records = []
    for translation in translation_levels:
        for rotation in rotation_levels_deg:
            if translation < 0 or rotation < 0:
                raise ValueError("Livelli di calibrazione negativi")
            selected = json.loads(json.dumps(config))
            selected.setdefault("rgbd", {})["translation_sigma"] = translation
            selected["rgbd"]["rotation_sigma_deg"] = rotation
            report = run_benchmark(selected, ("rgbd",))
            metric = report["summary"]["rgbd"]["metrics"]
            records.append({
                "translation_sigma_m": translation,
                "rotation_sigma_deg": rotation,
                "position_error_m": metric["mean_position_error_m"]["mean"],
                "size_error_m": metric["mean_size_error_m"]["mean"],
                "support_f1": metric["support_f1"]["mean"],
                "object_recall": metric["object_recall"]["mean"],
                "invariant_violations": report["summary"]["rgbd"]["invariant_violations"],
            })
    return {"depth_source": "MuJoCo rendered depth", "records": records}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Curva errore calibrazione vs OSSERVA RGB-D")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] /
                        "configs/observe_tests/clean.json")
    parser.add_argument("--scenes", type=int, default=3)
    parser.add_argument("--objects", type=int, default=3)
    parser.add_argument("--translation", type=float, nargs="+", default=[0, .005, .01])
    parser.add_argument("--rotation", type=float, nargs="+", default=[0, .25, .5])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] /
                        "outputs/observe_tests/calibration_curve.json")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["scene_count"] = args.scenes
    config["object_count"] = args.objects
    result = calibration_curve(config, tuple(args.translation), tuple(args.rotation))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for record in result["records"]:
        print(record)
    print(f"Report: {args.output}")
    return 1 if any(record["invariant_violations"] for record in result["records"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
