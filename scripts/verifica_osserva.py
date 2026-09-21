"""Smoke test end-to-end di tutte le sorgenti OSSERVA sui tre profili."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from physical_ai_mujoco.evaluation.observe_benchmark import (
    ALL_MODES,
    CONFIG_DIR,
    PROJECT_ROOT,
    _interactive_profiles,
    _resolve_detector_weights,
    run_benchmark,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verifica clean/fixed/random su tutte le sorgenti OSSERVA"
    )
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--objects", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    weights = _resolve_detector_weights(args.weights)
    results = []
    for profile_path, profile in _interactive_profiles(CONFIG_DIR):
        config = dict(profile)
        config["scene_count"] = 1
        config["object_count"] = args.objects
        config["seed"] = args.seed
        mode_results = []
        for mode in ALL_MODES:
            try:
                report = run_benchmark(
                    config,
                    (mode,),
                    show_viewer=False,
                    detector_weights=weights,
                    detector_confidence_threshold=args.confidence_threshold,
                )
            except Exception as error:
                mode_results.append(
                    {
                        "mode": mode,
                        "ok": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                continue
            mode_results.append(
                {
                    "mode": mode,
                    "ok": True,
                    "summary": report["summary"][mode],
                }
            )
        results.append(
            {
                "profile": profile_path.stem,
                "ok": all(item["ok"] for item in mode_results),
                "modes": mode_results,
            }
        )

    destination = args.output or PROJECT_ROOT / "outputs/observe_tests" / (
        "verifica_completa_" + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S") + ".json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": str(weights),
        "profiles": results,
        "all_ok": all(item["ok"] for item in results),
    }
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    for profile in results:
        print(f"{profile['profile']}:")
        for item in profile["modes"]:
            state = "OK" if item["ok"] else f"ERRORE: {item['error']}"
            print(f"  {item['mode']}: {state}")
    print(f"Report: {destination}")
    return 0 if payload["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
