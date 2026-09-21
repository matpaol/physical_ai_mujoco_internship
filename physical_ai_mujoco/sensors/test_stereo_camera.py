"""Test visivo autonomo della stereocamera simulata dentro Gymnasium.

Avvio: python -m physical_ai_mujoco.sensors.test_stereo_camera
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np

# Consente sia `python -m ...` sia `python percorso/del/file.py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from physical_ai_mujoco.sensors import (
    STEREO_BASELINE_M, BundleBuilder, DetectionNoise, OracleDetector, SimulatedStereoCamera,
)


def _annotate(rgb: np.ndarray, masks: dict[str, np.ndarray], label: str) -> np.ndarray:
    import cv2

    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    shades = (85, 145, 205, 250)
    for index, (object_id, mask) in enumerate(sorted(masks.items())):
        shade = shades[index % len(shades)]
        color = (shade, shade, shade)
        image[mask] = (
            0.55 * image[mask].astype(float) + 0.45 * np.asarray(color)
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, color, 2)
        y, x = np.rint(np.argwhere(mask).mean(axis=0)).astype(int)
        cv2.putText(
            image, object_id, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
            0.38, (255, 255, 255), 1, cv2.LINE_AA,
        )
    cv2.rectangle(image, (0, 0), (image.shape[1], 22), (30, 30, 30), -1)
    cv2.putText(image, label, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (255, 255, 255), 1, cv2.LINE_AA)
    return image


def preview_stereo(
    seed: int = 42,
    object_count: int = 3,
    degradation: DetectionNoise | None = None,
    duration_seconds: float | None = None,
    output_path: Path | None = None,
) -> Path:
    """Apre il viewer Gymnasium e una finestra monocromatica stereo.

    Premere q/Esc nella finestra delle camere per chiudere. Una durata finita
    consente anche una prova automatizzata senza interazione manuale.
    """
    import cv2
    import gymnasium as gym
    import physical_ai_mujoco.envs  # noqa: F401 (registra l'environment)

    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds deve essere positivo")
    camera = SimulatedStereoCamera()
    degradation = degradation or DetectionNoise()
    env = gym.make(
        "TargetExtraction-v0", object_count=object_count,
        obs_mode="state", render_mode="human", resample_shapes=False,
        highlight_target=False, realtime_factor=0.0, disable_env_checker=True,
        stereo_baseline=STEREO_BASELINE_M,
    )
    window_name = "Stereo simulata — sinistra | destra (q per chiudere)"
    window_open = False
    try:
        _, info = env.reset(seed=seed)
        env.render()
        frame = camera.capture(env.unwrapped.simulator)
        bundle = BundleBuilder(
            OracleDetector(env.unwrapped.simulator), degradation, seed=seed
        ).build(frame)
        preview = np.concatenate(
            (
                _annotate(bundle.rgb_left, bundle.left_masks, "CAMERA SINISTRA"),
                _annotate(bundle.rgb_right, bundle.right_masks, "CAMERA DESTRA"),
            ),
            axis=1,
        )
        destination = output_path or (
            Path(__file__).resolve().parents[2]
            / "outputs/observe_tests"
            / f"stereo_preview_seed_{seed}.png"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), preview):
            raise RuntimeError(f"Impossibile salvare {destination}")
        print(
            f"Scena {info['scene_seed']} | target {env.unwrapped.target_id} | "
            f"baseline {bundle.baseline:.3f} m"
        )
        print(f"Maschere sinistra: {sorted(bundle.left_masks)}")
        print(f"Maschere destra: {sorted(bundle.right_masks)}")
        print(
            f"Disturbo rilevazioni: rumore centroidi "
            f"{degradation.centroid_noise_px:.2f} px, "
            f"drop {degradation.drop_probability:.2f}"
        )
        print(f"Anteprima salvata: {destination}")
        print("Le etichette provengono dalla segmentazione MuJoCo, non da un detector RGB.")
        cv2.imshow(window_name, preview)
        window_open = True
        started = time.monotonic()
        while True:
            env.render()
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if duration_seconds is not None and time.monotonic() - started >= duration_seconds:
                break
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
        return destination
    finally:
        if window_open:
            cv2.destroyWindow(window_name)
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test visivo della stereocamera")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objects", type=int, default=3)
    parser.add_argument("--noise-px", type=float, default=0.0,
                        help="Rumore sui centroidi delle maschere (pixel)")
    parser.add_argument("--drop-probability", type=float, default=0.0,
                        help="Probabilita' di omettere una maschera per vista")
    parser.add_argument("--seconds", type=float, help="Chiudi automaticamente dopo N secondi")
    args = parser.parse_args(argv)
    preview_stereo(
        args.seed, args.objects,
        degradation=DetectionNoise(centroid_noise_px=args.noise_px, drop_probability=args.drop_probability),
        duration_seconds=args.seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
