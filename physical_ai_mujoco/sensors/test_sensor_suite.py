"""Ispezione autonoma di stereo, depth, segmentazione e LiDAR sulla stessa scena."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from physical_ai_mujoco.sensors import BundleBuilder, OracleDetector, SimulatedStereoCamera
from physical_ai_mujoco.sensors.lidar import LidarConfig, scan


def _panel(gray, title):
    import cv2
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(image, (0, 0), (image.shape[1], 22), (25, 25, 25), -1)
    cv2.putText(image, title, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, .5,
                (255, 255, 255), 1)
    return image


def _depth_panel(depth, title):
    image = np.asarray(depth, dtype=float)
    valid = np.isfinite(image) & (image > 0) & (image < 20)
    gray = np.zeros(image.shape, dtype=np.uint8)
    if valid.any():
        near, far = np.percentile(image[valid], (5, 95))
        gray[valid] = np.uint8(255*np.clip((image[valid]-near)/max(far-near, 1e-6), 0, 1))
    return _panel(gray, title)


def _lidar_panel(frame, width=320, height=240, extent_m=1.0):
    import cv2
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = 30
    origin = np.asarray(frame.origin)
    for point in frame.valid_points:
        dx, dy = point[:2]-origin[:2]
        x = int(width/2 + dx*width/(2*extent_m))
        y = int(height/2 - dy*height/(2*extent_m))
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image, (x, y), 1, (180, 255, 180), -1)
    cv2.circle(image, (width//2, height//2), 4, (255, 255, 255), -1)
    cv2.putText(image, "LIDAR - vista XY", (8, 16), cv2.FONT_HERSHEY_SIMPLEX,
                .5, (255, 255, 255), 1)
    return image


def preview_sensors(seed=42, object_count=3, *, output_path=None,
                    headless=False, seconds=None):
    import cv2
    import gymnasium as gym
    import physical_ai_mujoco.envs  # noqa: F401

    if seconds is not None and seconds <= 0:
        raise ValueError("seconds deve essere positivo")
    env = gym.make("TargetExtraction-v0", object_count=object_count,
                   obs_mode="state", render_mode=None if headless else "human",
                   highlight_target=False, disable_env_checker=True)
    window = "Sensori: stereo | depth | LiDAR (q per chiudere)"
    opened = False
    try:
        _, info = env.reset(seed=seed)
        simulator = env.unwrapped.simulator
        frame = SimulatedStereoCamera(with_depth=True).capture(simulator)
        bundle = BundleBuilder(OracleDetector(simulator), seed=seed).build(frame)
        lidar = scan(simulator, LidarConfig(
            n_rays=180, n_planes=8, fov_horizontal_deg=120,
            fov_vertical_deg=45, origin=(0, -.55, .25), max_range=2.0,
            rotation=((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        ))
        panels = [
            _panel(frame.gray_left, "SINISTRA"),
            _panel(frame.gray_right, "DESTRA"),
            _depth_panel(frame.depth_left, "DEPTH SINISTRA"),
            _lidar_panel(lidar),
        ]
        canvas = np.concatenate((np.concatenate(panels[:2], axis=1),
                                 np.concatenate(panels[2:], axis=1)), axis=0)
        destination = Path(output_path or Path(__file__).resolve().parents[2] /
                           "outputs/observe_tests" / f"sensor_suite_seed_{seed}.png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), canvas):
            raise RuntimeError(f"Impossibile salvare {destination}")
        print(f"Scena {info['scene_seed']} | oggetti {object_count} | target {env.unwrapped.target_id}")
        print(f"Maschere oracle L/R: {len(bundle.left_masks)}/{len(bundle.right_masks)}")
        print(f"LiDAR: {int(lidar.valid.sum())}/{len(lidar.valid)} ritorni validi")
        print(f"Anteprima: {destination}")
        if not headless:
            cv2.imshow(window, canvas)
            opened = True
            started = time.monotonic()
            while True:
                env.render()
                key = cv2.waitKey(30) & 0xff
                if key in (ord("q"), 27):
                    break
                if seconds is not None and time.monotonic()-started >= seconds:
                    break
                try:
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
        return destination
    finally:
        if opened:
            cv2.destroyWindow(window)
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ispezione stereo, RGB-D e LiDAR")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objects", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args(argv)
    preview_sensors(args.seed, args.objects, output_path=args.output,
                    headless=args.headless, seconds=args.seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
