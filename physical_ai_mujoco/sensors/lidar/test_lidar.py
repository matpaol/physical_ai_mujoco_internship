"""Test visivo autonomo del LiDAR dentro il suo package.

Esempio: python -m physical_ai_mujoco.sensors.lidar.test_lidar --seed 42
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from physical_ai_mujoco.sensors.lidar import LidarConfig, scan


def preview_lidar(seed=42, object_count=3, *, output_path=None,
                  headless=False, seconds=None):
    import cv2
    import gymnasium as gym
    import physical_ai_mujoco.envs  # noqa: F401

    if seconds is not None and seconds <= 0:
        raise ValueError("seconds deve essere positivo")
    env = gym.make("TargetExtraction-v0", object_count=object_count,
                   obs_mode="state", render_mode=None if headless else "human",
                   highlight_target=False, disable_env_checker=True)
    window = "LiDAR simulato - q per chiudere"
    opened = False
    try:
        _, info = env.reset(seed=seed)
        config = LidarConfig(n_rays=180, n_planes=16, fov_horizontal_deg=120,
                             fov_vertical_deg=45, origin=(0, -.55, .25),
                             rotation=((0, -1, 0), (1, 0, 0), (0, 0, 1)),
                             max_range=2.0)
        frame = scan(env.unwrapped.simulator, config)
        image = np.full((480, 640, 3), 25, dtype=np.uint8)
        origin = np.asarray(frame.origin)
        for point in frame.valid_points:
            x = int(320 + (point[0]-origin[0])*300)
            y = int(240 - (point[1]-origin[1])*300)
            if 0 <= x < 640 and 0 <= y < 480:
                cv2.circle(image, (x, y), 1, (180, 255, 180), -1)
        cv2.circle(image, (320, 240), 5, (255, 255, 255), -1)
        cv2.putText(image, f"{frame.valid.sum()}/{len(frame.valid)} ritorni validi",
                    (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
        destination = Path(output_path or Path(__file__).resolve().parents[3] /
                           "outputs/observe_tests" / f"lidar_preview_seed_{seed}.png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), image):
            raise RuntimeError(f"Impossibile salvare {destination}")
        print(f"Scena {info['scene_seed']} | raggi {len(frame.ranges)} | hit {frame.valid.sum()}")
        print(f"Vista LiDAR salvata: {destination}")
        if not headless:
            cv2.imshow(window, image)
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
    parser = argparse.ArgumentParser(description="Test visivo del LiDAR")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objects", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args(argv)
    preview_lidar(args.seed, args.objects, output_path=args.output,
                  headless=args.headless, seconds=args.seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
