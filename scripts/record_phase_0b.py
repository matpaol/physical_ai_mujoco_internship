"""Registra un episodio come GIF o video, senza finestra e senza HUD.

E' il modo in cui sono prodotte le immagini della documentazione di Gymnasium:
rendering OFFSCREEN in `rgb_array`, cioe' una matrice di pixel `numpy`, salvata
con `imageio`. Niente finestra del sistema operativo, niente barra del titolo,
niente overlay di debug — resta solo il riquadro 3D.

Da confrontare con gli altri due modi di guardare:

    render_mode="human"          -> finestra GLFW + HUD disegnato da Gymnasium
    scripts/watch_phase_0b.py    -> finestra OpenCV + overlay informativo
    questo script                -> nessuna finestra, nessun overlay

Uso
---
    python -m scripts.record_phase_0b                        # GIF, 3 oggetti
    python -m scripts.record_phase_0b --objects 5 --format mp4
    python -m scripts.record_phase_0b --camera cam_left      # vista dello student
    python -m scripts.record_phase_0b --no-highlight         # target non evidenziato

Su server o senza display serve MUJOCO_GL=egl (o osmesa).

Nota sul perche' non si usa `gymnasium.wrappers.RecordVideo`
------------------------------------------------------------
RecordVideo e' il wrapper standard, e cattura UN frame per `env.step()`. In
questo environment pero' uno step non e' un tick di fisica: e' "rimuovi un
oggetto e lascia riassestare la scena", cioe' quasi mezzo secondo di
simulazione. Il risultato sarebbe una sequenza di pose statiche, non un
filmato. Qui si usano invece i frame che l'environment cattura DURANTE
l'assestamento e restituisce in `info["frames"]`: sono decine per step, e
danno il movimento vero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

import physical_ai_mujoco.envs  # noqa: F401  (registra TargetExtraction-v0)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    arguments = parse_arguments()
    import imageio.v2 as imageio

    env = gym.make(
        "TargetExtraction-v0",
        obs_mode="state",
        object_count=arguments.objects,
        capture_camera=arguments.camera,
        capture_stride=arguments.stride,
        highlight_target=not arguments.no_highlight,
        # Nessuna finestra: il rendering e' offscreen.
        render_mode=None,
        realtime_factor=0,
        disable_env_checker=True,
    )

    destination = Path(arguments.output or default_output(arguments))
    destination.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    try:
        for episode in range(arguments.episodes):
            frames.extend(record_episode(env, episode, arguments))
    finally:
        env.close()

    if not frames:
        print("Nessun frame catturato.")
        return 1

    imageio.mimsave(destination, frames, fps=arguments.fps)
    print(
        f"\nSalvato: {destination}\n"
        f"{len(frames)} frame a {arguments.fps} fps "
        f"({len(frames) / arguments.fps:.1f} s), {frames[0].shape[1]}x{frames[0].shape[0]} px"
    )
    return 0


def record_episode(env, episode: int, arguments) -> list[np.ndarray]:
    _, info = env.reset(seed=arguments.seed + episode)
    generator = np.random.default_rng(arguments.seed + episode)

    print(f"=== Episodio {episode}: target {info['target_id']}")
    frames = list(info["frames"])
    done = False

    while not done:
        valid_actions = np.flatnonzero(info["action_mask"])
        action = int(generator.choice(valid_actions))
        _, reward, terminated, truncated, info = env.step(action)
        frames.extend(info["frames"])
        print(
            f"    rimuove indice {action} | reward {reward:+.3f} | "
            f"disturbo {info['disturbance_step']:.4f}"
        )
        done = terminated or truncated

    outcome = "successo" if info.get("is_success") else "fallimento"
    print(f"    esito: {outcome}")

    # Una pausa sull'ultimo fotogramma, cosi' il finale si legge.
    frames.extend([frames[-1]] * arguments.fps)
    return frames


def default_output(arguments) -> Path:
    suffix = "gif" if arguments.format == "gif" else "mp4"
    return PROJECT_ROOT / "outputs" / "video" / f"phase0b_{arguments.camera}.{suffix}"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--objects", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--camera",
        default="cam_overview",
        help="cam_overview | cam_left | cam_right",
    )
    parser.add_argument("--format", choices=["gif", "mp4"], default="gif")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="un frame ogni quanti passi di fisica (4 = molto fluido)",
    )
    parser.add_argument(
        "--no-highlight",
        action="store_true",
        help="non colorare di giallo il target",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
