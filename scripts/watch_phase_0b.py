"""Finestra persistente per guardare l'agente lavorare.

Perche' non basta il viewer di Gymnasium
----------------------------------------
Due motivi, e il secondo e' il piu' importante.

1. I viewer interattivi (nativo MuJoCo e Gymnasium) si legano a un modello
   COMPILATO. Qui il modello viene ricompilato a ogni reset, perche' forme e
   dimensioni degli oggetti sono campionate per scena. Una finestra
   interattiva dovrebbe quindi chiudersi e riaprirsi a ogni episodio. Questo
   script rende OFFSCREEN dentro una finestra OpenCV, che non appartiene a
   MuJoCo e sopravvive a qualunque ricompilazione.

2. In questo progetto **uno step non e' un tick di fisica**: e' "rimuovi un
   oggetto e lascia riassestare la scena". Un episodio sono pochi step, cioe'
   un paio di secondi di simulazione in tutto. Senza pause mirate l'episodio
   sfreccia via prima che si capisca cosa e' successo. Qui la scena si ferma
   sui momenti che contano: quando e' assestata, prima di ogni rimozione, e
   sull'esito finale.

Uso
---
    python -m scripts.watch_phase_0b                  # 1 scena, all'infinito
    python -m scripts.watch_phase_0b --objects 5
    python -m scripts.watch_phase_0b --speed 0.5      # meta' velocita'
    python -m scripts.watch_phase_0b --parallel 4     # 4 affiancate
    python -m scripts.watch_phase_0b --save video.mp4 # registra, niente finestra

L'oggetto GIALLO e' il target da estrarre. `q` chiude la finestra.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

import physical_ai_mujoco.envs  # noqa: F401  (registra TargetExtraction-v0)

WINDOW_TITLE = "Phase 0B - TargetExtraction"

# Pause, in secondi, sui momenti che contano.
HOLD_AFTER_SETTLE = 1.5
HOLD_BEFORE_REMOVAL = 1.0
HOLD_ON_OUTCOME = 2.5


@dataclass
class EnvRunner:
    """Un environment con lo stato di visualizzazione che lo accompagna."""

    index: int
    env: gym.Env
    generator: np.random.Generator
    fps: int
    info: dict = field(default_factory=dict)
    episode: int = 0
    step_number: int = 0
    last_reward: float = 0.0
    total_reward: float = 0.0
    # Il ritorno dell'ultimo episodio CONCLUSO. `total_reward` non va bene per
    # il riepilogo: viene azzerato da `start_episode`, quindi quando la
    # finestra si chiude mentre un episodio nuovo e' appena partito il
    # riepilogo stampa zero al posto del risultato vero.
    last_return: float | None = None
    # True quando questo riquadro ha finito gli episodi richiesti. Serve a
    # contarlo UNA volta sola: senza, ogni fotogramma successivo lo contava di
    # nuovo e la sessione finiva prima che gli altri riquadri avessero finito.
    done: bool = False
    phase: str = ""
    outcome: str = ""
    pending_frames: list = field(default_factory=list)
    last_frame: np.ndarray | None = None

    # -------------------------------------------------------------- episodio

    def start_episode(self, seed: int) -> None:
        _, self.info = self.env.reset(seed=seed)
        self.step_number = 0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.outcome = ""
        self.phase = "la scena cade e si assesta"

        self.pending_frames = list(self.info.get("frames", []))
        if self.pending_frames:
            self.last_frame = self.pending_frames[-1]
        self.hold(HOLD_AFTER_SETTLE, phase_after="scena assestata")

    def advance(self) -> None:
        """Sceglie un'azione, la esegue, e accoda i frame con le pause."""
        valid_actions = np.flatnonzero(self.info["action_mask"])
        action = int(self.generator.choice(valid_actions))
        removed = self.env.unwrapped._object_ids[action]

        # Pausa PRIMA della rimozione: si vede la configurazione di partenza,
        # e si legge quale oggetto sta per sparire.
        self.phase = f"sto per togliere {removed}"
        self.hold(HOLD_BEFORE_REMOVAL)

        _, reward, terminated, truncated, self.info = self.env.step(action)
        self.step_number += 1
        self.last_reward = reward
        self.total_reward += reward
        self.phase = f"tolto {removed}, la scena si riassesta"
        self.pending_frames.extend(self.info.get("frames", []))

        if terminated or truncated:
            self.outcome = (
                "SUCCESSO" if self.info.get("is_success") else "FALLIMENTO"
            )
            self.last_return = self.total_reward
            self.hold(HOLD_ON_OUTCOME)

    def hold(self, seconds: float, phase_after: str | None = None) -> None:
        """Congela l'ultimo frame per un po', ripetendolo."""
        frozen = None
        if self.pending_frames:
            frozen = self.pending_frames[-1]
        elif self.last_frame is not None:
            frozen = self.last_frame
        if frozen is None:
            return

        self.pending_frames.extend([frozen] * int(seconds * self.fps))
        if phase_after is not None:
            self.phase = phase_after

    # ----------------------------------------------------------------- frame

    def next_frame(self) -> np.ndarray | None:
        if self.pending_frames:
            self.last_frame = self.pending_frames.pop(0)
        return self.last_frame

    def has_frames(self) -> bool:
        return bool(self.pending_frames)


def main() -> int:
    arguments = parse_arguments()
    show_window = arguments.save is None
    cv2 = import_opencv(required=show_window)

    runners = [
        EnvRunner(
            index=index,
            env=gym.make(
                "TargetExtraction-v0",
                obs_mode="state",
                object_count=arguments.objects,
                capture_camera=arguments.camera,
                capture_stride=arguments.stride,
                # Il target diventa giallo: senza, guardando la scena non si
                # capisce quale oggetto conta.
                highlight_target=True,
                disable_env_checker=True,
            ),
            generator=np.random.default_rng(arguments.seed + index),
            fps=arguments.fps,
        )
        for index in range(arguments.parallel)
    ]

    for runner in runners:
        runner.start_episode(arguments.seed + runner.index)

    writer = None
    if arguments.save:
        import imageio.v2 as imageio

        writer = imageio.get_writer(arguments.save, fps=arguments.fps)

    if show_window:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, 960, 720)

    print(
        "L'oggetto GIALLO e' il target da estrarre.\n"
        + (
            "Premi 'q' nella finestra per uscire."
            if show_window
            else f"Registro in {arguments.save}."
        )
    )

    frame_interval = 1.0 / (arguments.fps * arguments.speed)
    finished = 0
    running = True

    try:
        while running:
            for runner in runners:
                if runner.done or runner.has_frames():
                    continue
                if runner.outcome:
                    runner.episode += 1
                    if arguments.episodes and runner.episode >= arguments.episodes:
                        runner.done = True
                        finished += 1
                        if finished >= len(runners):
                            running = False
                        continue
                    seed = arguments.seed + runner.index + 1000 * runner.episode
                    runner.start_episode(seed)
                else:
                    runner.advance()

            canvas = compose_grid(runners, arguments.columns)
            if writer is not None:
                writer.append_data(canvas)
            if show_window:
                cv2.imshow(WINDOW_TITLE, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    running = False
                time.sleep(frame_interval)
    except KeyboardInterrupt:
        print("\nInterrotto.")
    finally:
        if writer is not None:
            writer.close()
            print(f"Salvato: {arguments.save}")
        if show_window:
            cv2.destroyAllWindows()
        for runner in runners:
            runner.env.close()

    print("\nRiepilogo:")
    for runner in runners:
        quanti = runner.episode
        parola = "episodio completato" if quanti == 1 else "episodi completati"
        ritorno = (
            f"ultimo ritorno {runner.last_return:+.3f}"
            if runner.last_return is not None
            else "nessun episodio concluso"
        )
        print(f"  scena {runner.index}: {quanti} {parola}, {ritorno}")
    return 0


def compose_grid(runners: list[EnvRunner], columns: int | None) -> np.ndarray:
    frames = [overlay_text(runner) for runner in runners]
    height, width = frames[0].shape[:2]

    count = len(frames)
    columns = columns or int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / columns))

    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for position, frame in enumerate(frames):
        row, column = divmod(position, columns)
        canvas[
            row * height : (row + 1) * height,
            column * width : (column + 1) * width,
        ] = frame
    return canvas


def overlay_text(runner: EnvRunner) -> np.ndarray:
    frame = runner.next_frame()
    if frame is None:
        return np.zeros((240, 320, 3), dtype=np.uint8)

    try:
        import cv2
    except ImportError:
        return frame

    canvas = frame.copy()
    lines = [
        (f"episodio {runner.episode}  -  step {runner.step_number}", (255, 255, 255)),
        (f"target (giallo): {runner.info.get('target_id', '?')}", (255, 220, 60)),
        (runner.phase, (200, 200, 200)),
        (
            f"reward {runner.last_reward:+.2f}   "
            f"disturbo {runner.info.get('disturbance_step', 0.0):.3f}",
            (255, 255, 255),
        ),
    ]
    if runner.outcome:
        colour = (60, 230, 60) if runner.outcome == "SUCCESSO" else (255, 80, 80)
        lines.append((runner.outcome, colour))

    for line_number, (line, colour) in enumerate(lines):
        position = (6, 16 + 15 * line_number)
        cv2.putText(
            canvas, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3
        )
        cv2.putText(
            canvas, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1
        )
    return canvas


def import_opencv(required: bool):
    try:
        import cv2

        return cv2
    except ImportError:
        if required:
            raise SystemExit(
                "Serve OpenCV per la finestra. Reinstalla il progetto:\n"
                '  python -m pip install -e ".[test]"\n'
                "Oppure registra un video: --save video.mp4"
            )
        return None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--objects",
        type=int,
        default=None,
        help="quanti oggetti nella scena (predefinito: quelli del config)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="quante scene affiancare nella stessa finestra",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="episodi per scena (0 = finche' non chiudi la finestra)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="1 = normale, 0.5 = meta' velocita', 2 = doppia",
    )
    parser.add_argument("--columns", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="un frame ogni quanti passi di fisica (8 ~ 60 fps simulati)",
    )
    parser.add_argument("--camera", default="cam_overview")
    parser.add_argument(
        "--save", default=None, help="salva un video invece di aprire la finestra"
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
