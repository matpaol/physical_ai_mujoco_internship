"""Phase 0B: rollout dell'environment Gymnasium di estrazione del target.

Lanciato senza argomenti fa due domande — quanti oggetti, e se aprire la
finestra — poi parte. Ogni scelta e' anche un flag, per saltare le domande:

    python -m scripts.run_phase_0b                          # interattivo
    python -m scripts.run_phase_0b --objects 5 --render     # diretto
    python -m scripts.run_phase_0b --objects 3 --headless   # solo testo
    python -m scripts.run_phase_0b --obs-mode both --save-stereo outputs/stereo

La finestra
-----------
`--render` apre il viewer di Gymnasium, che si richiude e riapre a ogni
episodio perche' il modello MuJoCo viene ricompilato (le forme degli oggetti
sono campionate per scena). Per una finestra che resta aperta usa `main.py`,
che rende offscreen dentro una finestra non legata a MuJoCo.

`--fixed-shapes` tiene le stesse forme in tutti gli episodi. E' una scelta
sulla FISICA, non sulla grafica: riduce la varieta' delle scene, quindi
durante l'allenamento va lasciata spenta.

Su server o senza display usare MUJOCO_GL=egl e --headless.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

import physical_ai_mujoco.envs  # noqa: F401  (registra TargetExtraction-v0)
from physical_ai_mujoco.evaluation.inspection import (
    stampa_appoggi,
    stampa_catalogo,
    stampa_costruzione,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    arguments = parse_arguments()
    object_count = select_object_count(arguments)
    render = select_render_mode(arguments)

    env = gym.make(
        "TargetExtraction-v0",
        obs_mode=arguments.obs_mode,
        render_mode="human" if render else None,
        object_count=object_count,
        # La randomizzazione delle forme NON dipende dalla finestra: legarla
        # al rendering significherebbe simulare fisiche diverse a seconda di
        # come si guarda, e confrontare esperimenti diventerebbe impossibile.
        resample_shapes=not arguments.fixed_shapes,
    )

    save_directory = None
    if arguments.save_stereo:
        if arguments.obs_mode == "state":
            print("[avviso] --save-stereo ignorato: obs-mode 'state' non ha immagini")
        else:
            save_directory = Path(arguments.save_stereo)
            save_directory.mkdir(parents=True, exist_ok=True)

    print(
        f"\n{object_count} oggetti | osservazione '{arguments.obs_mode}' | "
        f"finestra {'aperta' if render else 'chiusa'}"
    )
    if render and not arguments.fixed_shapes:
        print(
            "Nota: la finestra di Gymnasium si richiude a ogni episodio, "
            "perche' il modello\nviene ricompilato. Per una finestra che "
            "resta aperta usa: python main.py"
        )

    successes = 0
    try:
        for episode in range(arguments.episodes):
            successes += run_episode(env, episode, arguments, save_directory)
    finally:
        env.close()

    print(f"\nSuccessi: {successes}/{arguments.episodes}")
    return 0


def run_episode(
    env,
    episode: int,
    arguments,
    save_directory: Path | None,
    scegli=None,
) -> int:
    """Un episodio completo. `scegli` decide le mosse.

    Senza, si usa la policy casuale mascherata: e' il riferimento piu' basso,
    e il comportamento di prima. Passandone un'altra — l'euristica, o un
    modello allenato — si guarda la stessa scena con un decisore diverso,
    che e' l'unico modo di vedere se un allenamento e' servito a qualcosa.
    """
    if scegli is None:
        from physical_ai_mujoco.infrastructure.policy_adapter import casuale as scegli

    observation, info = env.reset(seed=arguments.seed + episode)
    generator = np.random.default_rng(arguments.seed + episode)
    inner = env.unwrapped

    print(f"\n=== Episodio {episode}")
    print(f"    seed scena : {info['scene_seed']}")
    print(f"    target     : {info['target_id']} (indice {info['target_index']})")
    print(f"    oggetti    : {', '.join(info['present_objects'])}")

    # Cosa c'e' in scena e come sta appoggiato, PRIMA che la policy scelga.
    # Senza questo la sequenza di indici che segue non dice niente: non si sa
    # cosa fosse l'indice 2, ne' se stava sopra o sotto il target.
    if not getattr(arguments, "no_catalogue", False):
        stampa_costruzione(info)
        stampa_catalogo(inner)
        stampa_appoggi(inner)

    if save_directory is not None:
        save_stereo_pair(observation, save_directory, episode, 0)

    total_reward = 0.0
    step_number = 0
    done = False
    # Ordine effettivo delle rimozioni: i tentativi su oggetti gia' rimossi
    # non entrano, perche' non rimuovono niente.
    removal_order: list[str] = []

    print(
        f"\n    {'#':>2}  {'azione':>6}  {'oggetto':<12} {'reward':>8} "
        f"{'disturbo':>9}  esito"
    )
    print("    " + "-" * 62)

    while not done:
        action = int(scegli(env, info, generator, observation))
        name = inner.simulator.scene.objects[action].instance_id

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_number += 1

        notes = []
        if info.get("invalid_action"):
            notes.append("non valida (gia' rimosso)")
        else:
            removal_order.append(name)
        if info.get("target_just_removed"):
            notes.append("TARGET RIMOSSO")
        if info.get("collapsed"):
            notes.append("CROLLO: disturbo oltre soglia")
        if truncated:
            notes.append("troncato: passi esauriti")

        print(
            f"    {step_number:>2}  {action:>6}  {name:<12} {reward:>+8.3f} "
            f"{info['disturbance_step']:>9.4f}  {' · '.join(notes)}"
        )

        if save_directory is not None:
            save_stereo_pair(observation, save_directory, episode, step_number)

        done = terminated or truncated

    outcome = "successo" if info.get("is_success") else "fallimento"
    print("    " + "-" * 62)
    print(f"    ordine di rimozione: {format_removal_order(removal_order, inner)}")
    print(f"    esito: {outcome} | ritorno {total_reward:+.3f}")
    return int(bool(info.get("is_success")))


def format_removal_order(removal_order: list[str], inner) -> str:
    """La sequenza di oggetti effettivamente rimossi, in ordine.

    E' la risposta che la policy ha dato per questa scena: e' questa sequenza,
    non la ricompensa, che si confronta fra politiche diverse.
    """
    if not removal_order:
        return "(nessuna rimozione)"
    etichette = [
        f"{name} (TARGET)" if name == inner.target_id else name
        for name in removal_order
    ]
    rimasti = len(inner.simulator.scene.objects) - len(removal_order)
    coda = f"   [{rimasti} rimasti in scena]" if rimasti else ""
    return " → ".join(etichette) + coda


def save_stereo_pair(
    observation,
    directory: Path,
    episode: int,
    step_number: int,
) -> None:
    import imageio.v2 as imageio

    for side in ("rgb_left", "rgb_right"):
        frame = observation[side]
        name = f"ep{episode:02d}_step{step_number:02d}_{side}.png"
        imageio.imwrite(directory / name, frame)


# ------------------------------------------------------------- scelte iniziali


def select_object_count(arguments: argparse.Namespace) -> int:
    if arguments.objects is not None:
        return arguments.objects

    while True:
        answer = input("Quanti oggetti nella scena? [1-12, invio = 3]: ").strip()
        if not answer:
            return 3
        if answer.isdigit() and 1 <= int(answer) <= 12:
            return int(answer)
        print("Inserire un numero fra 1 e 12.")


def select_render_mode(arguments: argparse.Namespace) -> bool:
    if arguments.render:
        return True
    if arguments.headless:
        return False

    while True:
        answer = input("Aprire la finestra della simulazione? [y/n]: ").strip().lower()
        if answer in {"y", "yes", "s", "si"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Rispondere y o n.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--objects",
        type=int,
        default=None,
        help="quanti oggetti nella scena (se assente viene chiesto)",
    )
    parser.add_argument(
        "--obs-mode",
        choices=["state", "stereo", "both"],
        default="state",
        help="state = teacher, stereo = student, both = distillazione",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--render",
        action="store_true",
        help="apre il viewer di Gymnasium (si richiude a ogni episodio)",
    )
    mode.add_argument(
        "--headless", action="store_true", help="nessuna finestra, solo testo"
    )

    parser.add_argument(
        "--fixed-shapes",
        action="store_true",
        help="stesse forme in tutti gli episodi (cambiano solo le pose)",
    )
    parser.add_argument(
        "--save-stereo",
        default=None,
        help="cartella dove salvare le coppie stereo (richiede --obs-mode stereo/both)",
    )
    parser.add_argument(
        "--no-catalogue",
        action="store_true",
        help="niente catalogo ne' grafo degli appoggi: solo azioni ed esito",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
