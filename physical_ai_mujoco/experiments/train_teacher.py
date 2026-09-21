"""Allena con PPO la policy che sceglie l'ordine di rimozione.

Il compito e': togliere gli oggetti fino a estrarre il target, smuovendo il
mucchio il meno possibile. L'episodio finisce quando il target esce
(`terminate_on_target`), quindi **conta anche quanto si scava**: ogni rimozione
in piu' costa, e l'ottimo toglie solo cio' che grava davvero sul target.

**Senza maschera delle azioni.** La maschera e' informazione privilegiata: dice
quali oggetti sono ancora in scena, e nel mondo vero quell'elenco non arriva
gratis — va dedotto da cio' che si vede. Una policy allenata con la maschera
non imparerebbe a dedurlo, e non si trasferirebbe allo student che guarda solo
le telecamere. L'environment penalizza le azioni non valide invece di
rifiutarle, quindi la policy impara a evitarle da sola.

    python scripts/allena.py --oggetti 6 --passi 200000
    python scripts/allena.py --oggetti 6 --passi 20000 --paralleli 4

Alla fine confronta il modello con le politiche di riferimento, sulle stesse
scene di valutazione. I valori dipendono dalla configurazione registrata.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym  # noqa: E402

import physical_ai_mujoco.envs  # noqa: E402,F401  (registra l'environment)

USCITA_PREDEFINITA = PROJECT_ROOT / "outputs" / "modelli"


def _training_obs_mode() -> str:
    from physical_ai_mujoco.infrastructure.experiment import selected_profile

    profile = selected_profile()
    return "sensor" if profile is not None and profile.path.stem.lower() in {"1b", "1c"} else "state"


def crea_env(
    oggetti: int, seed: int | None = None, magazzino: int = 0, fresche: float = 0.1
):
    """Costruisce un environment. Definita a livello di modulo di proposito.

    `SubprocVecEnv` manda la funzione ai processi figli con pickle, e una
    chiusura o una lambda non sono picklabili: la fabbrica deve poter essere
    ricostruita dal solo nome del modulo piu' i parametri.
    """
    from stable_baselines3.common.monitor import Monitor

    env = gym.make(
        "TargetExtraction-v0",
        disable_env_checker=True,
        obs_mode=_training_obs_mode(),
        object_count=oggetti,
        scene_pool_size=magazzino,
        fresh_scene_probability=fresche,
    )
    if seed is not None:
        env.reset(seed=seed)
    return Monitor(env)


def _fabbrica(oggetti: int, seed: int, magazzino: int, fresche: float):
    def costruisci():
        return crea_env(oggetti, seed, magazzino, fresche)

    return costruisci


def costruisci_vettore(
    oggetti: int, paralleli: int, seed: int, magazzino: int, fresche: float
):
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecNormalize,
    )

    fabbriche = [
        _fabbrica(oggetti, seed + indice, magazzino, fresche)
        for indice in range(paralleli)
    ]
    vettore = DummyVecEnv(fabbriche) if paralleli == 1 else SubprocVecEnv(fabbriche)
    # Le osservazioni mescolano metri, chilogrammi e coefficienti d'attrito:
    # senza normalizzare, la rete vede una manciata di numeri grandi e tutto
    # il resto schiacciato a zero. Le ricompense NO: normalizzarle
    # renderebbe i ritorni non confrontabili con quelli delle politiche di
    # riferimento, che e' l'unica misura che abbiamo.
    return VecNormalize(vettore, norm_obs=True, norm_reward=False)


# ---------------------------------------------------------------- confronto

from physical_ai_mujoco.infrastructure.policy_adapter import alto, casuale  # noqa: E402


def valuta(scegli, oggetti: int, scene: int) -> np.ndarray:
    """Confronto su scene COSTRUITE DA ZERO, mai passate dal magazzino.

    Il magazzino e disattivato. I seed restano quelli della baseline; una
    separazione formale dei dataset e un requisito della validazione 1A.
    """
    env = gym.make(
        "TargetExtraction-v0",
        disable_env_checker=True,
        obs_mode=_training_obs_mode(),
        object_count=oggetti,
    )
    from physical_ai_mujoco.evaluation.evaluator import PolicyEvaluator

    try:
        results = PolicyEvaluator().evaluate(env, scegli, range(scene))
        return np.asarray([result.reward for result in results])
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oggetti", type=int, default=6)
    parser.add_argument("--passi", type=int, default=200_000)
    parser.add_argument("--paralleli", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--magazzino",
        type=int,
        default=200,
        help="quante scene tenere gia' costruite (0 = ricostruisci sempre)",
    )
    parser.add_argument(
        "--scene-fresche",
        type=float,
        default=0.1,
        help="con che probabilita' costruirne una nuova a magazzino pieno",
    )
    parser.add_argument("--scene-di-prova", type=int, default=20)
    parser.add_argument("--uscita", type=str, default=str(USCITA_PREDEFINITA))
    argomenti = parser.parse_args()

    from stable_baselines3 import PPO
    from physical_ai_mujoco.experiments.metadata import save_run_metadata

    uscita = Path(argomenti.uscita)
    uscita.mkdir(parents=True, exist_ok=True)
    from physical_ai_mujoco.infrastructure.experiment import selected_profile

    profile = selected_profile()
    phase_code = None if profile is None else profile.path.stem.lower()
    sensor_training = phase_code in {"1b", "1c"}
    prefix = (
        "ppo_sensor_robust"
        if phase_code == "1c"
        else "ppo_sensor" if sensor_training else "ppo"
    )
    nome = f"{prefix}_{argomenti.oggetti}oggetti_{argomenti.passi}passi"

    print(
        f"\nAlleno PPO{' sensoriale' if sensor_training else ''}: "
        f"{argomenti.oggetti} oggetti, {argomenti.passi} passi, "
        f"{argomenti.paralleli} ambienti in parallelo, senza maschera"
    )
    if argomenti.magazzino:
        print(
            f"Magazzino: {argomenti.magazzino} scene per ambiente "
            f"({argomenti.scene_fresche:.0%} costruite nuove a regime). "
            "I primi episodi sono lenti: e' il riempimento.\n"
        )
    else:
        print("Magazzino spento: ogni episodio ricostruisce la scena.\n")

    vettore = costruisci_vettore(
        argomenti.oggetti,
        argomenti.paralleli,
        argomenti.seed,
        argomenti.magazzino,
        argomenti.scene_fresche,
    )
    try:
        modello = PPO(
            "MlpPolicy",
            vettore,
            seed=argomenti.seed,
            verbose=1,
            # Gli episodi durano pochi passi (l'ottimo ne usa 1,7): con rollout
            # lunghi un aggiornamento vedrebbe centinaia di episodi e pochissima
            # varieta' di scene. Meglio corti e frequenti.
            n_steps=256,
            batch_size=256,
            gamma=0.99,
        )

        save_run_metadata(uscita / f"{nome}_run.json", vars(argomenti))
        inizio = time.perf_counter()
        try:
            modello.learn(total_timesteps=argomenti.passi, progress_bar=False)
        except KeyboardInterrupt:
            print("\nInterrotto: salvo comunque quello che c'e'.")
        durata = time.perf_counter() - inizio

        percorso = uscita / nome
        modello.save(percorso)
        vettore.save(str(uscita / f"{nome}_normalizzazione.pkl"))
        print(f"\nModello salvato in {percorso}.zip   ({durata / 60:.1f} minuti)\n")

        # --------------------------------------------------------- il confronto

        def politica_appresa(env, info, rng, osservazione):
            # L'osservazione va normalizzata con LE STESSE statistiche
            # dell'allenamento. Passare quella grezza a un modello allenato su
            # osservazioni normalizzate non solleva nessun errore: la forma e' la
            # stessa, i numeri no, e la policy sceglie a caso senza dirlo. E'
            # successo davvero: -0,587 in valutazione contro +0,713 in
            # allenamento, prima di accorgersene.
            azione, _ = modello.predict(
                vettore.normalize_obs(osservazione), deterministic=True
            )
            return int(azione)

        print(
            f"Confronto su {argomenti.scene_di_prova} scene ricostruite di valutazione:\n"
        )
        righe = {"casuale": casuale, "alto": alto, "PPO": politica_appresa}
        metrics = {}
        for etichetta, scegli in righe.items():
            ritorni = valuta(scegli, argomenti.oggetti, argomenti.scene_di_prova)
            metrics[etichetta] = ritorni.tolist()
            print(f"  {etichetta:10s} {ritorni.mean():+8.3f} ± {ritorni.std():5.3f}")
        save_run_metadata(
            uscita / f"{nome}_run.json",
            {
                **vars(argomenti),
                "actual_timesteps": modello.num_timesteps,
                "evaluation_seeds": list(range(argomenti.scene_di_prova)),
            },
            metrics,
        )

        # Il tetto dipende dal numero di oggetti e dalle scene: citare un numero
        # misurato su una configurazione diversa sarebbe un confronto falso.
        print(
            "\n  Il metro e' la riga `alto`: un modello che non la batte non ha "
            "imparato\n  niente che un'euristica da una riga non sapesse gia'."
            "\n  Per il tetto vero su QUESTA configurazione:"
            "\n    python scripts/quanto_margine.py --oggetti "
            f"{argomenti.oggetti} --finisci-al-target"
        )
    finally:
        vettore.close()


if __name__ == "__main__":
    main()
