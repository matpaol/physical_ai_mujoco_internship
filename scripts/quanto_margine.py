"""Quanto si puo' guadagnare imparando l'ordine? Il margine, misurato.

E' la domanda da farsi PRIMA di allenare qualsiasi policy. Se l'euristica
banale — togli sempre quello piu' in alto — e' gia' vicina al meglio
raggiungibile, una policy appresa non ha niente da guadagnare, e il compito va
reso piu' difficile prima di spenderci sopra un allenamento.

Confronta quattro politiche sulle STESSE scene:

- `casuale`  : fra le azioni valide, a caso. Il pavimento.
- `alto`     : sempre l'oggetto piu' in alto. L'euristica ovvia.
- `ottimo`   : il MIGLIOR ordine possibile, trovato provandoli tutti. Non e'
               una politica: e' il tetto vero, quello che nessuna policy puo'
               superare. E' anche, a scene piccole, un maestro da imitare.

`ottimo` si permette di provare e disfare perche' lo stato del simulatore
viene fotografato e ripristinato: i prefissi comuni si simulano una volta
sola, nessuna scena viene ricostruita, e il confronto e' a parita' di scena.

    python scripts/quanto_margine.py --oggetti 6 --scene 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym  # noqa: E402

import physical_ai_mujoco.envs  # noqa: E402,F401  (registra l'environment)


def fotografa(env) -> dict:
    """Lo stato completo da cui si puo' ripartire: simulatore piu' contabilita'.

    Le pose non bastano. `disturbance` si misura rispetto a
    `_reference_positions`, che cambia a ogni mossa: ripristinare la fisica
    senza ripristinare anche quel riferimento farebbe misurare il disturbo
    rispetto a un istante sbagliato.
    """
    interno = env.unwrapped
    simulatore = interno.simulator
    return {
        "qpos": simulatore.data.qpos.copy(),
        "qvel": simulatore.data.qvel.copy(),
        "tempo": float(simulatore.data.time),
        "contype": simulatore.model.geom_contype.copy(),
        "conaffinity": simulatore.model.geom_conaffinity.copy(),
        "rimossi": set(simulatore._removed),
        "riferimento": {
            nome: posizione.copy()
            for nome, posizione in interno._reference_positions.items()
        },
        "disturbo_totale": float(interno._total_disturbance),
        # Il contatore di `TimeLimit`. Senza, la ricerca esaustiva lo fa
        # crescere di migliaia di passi senza mai un reset: superato
        # `max_episode_steps` ogni ramo viene TRONCATO a meta', e un ramo
        # troncato dopo due mosse risulta migliore di uno completo perche' ha
        # pagato due rimozioni invece di cinque. E' come la ricerca aveva
        # trovato ritorni sopra il tetto teorico.
        "passi_trascorsi": getattr(env, "_elapsed_steps", None),
    }


def ripristina(env, istantanea: dict) -> None:
    import mujoco

    interno = env.unwrapped
    simulatore = interno.simulator
    simulatore.data.qpos[:] = istantanea["qpos"]
    simulatore.data.qvel[:] = istantanea["qvel"]
    simulatore.data.time = istantanea["tempo"]
    simulatore.model.geom_contype[:] = istantanea["contype"]
    simulatore.model.geom_conaffinity[:] = istantanea["conaffinity"]
    simulatore._removed = set(istantanea["rimossi"])
    mujoco.mj_forward(simulatore.model, simulatore.data)
    interno._reference_positions = {
        nome: posizione.copy()
        for nome, posizione in istantanea["riferimento"].items()
    }
    interno._total_disturbance = istantanea["disturbo_totale"]
    if istantanea["passi_trascorsi"] is not None:
        env._elapsed_steps = istantanea["passi_trascorsi"]


def azione_casuale(env, info, rng):
    return int(rng.choice(np.flatnonzero(info["action_mask"])))


def azione_alta(env, info, rng):
    interno = env.unwrapped
    stati = interno.simulator.get_present_object_states()
    validi = np.flatnonzero(info["action_mask"])
    quote = [stati[interno._object_ids[a]].position[2] for a in validi]
    return int(validi[int(np.argmax(quote))])


def _esplora(env, info, ritorno_finora: float, percorso: list, migliore: dict) -> None:
    """Ricerca esaustiva su tutti gli ordini di rimozione possibili.

    Lo stato del simulatore viene fotografato prima di ogni ramo e
    ripristinato dopo: i prefissi comuni si simulano una volta sola, e nessuna
    scena viene ricostruita. E' quello che rende praticabile l'esaustiva.
    """
    validi = [int(a) for a in np.flatnonzero(info["action_mask"])]
    if not validi:
        return

    istantanea = fotografa(env)
    for azione in validi:
        _, ricompensa, terminato, troncato, esito = env.step(azione)
        totale = ritorno_finora + ricompensa
        percorso.append(azione)
        if terminato or troncato:
            if totale > migliore["ritorno"]:
                migliore["ritorno"] = totale
                migliore["passi"] = len(percorso)
                migliore["ordine"] = list(percorso)
        else:
            _esplora(env, esito, totale, percorso, migliore)
        percorso.pop()
        ripristina(env, istantanea)


def ottimo(env, seed: int) -> dict:
    """Il ritorno migliore raggiungibile su questa scena, cercato per intero."""
    _, info = env.reset(seed=seed)
    sepolto = target_sepolto(env)
    migliore = {"ritorno": -float("inf"), "passi": 0, "ordine": []}
    _esplora(env, info, 0.0, [], migliore)
    return {
        "ritorno": migliore["ritorno"],
        "passi": migliore["passi"],
        "ordine": migliore["ordine"],
        "disturbo_totale": float("nan"),
        "mossa_peggiore": float("nan"),
        "sepolto": sepolto,
    }


POLITICHE = {
    "casuale": azione_casuale,
    "alto": azione_alta,
}


def target_sepolto(env) -> bool:
    """True se qualcosa grava sul target, anche per vie indirette.

    Nelle scene in cui il target e' scoperto l'ordine non conta: si prende e
    basta. Contarle insieme alle altre nasconde il margine vero.
    """
    interno = env.unwrapped
    grafo = interno.simulator.support_graph()
    sopra, frontiera = set(), [interno.target_id]
    while frontiera:
        sotto = frontiera.pop()
        for oggetto, appoggi in grafo.items():
            if sotto in appoggi and oggetto not in sopra:
                sopra.add(oggetto)
                frontiera.append(oggetto)
    return bool(sopra)


def episodio(env, politica, seed: int, rng) -> dict:
    _, info = env.reset(seed=seed)
    sepolto = target_sepolto(env)
    scegli = POLITICHE[politica]
    ritorno, passi, peggiore = 0.0, 0, 0.0
    finito = False

    while not finito:
        azione = scegli(env, info, rng)
        _, ricompensa, terminato, troncato, info = env.step(azione)
        ritorno += ricompensa
        passi += 1
        peggiore = max(peggiore, info["disturbance_step"])
        finito = terminato or troncato

    return {
        "ritorno": ritorno,
        "passi": passi,
        "disturbo_totale": info["disturbance_total"],
        "mossa_peggiore": peggiore,
        "sepolto": sepolto,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oggetti", type=int, default=6)
    parser.add_argument("--scene", type=int, default=10)
    parser.add_argument(
        "--finisci-al-target",
        action="store_true",
        help="l'episodio finisce quando il target esce, invece di svuotare la scena",
    )
    argomenti = parser.parse_args()

    env = gym.make(
        "TargetExtraction-v0",
        disable_env_checker=True,
        obs_mode="state",
        object_count=argomenti.oggetti,
        terminate_on_target=argomenti.finisci_al_target or None,
    )
    compito = env.unwrapped._task
    # Col traguardo al target il tetto e' prendere il target subito: una sola
    # rimozione. Svuotando la scena invece si pagano tutte le rimozioni,
    # qualunque cosa faccia la policy.
    rimozioni_minime = 1 if argomenti.finisci_al_target else argomenti.oggetti
    massimo = float(compito["target_reward"]) - float(
        compito["removal_cost"]
    ) * rimozioni_minime

    print(
        f"\n{argomenti.oggetti} oggetti, {argomenti.scene} scene, "
        f"soglia di disturbo {compito['disturbance_threshold']} m"
    )
    print(
        f"tetto teorico: {massimo:+.3f}  "
        f"(premio {compito['target_reward']:+.2f} "
        f"meno {rimozioni_minime} rimozioni da {compito['removal_cost']})\n"
    )
    print(
        f"  {'politica':10s} {'ritorno':>16s} {'margine dal tetto':>18s} "
        f"{'disturbo totale':>16s} {'passi':>8s}"
    )
    print("  " + "-" * 78)

    risultati, sepolte = {}, {}
    try:
        for politica in list(POLITICHE) + ["ottimo"]:
            if politica == "ottimo":
                righe = [ottimo(env, seed) for seed in range(argomenti.scene)]
            else:
                righe = [
                    episodio(env, politica, seed, np.random.default_rng(seed))
                    for seed in range(argomenti.scene)
                ]
            ritorni = np.array([r["ritorno"] for r in righe])
            disturbi = np.array([r["disturbo_totale"] for r in righe])
            risultati[politica] = ritorni
            sepolte[politica] = np.array([r["sepolto"] for r in righe])
            print(
                f"  {politica:10s} {ritorni.mean():+8.3f} ± {ritorni.std():5.3f} "
                f"{massimo - ritorni.mean():18.3f} "
                f"{disturbi.mean() * 1000:13.1f} mm "
                f"{np.mean([r['passi'] for r in righe]):8.2f}"
            )
    finally:
        env.close()

    guadagno = risultati["ottimo"].mean() - risultati["alto"].mean()
    rumore = risultati["alto"].std() / np.sqrt(argomenti.scene)

    # Lo stesso confronto sulle sole scene che pongono il problema. E' li' che
    # si vede se c'e' qualcosa da imparare: dove il target e' scoperto,
    # qualunque politica lo prende al primo colpo e i ritorni coincidono per
    # forza, abbassando la media di tutti allo stesso modo e nascondendo il
    # margine.
    maschera = sepolte["ottimo"]
    quante = int(maschera.sum())
    print()
    if quante:
        print(
            f"  Solo sulle {quante}/{argomenti.scene} scene col target "
            f"SEPOLTO:"
        )
        for politica in risultati:
            valori = risultati[politica][maschera]
            print(f"    {politica:10s} {valori.mean():+8.3f} ± {valori.std():5.3f}")
        margine_sepolte = (
            risultati["ottimo"][maschera].mean()
            - risultati["alto"][maschera].mean()
        )
        print(f"    margine su 'alto': **{margine_sepolte:+.3f}**")
    print()
    print(
        f"  Margine che una policy potrebbe ancora guadagnare su 'alto': "
        f"**{guadagno:+.3f}**"
    )
    print(
        f"  Incertezza su 'alto' con {argomenti.scene} scene: ± {rumore:.3f}  "
        f"({'MISURABILE' if abs(guadagno) > 2 * rumore else 'DENTRO IL RUMORE'})"
    )


if __name__ == "__main__":
    main()
