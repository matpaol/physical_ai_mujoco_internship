"""Le politiche che scelgono quale oggetto togliere, tutte con la stessa firma.

    scegli(env, info, rng, osservazione) -> indice dell'azione

Averla unica permette di passare la stessa cosa al viewer, al confronto e alla
registrazione, invece di riscrivere ogni volta la stessa scelta in tre posti.

Le prime due sono i **riferimenti**: senza qualcuno con cui confrontarsi, il
ritorno di una policy appresa e' un numero senza scala. `alto` in particolare
e' la riga di euristica che una policy deve battere per avere un senso.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CARTELLA_MODELLI = PROJECT_ROOT / "outputs" / "modelli"


def casuale(env, info, rng, osservazione=None) -> int:
    """A caso fra gli oggetti ancora in scena. Il pavimento del confronto."""
    return int(rng.choice(np.flatnonzero(info["action_mask"])))


def alto(env, info, rng, osservazione=None) -> int:
    """Sempre l'oggetto piu' in alto: l'euristica ovvia, e il metro da battere."""
    interno = env.unwrapped
    stati = interno.simulator.get_present_object_states()
    validi = np.flatnonzero(info["action_mask"])
    quote = [stati[interno._object_ids[a]].position[2] for a in validi]
    return int(validi[int(np.argmax(quote))])


def modelli_disponibili() -> list[Path]:
    """I modelli allenati che si trovano in outputs/modelli/, dal piu' recente."""
    if not CARTELLA_MODELLI.is_dir():
        return []
    return sorted(
        CARTELLA_MODELLI.glob("*.zip"),
        key=lambda percorso: percorso.stat().st_mtime,
        reverse=True,
    )


def carica(percorso: str | Path):
    """Una policy che usa un modello allenato. Ritorna una funzione `scegli`.

    Carica anche le statistiche di normalizzazione salvate accanto al modello.
    **Non sono facoltative**: il modello e' stato allenato su osservazioni
    normalizzate, e dargli i numeri grezzi non solleva nessun errore — la forma
    del vettore e' la stessa, la scala no — ma fa scegliere a caso. Misurato:
    -0,587 invece di +0,849 sullo stesso identico modello.
    """
    from stable_baselines3 import PPO

    percorso = Path(percorso)
    modello = PPO.load(percorso)

    statistiche = percorso.with_name(f"{percorso.stem}_normalizzazione.pkl")
    if not statistiche.is_file():
        raise FileNotFoundError(
            f"Manca {statistiche.name}, il file con le statistiche di "
            "normalizzazione. Senza, il modello riceverebbe osservazioni su "
            "una scala diversa da quella su cui e' stato allenato e "
            "sceglierebbe a caso."
        )

    import pickle

    with statistiche.open("rb") as file:
        normalizzatore = pickle.load(file)

    def scegli(env, info, rng, osservazione) -> int:
        azione, _ = modello.predict(
            normalizzatore.normalize_obs(osservazione), deterministic=True
        )
        return int(azione)

    return scegli


RIFERIMENTI = {"casuale": casuale, "alto": alto}
