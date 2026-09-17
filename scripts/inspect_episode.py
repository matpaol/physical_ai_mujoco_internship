"""Ispeziona un episodio di `TargetExtraction-v0`: catalogo, appoggi, azioni.

`run_phase_0b` **esegue** gli episodi. Questo modulo li **spiega**: stampa in
chiaro tre cose che altrimenti restano dentro la simulazione.

1. il **catalogo** degli oggetti dopo l'assestamento — chi sono, quanto pesano,
   che attrito hanno, dove stanno, quale e' il target;
2. il **grafo degli appoggi** — chi poggia su chi, letto dai contatti veri di
   MuJoCo. E' questo a decidere quale ordine di rimozione e' sicuro, ed e'
   esattamente l'informazione che l'agente dovra' dedurre da solo;
3. le **azioni** scelte passo per passo, con l'effetto di ognuna sul disturbo.

Non allena niente e non modifica l'environment: legge e stampa.

Dal menu: voce 6 di `main.py`. Da riga di comando:

    python -m scripts.inspect_episode                          # scegli tu le azioni
    python -m scripts.inspect_episode --objects 5 --policy top
    python -m scripts.inspect_episode --policy random --episodes 20 --quiet
"""

from __future__ import annotations

import argparse
import select
import sys

import gymnasium as gym
import numpy as np

import physical_ai_mujoco.envs  # noqa: F401  (registra TargetExtraction-v0)
from physical_ai_mujoco.scene.scene_description import half_extents

LARGHEZZA = 96

POLITICHE = ("manual", "random", "top", "target")

DESCRIZIONE_POLITICHE = {
    "manual": "scegli tu l'indice a ogni passo",
    "random": "a caso fra le azioni valide",
    "top": "sempre l'oggetto piu' in alto — l'euristica ovvia",
    "target": "subito il target, senza sgomberare — massima fretta",
}


# --------------------------------------------------------------- lettura scena


def parametri_task(interno) -> dict:
    """Parametri del task (soglie, pesi), comunque l'env li esponga."""
    return interno.task


def grafo_appoggi(simulatore) -> dict[str, list[str]]:
    """`{oggetto: [cosa lo sostiene]}`, letto dai contatti veri di MuJoCo."""
    return simulatore.support_graph()


def cosa_grava_sul_target(interno) -> list[str]:
    """Oggetti che gravano sul target, direttamente o attraverso altri.

    E' la misura di quanto il target e' sepolto, e quindi di quanto l'episodio
    pone davvero il problema dell'ordine di rimozione.

    Confrontare le quote non basta, ed e' un errore che nasconde i guai: un
    oggetto piu' in alto ma appoggiato altrove non blocca niente, e uno ancora
    in volo risulterebbe "sopra" senza toccare nulla. Qui si risale il grafo
    degli appoggi, che e' la relazione vera.
    """
    sostegni = interno.simulator.support_graph()
    sopra: set[str] = set()
    frontiera = [interno.target_id]

    while frontiera:
        corrente = frontiera.pop()
        for oggetto, sotto in sostegni.items():
            if corrente in sotto and oggetto not in sopra:
                sopra.add(oggetto)
                frontiera.append(oggetto)

    return sorted(sopra)


def _centro_di_massa(item) -> str:
    """Scostamento del centro di massa, in percentuale della semi-dimensione.

    Percentuale e non millimetri: e' l'eccentricita' a decidere se un oggetto
    si ribalta, e quella e' relativa alla sua dimensione.
    """
    if not item.has_offset_mass:
        return "centrato"
    estensioni = half_extents(item.shape, item.size)
    return " ".join(
        f"{offset / extent * 100:+.0f}%"
        for offset, extent in zip(item.center_of_mass, estensioni, strict=True)
    )


def dimensione_leggibile(forma: str, size: dict) -> str:
    if forma == "box":
        return f"{size['x']*100:.1f}×{size['y']*100:.1f}×{size['z']*100:.1f} cm"
    if forma == "cylinder":
        return f"r {size['radius']*100:.1f} × h {size['height']*100:.1f} cm"
    if forma == "sphere":
        return f"r {size['radius']*100:.1f} cm"
    return str(size)


# ------------------------------------------------------------------- stampa


def stampa_catalogo(interno) -> None:
    """Tabella degli oggetti, nell'ordine degli indici delle azioni."""
    simulatore = interno.simulator

    print()
    print("CATALOGO OGGETTI  (l'indice e' l'azione da passare a step)")
    print("-" * LARGHEZZA)
    print(
        f"{'idx':>3}  {'id':<12} {'tipo':<9} {'massa':>7} {'attr.':>6} "
        f"{'dimensioni':<22} {'x':>7} {'y':>7} {'z':>7}  {'c.massa':>12}"
    )
    print("-" * LARGHEZZA)

    for indice, item in enumerate(simulatore.scene.objects):
        stato = simulatore.get_object_state(item.instance_id)
        marchio = "  ← TARGET" if item.instance_id == interno.target_id else ""
        print(
            f"{indice:>3}  {item.instance_id:<12} {item.type_id:<9} "
            f"{item.mass*1000:>6.0f}g {item.friction[0]:>6.2f} "
            f"{dimensione_leggibile(item.shape, item.size):<22} "
            f"{stato.position[0]:>7.3f} {stato.position[1]:>7.3f} "
            f"{stato.position[2]:>7.3f}  {_centro_di_massa(item):>12}{marchio}"
        )
    print("-" * LARGHEZZA)


def stampa_appoggi(interno) -> None:
    """Chi poggia su chi, e quanto e' sepolto il target."""
    simulatore = interno.simulator
    sostegni = grafo_appoggi(simulatore)

    print()
    print("GRAFO DEGLI APPOGGI  (dopo l'assestamento)")
    print("-" * LARGHEZZA)
    for item in simulatore.scene.objects:
        sotto = sostegni.get(item.instance_id, [])
        etichetta = ", ".join(sotto) if sotto else "niente (sospeso o isolato)"
        marchio = "  ← TARGET" if item.instance_id == interno.target_id else ""
        print(f"  {item.instance_id:<12} poggia su: {etichetta}{marchio}")

    sopra = cosa_grava_sul_target(interno)
    print("-" * LARGHEZZA)
    print(f"  Gravano sul target: {len(sopra)}  {sopra if sopra else ''}")
    if not sopra:
        print(
            "  ATTENZIONE: sul target non poggia niente, quindi l'ordine non conta.\n"
            "  L'episodio non pone il problema che vogliamo studiare."
        )
    print("-" * LARGHEZZA)


def stampa_costruzione(info) -> None:
    """Come e' stata costruita la scena, una caduta per riga.

    Serve a controllare la FISICA, non l'agente. Le colonne da guardare:

    - **caduta**: di quanto e' sceso il baricentro. Se e' molto piu' del
      gioco di rilascio, l'oggetto ha mancato la pila ed e' finito a terra.
    - **assest.**: secondi simulati fino alla quiete. Valori molto lunghi, o
      un `NO` nella colonna della quiete, dicono che qualcosa continua a
      muoversi — e allora le pose lette dopo non sono una configurazione
      stabile.
    - **smosso**: quanto questa caduta ha spostato cio' che c'era gia'. E' la
      misura di quanto la pila regge: se ogni aggiunta rimescola tutto, la
      scena non e' un mucchio, e' un lancio di dadi.
    """
    righe = info.get("release_log") or []
    if not righe:
        return

    print()
    print("COSTRUZIONE DELLA SCENA  (rilascio sequenziale, un oggetto alla volta)")
    print("-" * LARGHEZZA)
    print(
        f"{'#':>2}  {'oggetto':<12} {'tipo':<8} {'massa':>7} {'rilascio':>9} "
        f"{'caduta':>8} {'assest.':>8} {'reale':>7} {'quiete':>7} {'smosso':>8}  poggia su"
    )
    print("-" * LARGHEZZA)

    for riga in righe:
        marchio = " ←T" if riga["is_target"] else "   "
        poggia = ", ".join(riga["resting_on"]) or "niente"
        print(
            f"{riga['index']:>2}  {riga['instance_id']:<12}{marchio}"
            f" {riga['type_id']:<8} {riga['mass']*1000:>6.0f}g "
            f"{riga['drop_height']:>9.3f} {riga['fall_distance']:>8.3f} "
            f"{riga['settling_seconds']:>7.2f}s {riga['wall_seconds']*1000:>6.0f}ms "
            f"{'si' if riga['settled'] else 'NO':>7} "
            f"{riga['moved_others']:>8.4f}  {poggia}"
        )

    print("-" * LARGHEZZA)
    totale_sim = sum(r["settling_seconds"] for r in righe)
    totale_reale = sum(r["wall_seconds"] for r in righe)
    non_fermi = [r["instance_id"] for r in righe if not r["settled"]]
    print(
        f"  totale: {totale_sim:.2f} s simulati in {totale_reale*1000:.0f} ms reali"
        f"   ({totale_sim/max(totale_reale, 1e-9):.2f}× tempo reale)"
    )
    if non_fermi:
        print(f"  NON ASSESTATI entro il timeout: {non_fermi}")
    print("-" * LARGHEZZA)


def stampa_intestazione_azioni() -> None:
    print()
    print("AZIONI")
    print("-" * LARGHEZZA)
    print(
        f"{'#':>2}  {'azione':>6}  {'oggetto':<12} {'reward':>8} "
        f"{'mossa':>9} {'cumul.':>9}  esito"
    )
    print("-" * LARGHEZZA)


# ----------------------------------------------------------------- politiche


# Ogni quanto ridisegnare mentre si aspetta che l'utente batta un tasto. 30 ms
# sono circa 33 fotogrammi al secondo: piu' che sufficienti per una scena ferma,
# e abbastanza frequenti da non far mai scadere il controllo del sistema.
PAUSA_DISEGNO = 0.03


def _chiedi(prompt: str, interno) -> str:
    """Legge una riga da tastiera **continuando a ridisegnare il viewer**.

    `input()` blocca il processo finche' non si preme Invio. Con il viewer
    aperto quel blocco ferma anche il ciclo degli eventi della finestra, e
    macOS dopo qualche secondo la dichiara «non risponde»: la finestra si
    congela e l'unico modo di uscirne e' l'uscita forzata. Qui si aspetta a
    piccoli intervalli, e in ognuno si ridisegna.

    Senza viewer, o quando l'ingresso non e' un terminale (script, pipe,
    test), si legge e basta: non c'e' niente da ridisegnare.
    """
    print(prompt, end="", flush=True)

    disegnare = getattr(interno, "render_mode", None) == "human"
    try:
        interattivo = disegnare and sys.stdin.isatty()
    except ValueError:  # stdin chiuso
        interattivo = False

    if not interattivo:
        riga = sys.stdin.readline()
        if not riga:
            raise EOFError("ingresso terminato")
        return riga.strip()

    while True:
        try:
            pronti, _, _ = select.select([sys.stdin], [], [], PAUSA_DISEGNO)
        except OSError:
            # `select` su stdin non e' disponibile ovunque (Windows): meglio
            # una finestra che si congela che un comando che non parte.
            riga = sys.stdin.readline()
            if not riga:
                raise EOFError("ingresso terminato") from None
            return riga.strip()

        if pronti:
            riga = sys.stdin.readline()
            if not riga:
                raise EOFError("ingresso terminato")
            return riga.strip()

        interno.render()


def scegli_azione(politica: str, interno, info, rng) -> int:
    """Indice dell'oggetto da rimuovere, secondo la politica scelta."""
    maschera = np.asarray(info["action_mask"], dtype=bool)
    validi = np.flatnonzero(maschera)

    if politica == "random":
        return int(rng.choice(validi))

    if politica == "top":
        # Il piu' in alto fra quelli ancora presenti. E' il riferimento: se una
        # policy appresa non batte questo, non ha imparato niente.
        quote = [
            interno.simulator.get_object_state(
                interno.simulator.scene.objects[indice].instance_id
            ).position[2]
            for indice in validi
        ]
        return int(validi[int(np.argmax(quote))])

    if politica == "target":
        indice = info["target_index"]
        return int(indice) if indice in validi else int(rng.choice(validi))

    if politica == "manual":
        indici = [int(valore) for valore in validi]
        while True:
            risposta = _chiedi(f"  azione fra {indici} > ", interno)
            if risposta.isdigit() and int(risposta) in indici:
                return int(risposta)
            print("  indice non valido")

    raise ValueError(f"Politica sconosciuta: {politica}")


# ------------------------------------------------------------------- episodio


def esegui_episodio(env, politica: str, seed, rng, verboso: bool) -> dict:
    """Un episodio completo. Restituisce le metriche, stampa se `verboso`."""
    _, info = env.reset(seed=seed)
    interno = env.unwrapped

    if verboso:
        print()
        print("=" * LARGHEZZA)
        print(
            f"SEED SCENA {info['scene_seed']}   "
            f"oggetti {interno.object_count}   "
            f"politica '{politica}'   "
            f"soglia disturbo {float(parametri_task(interno)['disturbance_threshold']):.3f} m"
        )
        print("=" * LARGHEZZA)
        stampa_costruzione(info)
        stampa_catalogo(interno)
        stampa_appoggi(interno)
        stampa_intestazione_azioni()

    sepolto = len(cosa_grava_sul_target(interno)) > 0
    ordine: list[str] = []
    disturbo_peggiore = 0.0
    ricompensa_totale = 0.0
    passi = 0
    terminato = truncato = False

    while not (terminato or truncato):
        azione = scegli_azione(politica, interno, info, rng)
        nome = interno.simulator.scene.objects[azione].instance_id

        _, ricompensa, terminato, truncato, info = env.step(azione)
        passi += 1
        ricompensa_totale += ricompensa

        disturbo = float(info["disturbance_step"])
        disturbo_peggiore = max(disturbo_peggiore, disturbo)

        esiti = []
        if info.get("invalid_action"):
            esiti.append("AZIONE NON VALIDA (gia' rimosso)")
        else:
            ordine.append(nome)
        if info.get("target_just_removed"):
            esiti.append("TARGET RIMOSSO")
        if info.get("collapsed"):
            esiti.append("oltre soglia")
        if not info.get("settled", True):
            esiti.append("NON ASSESTATO: timeout")
        if truncato:
            esiti.append("troncato: passi esauriti")

        if verboso:
            print(
                f"{passi:>2}  {azione:>6}  {nome:<12} {ricompensa:>8.3f} "
                f"{disturbo:>9.4f} {info['disturbance_total']:>9.4f}  "
                f"{' · '.join(esiti)}"
            )

    successo = bool(info.get("is_success", False))
    if verboso:
        print("-" * LARGHEZZA)
        print(f"  ordine di rimozione: {_formatta_ordine(ordine, interno)}")
        print(
            f"  passi {passi}   ricompensa {ricompensa_totale:.3f}   "
            f"disturbo totale {info['disturbance_total']:.4f} m   "
            f"mossa peggiore {disturbo_peggiore:.4f} m"
        )
        print("=" * LARGHEZZA)

    return {
        "successo": successo,
        "passi": passi,
        "ricompensa": ricompensa_totale,
        "disturbo_totale": float(info["disturbance_total"]),
        "disturbo_peggiore": disturbo_peggiore,
        "target_sepolto": sepolto,
    }


def _formatta_ordine(ordine: list[str], interno) -> str:
    if not ordine:
        return "(nessuna rimozione)"
    return " → ".join(
        f"{nome} (TARGET)" if nome == interno.target_id else nome for nome in ordine
    )


def stampa_riepilogo(risultati: list[dict], politica: str, object_count: int) -> None:
    successi = sum(1 for r in risultati if r["successo"])
    sepolti = sum(1 for r in risultati if r["target_sepolto"])
    totale = len(risultati)

    print()
    print("=" * LARGHEZZA)
    print(
        f"RIEPILOGO   politica '{politica}'   "
        f"{object_count} oggetti   {totale} episodi"
    )
    print("-" * LARGHEZZA)
    print(f"  successi          {successi}/{totale}  ({100.0*successi/totale:.0f}%)")
    print(
        f"  target sepolto    {sepolti}/{totale}  ({100.0*sepolti/totale:.0f}%)"
        "   ← quante scene pongono davvero il problema"
    )
    print(f"  passi medi        {np.mean([r['passi'] for r in risultati]):.2f}")
    print(
        f"  ricompensa        {np.mean([r['ricompensa'] for r in risultati]):.3f} "
        f"± {np.std([r['ricompensa'] for r in risultati]):.3f}"
    )
    disturbi = [r["disturbo_peggiore"] for r in risultati]
    print(
        f"  mossa peggiore    media {np.mean(disturbi):.4f} m   "
        f"mediana {np.median(disturbi):.4f} m   max {np.max(disturbi):.4f} m"
    )
    print(
        f"  disturbo totale   {np.mean([r['disturbo_totale'] for r in risultati]):.4f} m"
    )
    print("=" * LARGHEZZA)


# -------------------------------------------------- punto di ingresso pubblico


def ispeziona(
    object_count: int = 3,
    policy: str = "manual",
    seed: int | None = None,
    episodes: int = 1,
    render: bool = False,
    highlight_target: bool = False,
    quiet: bool = False,
) -> list[dict]:
    """Esegue e racconta N episodi. E' la funzione che chiama anche `main.py`."""
    if policy not in POLITICHE:
        raise ValueError(f"Politica sconosciuta: {policy!r}. Scegli fra {POLITICHE}.")
    if policy == "manual" and episodes > 1:
        raise ValueError("La politica 'manual' funziona con un episodio alla volta.")

    env = gym.make(
        "TargetExtraction-v0",
        obs_mode="state",
        object_count=object_count,
        render_mode="human" if render else None,
        # Con il viewer aperto il modello non va ricompilato, altrimenti la
        # finestra si chiuderebbe a ogni episodio.
        resample_shapes=not render,
        highlight_target=highlight_target or render,
    )
    rng = np.random.default_rng(seed)

    risultati = []
    try:
        for episodio in range(episodes):
            seme = None if seed is None else seed + episodio
            risultati.append(
                esegui_episodio(env, policy, seme, rng, verboso=not quiet)
            )
    except (EOFError, KeyboardInterrupt):
        # Ctrl-C o Ctrl-D durante la scelta manuale: uscire e basta. Una
        # traccia di errore farebbe pensare a un guasto quando invece
        # l'utente ha semplicemente smesso.
        print("\n  interrotto")
    finally:
        env.close()

    if episodes > 1 or quiet:
        stampa_riepilogo(risultati, policy, object_count)
    return risultati


# ----------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Catalogo, grafo degli appoggi e azioni di un episodio."
    )
    parser.add_argument("--objects", type=int, default=3)
    parser.add_argument(
        "--policy",
        choices=POLITICHE,
        default="manual",
        help=" · ".join(f"{k}: {v}" for k, v in DESCRIZIONE_POLITICHE.items()),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--render", action="store_true", help="apre il viewer")
    parser.add_argument(
        "--highlight-target",
        action="store_true",
        help="colora il target di giallo (cambia cio' che vedono le camere)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="niente dettagli per episodio: solo il riepilogo finale",
    )
    argomenti = parser.parse_args()

    ispeziona(
        object_count=argomenti.objects,
        policy=argomenti.policy,
        seed=argomenti.seed,
        episodes=argomenti.episodes,
        render=argomenti.render,
        highlight_target=argomenti.highlight_target,
        quiet=argomenti.quiet,
    )


if __name__ == "__main__":
    main()
