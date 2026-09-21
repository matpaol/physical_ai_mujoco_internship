"""Punto di ingresso unico del progetto.

Apri questo file in VS Code e premi Run (o esegui `python main.py`): un menu
chiede cosa fare, e ogni domanda ha una risposta predefinita che si accetta
premendo Invio.

Gli script sotto `scripts/` sono gli equivalenti non interattivi, per l'uso
dentro altri script o in CI.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


from physical_ai_mujoco.infrastructure.experiment import (
    ExperimentProfile,
    available_profiles,
    selected_profile,
)

ACTIVE_EXPERIMENT = selected_profile() or ExperimentProfile.load(
    PROJECT_ROOT / "configs/experiments/0b.json"
)


def choose_experiment():
    global ACTIVE_EXPERIMENT
    profiles = available_profiles()
    for i, profile in enumerate(profiles, 1):
        status = "disponibile" if profile.available else "da sviluppare"
        print(f"  {i}. {profile.name} [{status}]")
        print(f"     {profile.description}")
    choice = ask_int("Configurazione", default=2, low=1, high=len(profiles))
    profile = profiles[choice - 1]
    if not profile.available:
        print(f"Questa configurazione richiede ancora sviluppo: {profile.description}")
        return
    profile.activate()
    ACTIVE_EXPERIMENT = profile
    print(f"Configurazione selezionata: {profile.name}")


MENU = [
    (
        "1",
        "Esegui degli episodi",
        "L’agente sceglie e rimuove gli oggetti, un episodio dopo l'altro.\n"
        "     Come guardarli — finestra, GIF o solo testo — te lo chiedo\n"
        "     dopo: e' la stessa cosa vista in modi diversi.",
        lambda: run_experiment(),
    ),
    (
        "2",
        "Ispeziona una scena",
        "Cosa c'e' in scena, oggetto per oggetto, CHI POGGIA SU CHI letto\n"
        "     dai contatti veri di MuJoCo, se il target e' sepolto, e che\n"
        "     effetto ha ogni rimozione. Puoi scegliere tu le mosse.",
        lambda: inspect_episode(),
    ),
    (
        "3",
        "Allena la policy che sceglie l'ordine",
        "PPO impara a togliere gli oggetti fino a estrarre il target\n"
        "     smuovendo il mucchio il meno possibile. Alla fine confronta il\n"
        "     modello con le politiche di riferimento sulle stesse scene di valutazione:\n"
        "     e' l'unico modo di sapere se ha imparato qualcosa.",
        lambda: train_policy(),
    ),
    (
        "4",
        "Raccogli i dati stereo per lo student",
        "Esegue episodi salvando a ogni passo la coppia di immagini delle\n"
        "     due camere, in outputs/stereo/. Sono i dati su cui lo student\n"
        "     imparera' a imitare il teacher.",
        lambda: collect_stereo(),
    ),
    (
        "5",
        "Verifica che tutto funzioni",
        "Tre controlli: i test del codice; le prove di fisica con risposta\n"
        "     calcolabile a mano (caduta libera, scivolamento, ribaltamento,\n"
        "     inerzia); e quanto margine c'e' ancora fra l'euristica e\n"
        "     l'ordine migliore possibile, che dice se il compito ha ancora\n"
        "     qualcosa da insegnare.",
        lambda: verify(),
    ),
    (
        "6",
        "Avanzate",
        "Fase 0A: genera una scena da un seed e la misura, senza nessun\n"
        "     agente. E' congelata, serve a controllare la generazione.",
        lambda: inspect_scene(),
    ),
    ("7", "Scegli fase / configurazione esperimento", None, choose_experiment),
    ("0", "Esci", None, None),
]


def run_experiment() -> None:
    """Voce 1: esegui episodi. Il "come guardarli" e' la prima domanda.

    Le quattro modalita' fanno la STESSA cosa — episodi con una politica —
    e differiscono solo nel rendering. Tenerle come quattro voci di menu
    faceva sembrare che fossero quattro esperimenti diversi.
    """
    print("  Come vuoi guardarli?")
    print("    1  finestra interattiva   ruoti col mouse, zoomi con lo scroll")
    print("    2  finestra con pause     si ferma e commenta; piu' scene affiancate")
    print("    3  registra un filmato    nessuna finestra, esce una GIF o un mp4")
    print("    4  senza finestra         solo testo: veloce, per misurare")
    print()
    modo = ask_int("Modalita'", default=1, low=1, high=4)
    print()

    # Le modalita' 2 e 3 girano in un altro processo con il loro ciclo, e
    # usano sempre la policy casuale. Chiedere li' chi decide sarebbe una
    # domanda la cui risposta viene buttata via.
    scegli = chiedi_politica() if modo in (1, 4) else None

    if modo == 1:
        watch_gymnasium(scegli)
    elif modo == 2:
        watch_agent()
    elif modo == 3:
        record_clip()
    else:
        run_episodes(scegli)


def chiedi_politica():
    """Chi decide le mosse: un riferimento, o un modello allenato.

    Restituisce `None` per la policy casuale, che e' il comportamento
    predefinito del resto del codice.
    """
    from physical_ai_mujoco.infrastructure import policy_adapter as politiche

    modelli = politiche.modelli_disponibili()

    print("  Chi sceglie le mosse?")
    print("    1  a caso             il riferimento piu' basso")
    print("    2  il piu' in alto    l'euristica ovvia, il metro da battere")
    if modelli:
        print(f"    3  un modello allenato  ({len(modelli)} in outputs/modelli/)")
    print()

    preferred = {"random": 1, "highest": 2, "ppo": 3}.get(
        ACTIVE_EXPERIMENT.default_decider, 1
    )
    if preferred == 3 and not modelli:
        print(
            "Nessun modello presente: usa la voce 3 del menu per allenarlo. Qui puoi confrontare le baseline."
        )
        preferred = 2
    scelta = ask_int("Chi decide", default=preferred, low=1, high=3 if modelli else 2)
    if scelta == 1:
        return politiche.casuale
    if scelta == 2:
        return politiche.alto

    print("\n  Modelli disponibili, dal piu' recente:")
    for indice, percorso in enumerate(modelli, start=1):
        print(f"    {indice}  {percorso.stem}")
    print()
    quale = ask_int("Quale modello", default=1, low=1, high=len(modelli))
    percorso = modelli[quale - 1]

    try:
        scegli = politiche.carica(percorso)
    except Exception as errore:  # noqa: BLE001
        print(f"\n  Non riesco a usarlo: {errore}")
        print("  Uso l'euristica al suo posto.\n")
        return politiche.alto

    # Un modello e' allenato per un numero di oggetti preciso: l'osservazione
    # e lo spazio delle azioni hanno quella dimensione. Con un numero diverso
    # non c'e' un errore chiaro, solo scelte senza senso.
    numero = _oggetti_dal_nome(percorso.stem)
    if numero:
        print(f"\n  Nota: questo modello e' stato allenato su {numero} oggetti.")
        print("  Con un numero diverso le sue scelte non vogliono dire niente.\n")
    return scegli


def _oggetti_dal_nome(nome: str) -> int | None:
    """Estrae il numero di oggetti da un nome tipo `ppo_6oggetti_25000passi`."""
    for pezzo in nome.split("_"):
        if pezzo.endswith("oggetti") and pezzo[:-7].isdigit():
            return int(pezzo[:-7])
    return None


def train_policy() -> None:
    """Voce 3: allenamento PPO della policy di rimozione."""
    object_count = ask_int("Quanti oggetti nella scena?", default=6, low=2, high=12)
    steps = ask_int(
        "Quanti passi di allenamento? (25000 = ~35 min su 2 core)",
        default=25_000,
        low=1000,
        high=5_000_000,
    )
    parallel = ask_int(
        "Quanti ambienti in parallelo? (uno per core disponibile)",
        default=2,
        low=1,
        high=16,
    )
    pool = ask_int(
        "Quante scene tenere gia' costruite? (0 = ricostruiscile sempre)",
        default=200,
        low=0,
        high=5000,
    )

    print(
        "\nCostruire una scena costa 0,7 s, una mossa 0,05 s: senza il"
        "\nmagazzino l'84% del tempo se ne va a far cadere oggetti."
        "\nI primi episodi sono lenti comunque, e' il riempimento.\n"
    )
    command = [
        sys.executable,
        "scripts/allena.py",
        "--oggetti",
        str(object_count),
        "--passi",
        str(steps),
        "--paralleli",
        str(parallel),
        "--magazzino",
        str(pool),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=False)


def verify() -> None:
    """Voce 4: i test del codice e le prove di fisica, in quest'ordine."""
    print("--- test del codice ---\n")
    run_tests()

    if not ask_yes_no("\nEseguire anche le prove di fisica?", default=True):
        return

    scene = ask_int(
        "\nSu quante scene provare anche le pile? (0 = solo prove analitiche)",
        default=0,
        low=0,
        high=100,
    )
    comando = [sys.executable, "scripts/verifica_fisica.py"]
    if scene:
        comando += ["--scene", str(scene)]
    print()
    subprocess.run(comando, cwd=PROJECT_ROOT, check=False)

    if not ask_yes_no(
        "\nMisurare anche quanto margine resta fra l'euristica e l'ottimo?",
        default=False,
    ):
        return

    print(
        "\nCerca l'ordine migliore possibile provandoli tutti: e' lento."
        "\nCon 5 oggetti circa venti secondi a scena, con 6 circa un minuto.\n"
    )
    oggetti = ask_int("Quanti oggetti?", default=5, low=2, high=7)
    quante = ask_int("Su quante scene?", default=8, low=1, high=100)
    print()
    subprocess.run(
        [
            sys.executable,
            "scripts/quanto_margine.py",
            "--oggetti",
            str(oggetti),
            "--scene",
            str(quante),
            "--finisci-al-target",
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )


def main() -> int:
    """Avvio guidato: prima la fase, poi soltanto le sue scelte."""
    print_banner()
    while True:
        profile = choose_phase()
        if profile is None:
            return 0
        try:
            if profile.path.stem == "0a":
                run_phase_0a()
            elif profile.path.stem == "0b":
                run_phase_0b()
            elif profile.path.stem == "1a":
                run_phase_1a()
            elif profile.path.stem in {"1b", "1c"}:
                run_phase_1b()
        except KeyboardInterrupt:
            print("\nEsecuzione interrotta.")
        except Exception as error:  # noqa: BLE001
            print(f"\nErrore: {error}")


def choose_phase() -> ExperimentProfile | None:
    """Sceglie il grado di de-idealizzazione, senza numeri ambigui."""
    global ACTIVE_EXPERIMENT
    profiles = {profile.path.stem.lower(): profile for profile in available_profiles()}
    print("\nQuale fase vuoi eseguire?\n")
    for code, profile in profiles.items():
        status = "" if profile.available else "  [da sviluppare]"
        print(f"  {code.upper():>2}  {profile.name.split('·', 1)[-1].strip()}{status}")
        print(f"      {profile.description}")
    print("   X  Esci")

    while True:
        choice = ask("Fase", default="0B").strip().lower()
        if choice in {"x", "esci", "exit", "0"}:
            return None
        profile = profiles.get(choice)
        if profile is None:
            print("Scrivi il nome della fase, per esempio 0A, 0B, 1A oppure 1B.")
            continue
        if not profile.available:
            print(f"\n{profile.name} non e' ancora eseguibile: {profile.description}\n")
            continue
        profile.activate()
        ACTIVE_EXPERIMENT = profile
        print(f"\nFase selezionata: {profile.name}\n")
        return profile


def run_phase_0a() -> None:
    """Dataset -> descrizione della scena -> MuJoCo, senza Gymnasium."""
    print("Carico i dataset e genero una scena. Qui non esistono agente, azioni o reward.\n")
    seed = ask_int("Seed della scena", default=42, low=0, high=10**9)
    visual = ask_yes_no("Vuoi vedere la scena in MuJoCo?", default=True)
    slow = visual and ask_yes_no("Vuoi vederla al rallentatore?", default=False)
    inspect_scene(seed=seed, visual=visual, slow=slow)


def run_phase_0b() -> None:
    """MuJoCo dentro Gymnasium, decisioni casuali, fino al terreno vuoto."""
    from physical_ai_mujoco.infrastructure.policy_adapter import casuale

    print(
        "MuJoCo calcola la fisica; Gymnasium esegue reset(), step(), reward e fine episodio.\n"
        "La scelta e' casuale e l'episodio continua finche' tutti gli oggetti sono rimossi.\n"
    )
    object_count = ask_int("Quanti oggetti nella scena?", default=3, low=1, high=12)
    episodes = ask_int("Quanti episodi?", default=1, low=1, high=10000)
    visual, speed = ask_visualization()
    run_guided_episodes(casuale, object_count, episodes, visual, speed)


def run_phase_1a() -> None:
    """Teacher con stato esatto: allena PPO oppure esegue un modello salvato."""
    while True:
        print("  1  Allena una nuova policy PPO")
        print("  2  Esegui una policy gia' allenata")
        print("  0  Torna alla scelta della fase\n")
        choice = ask("Operazione", default="2")
        if choice == "0":
            return
        if choice == "1":
            train_policy()
            return
        if choice == "2":
            selected = choose_trained_policy(sensor=False)
            if selected is None:
                continue
            policy, trained_object_count = selected
            object_count = trained_object_count or ask_int(
                "Quanti oggetti usava il modello?", default=6, low=1, high=12
            )
            episodes = ask_int("Quanti episodi?", default=5, low=1, high=10000)
            visual, speed = ask_visualization()
            run_guided_episodes(policy, object_count, episodes, visual, speed)
            return
        print("Scegli 1, 2 oppure 0.")


def run_phase_1b() -> None:
    """Policy che riceve soltanto il vettore prodotto da stereo + LiDAR."""
    from physical_ai_mujoco.sensors import resolve_detector_weights

    try:
        detector_path = resolve_detector_weights()
    except FileNotFoundError as error:
        print(f"\nLa fase 1B non puo' partire: {error}\n")
        return
    print(
        "La policy riceve detection B/N, geometria LiDAR e relazioni codificate; "
        "non riceve lo stato esatto MuJoCo.\n"
        f"Detector: {detector_path.name}\n"
    )
    while True:
        print("  1  Allena una nuova policy PPO sensoriale")
        print("  2  Esegui una policy sensoriale gia' allenata")
        print("  0  Torna alla scelta della fase\n")
        choice = ask("Operazione", default="2")
        if choice == "0":
            return
        if choice == "1":
            train_policy()
            return
        if choice == "2":
            selected = choose_trained_policy(sensor=True)
            if selected is None:
                continue
            policy, trained_object_count = selected
            object_count = trained_object_count or ask_int(
                "Quanti oggetti usava il modello?", default=6, low=1, high=12
            )
            episodes = ask_int("Quanti episodi?", default=5, low=1, high=10000)
            visual, speed = ask_visualization()
            run_guided_episodes(policy, object_count, episodes, visual, speed)
            return
        print("Scegli 1, 2 oppure 0.")


def choose_trained_policy(sensor: bool | None = None):
    from physical_ai_mujoco.infrastructure import policy_adapter as policies

    models = policies.modelli_disponibili()
    if sensor is True:
        models = [path for path in models if path.stem.startswith("ppo_sensor_")]
    elif sensor is False:
        models = [path for path in models if not path.stem.startswith("ppo_sensor_")]
    if not models:
        tipo = " sensoriali" if sensor else ""
        print(
            f"\nNon ci sono modelli{tipo} compatibili in outputs/modelli/. "
            "Prima allena una policy.\n"
        )
        return None
    print("\nModelli disponibili:")
    for index, path in enumerate(models, 1):
        print(f"  {index}  {path.stem}")
    selected = models[ask_int("Quale modello", 1, 1, len(models)) - 1]
    return policies.carica(selected), _oggetti_dal_nome(selected.stem)


def ask_visualization() -> tuple[bool, float]:
    visual = ask_yes_no("Vuoi vedere la simulazione?", default=True)
    slow = visual and ask_yes_no("Vuoi vederla al rallentatore?", default=False)
    return visual, 0.5 if slow else (1.0 if visual else 0.0)


def run_guided_episodes(scegli, object_count, episodes, visual, speed) -> None:
    from physical_ai_mujoco.experiments.rollout import run_episode

    env = make_env(
        obs_mode="state",
        render_mode="human" if visual else None,
        object_count=object_count,
        highlight_target=visual,
        realtime_factor=speed,
        resample_shapes=not visual,
    )
    arguments = argparse.Namespace(
        seed=0, save_stereo=None, obs_mode="state", no_catalogue=episodes > 10
    )
    successes = 0
    try:
        for episode in range(episodes):
            successes += run_episode(env, episode, arguments, None, scegli)
    finally:
        env.close()
    print(f"\nEpisodi completati: {episodes} | target estratti: {successes}")


# ----------------------------------------------------------------- le azioni


def watch_gymnasium(scegli=None) -> None:
    """Il viewer interattivo di Gymnasium (render_mode="human").

    Usa forme fisse fra un episodio e l'altro: il viewer e' legato a un
    modello compilato, e ricompilarlo lo chiuderebbe. E' una rinuncia alla
    randomizzazione delle forme, accettabile per guardare, da NON usare
    quando si allena.
    """
    from physical_ai_mujoco.experiments.rollout import run_episode

    object_count = ask_int("Quanti oggetti nella scena?", default=3, low=1, high=12)
    episodes = ask_int("Quanti episodi?", default=5, low=1, high=1000)
    slow = ask_yes_no("Rallentare a meta' velocita'?", default=False)

    env = make_env(
        obs_mode="state",
        render_mode="human",
        object_count=object_count,
        highlight_target=True,
        realtime_factor=0.5 if slow else 1.0,
        # Forme fisse: e' cio' che tiene la finestra aperta fra gli episodi.
        resample_shapes=False,
    )

    print(
        "\nIl target da estrarre e' l'oggetto GIALLO."
        "\nTrascina col mouse per ruotare, scroll per lo zoom.\n"
    )
    arguments = argparse.Namespace(seed=0, save_stereo=None, obs_mode="state")
    successes = 0
    try:
        for episode in range(episodes):
            successes += run_episode(env, episode, arguments, None, scegli)
    finally:
        env.close()
    print(f"\nSuccessi: {successes}/{episodes}")


def watch_agent() -> None:
    """Finestra persistente con rendering offscreen (scripts/watch_phase_0b)."""
    object_count = ask_int("Quanti oggetti nella scena?", default=3, low=1, high=12)
    parallel = ask_int("Quanti episodi mostrare affiancati?", default=1, low=1, high=9)
    episodes = ask_int(
        "Quanti episodi? (0 = finche' non chiudi la finestra)",
        default=0,
        low=0,
        high=1000,
    )
    slow = ask_yes_no("Rallentare a meta' velocita'?", default=False)

    command = [
        sys.executable,
        "-m",
        "scripts.watch_phase_0b",
        "--objects",
        str(object_count),
        "--parallel",
        str(parallel),
        "--episodes",
        str(episodes),
        "--speed",
        "0.5" if slow else "1.0",
    ]
    print("\nL'oggetto GIALLO e' il target. Premi 'q' nella finestra per chiuderla.\n")
    subprocess.run(command, cwd=PROJECT_ROOT, check=False)


def record_clip() -> None:
    """Export offscreen: nessuna finestra, nessun overlay."""
    object_count = ask_int("Quanti oggetti nella scena?", default=3, low=1, high=12)
    episodes = ask_int("Quanti episodi?", default=1, low=1, high=20)
    camera = ask(
        "Quale camera? (cam_overview = panoramica, cam_left = vista student)",
        default="cam_overview",
    )
    fmt = ask("Formato? (gif / mp4)", default="gif")

    command = [
        sys.executable,
        "-m",
        "scripts.record_phase_0b",
        "--objects",
        str(object_count),
        "--episodes",
        str(episodes),
        "--camera",
        camera,
        "--format",
        fmt if fmt in {"gif", "mp4"} else "gif",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=False)


def run_episodes(scegli=None) -> None:
    from physical_ai_mujoco.experiments.rollout import run_episode

    object_count = ask_int("Quanti oggetti nella scena?", default=3, low=1, high=12)
    episodes = ask_int("Quanti episodi?", default=5, low=1, high=10000)

    # Con pochi episodi il catalogo e il grafo degli appoggi si leggono; con
    # molti diventano rumore, quindi si chiede invece di imporre.
    catalogue = (
        ask_yes_no("Stampare catalogo e appoggi di ogni scena?", default=True)
        if episodes > 10
        else True
    )

    env = make_env(obs_mode="state", object_count=object_count)
    arguments = argparse.Namespace(
        seed=0, save_stereo=None, obs_mode="state", no_catalogue=not catalogue
    )
    successes = 0
    try:
        for episode in range(episodes):
            successes += run_episode(env, episode, arguments, None, scegli)
    finally:
        env.close()
    print(f"\nSuccessi: {successes}/{episodes}")


def collect_stereo() -> None:
    from physical_ai_mujoco.experiments.rollout import run_episode

    object_count = ask_int("Quanti oggetti nella scena?", default=3, low=1, high=12)
    episodes = ask_int("Quanti episodi?", default=5, low=1, high=10000)
    destination = PROJECT_ROOT / "outputs" / "stereo"
    destination.mkdir(parents=True, exist_ok=True)

    env = make_env(obs_mode="both", object_count=object_count)
    arguments = argparse.Namespace(
        seed=0,
        save_stereo=str(destination),
        obs_mode="both",
        no_catalogue=episodes > 10,
    )
    try:
        for episode in range(episodes):
            run_episode(env, episode, arguments, destination)
    finally:
        env.close()
    print(f"\nCoppie stereo salvate in: {destination}")


def inspect_episode() -> None:
    """Catalogo degli oggetti, grafo degli appoggi, azioni passo per passo.

    A differenza delle voci 1-5, che ESEGUONO gli episodi, questa li SPIEGA:
    stampa cosa c'e' in scena e perche' un ordine di rimozione e' migliore di
    un altro. Con la politica 'manual' le rimozioni le scegli tu.
    """
    from physical_ai_mujoco.evaluation.inspection import (
        DESCRIZIONE_POLITICHE,
        POLITICHE,
        ispeziona,
    )

    object_count = ask_int("Quanti oggetti nella scena?", default=3, low=1, high=12)

    print("\n  Politiche disponibili:")
    for nome, spiegazione in DESCRIZIONE_POLITICHE.items():
        print(f"    {nome:<8} {spiegazione}")
    print()
    policy = ask("Quale politica?", default="manual")
    while policy not in POLITICHE:
        print(f"Scegliere fra: {', '.join(POLITICHE)}")
        policy = ask("Quale politica?", default="manual")

    if policy == "manual":
        # A mano si va un episodio per volta: serve il tuo input a ogni passo.
        episodes = 1
    else:
        episodes = ask_int("Quanti episodi?", default=1, low=1, high=10000)

    seed = ask_int("Seed? (0 = casuale)", default=0, low=0, high=10**9)
    render = ask_yes_no("Aprire anche il viewer?", default=False)
    quiet = False
    if episodes > 1:
        quiet = not ask_yes_no(
            "Stampare il dettaglio di ogni episodio?", default=episodes <= 5
        )

    ispeziona(
        object_count=object_count,
        policy=policy,
        seed=seed or None,
        episodes=episodes,
        render=render,
        quiet=quiet,
    )


def inspect_scene(seed=None, visual=False, slow=False) -> None:
    from physical_ai_mujoco.simulation.settling import run_until_settled
    from physical_ai_mujoco.simulation.simulator import Simulator
    from physical_ai_mujoco.simulation.viewer import SimulationViewer
    from scripts.run_phase_0a import create_scene, print_result

    if seed is None:
        seed = ask_int("Seed della scena?", default=42, low=0, high=10**9)
    scene = create_scene(seed)
    output_path = PROJECT_ROOT / f"outputs/scenes/{scene.scene_id}.json"
    scene.save(output_path)
    print(f"Scena: {scene.scene_id}")
    print(f"Descrizione: {output_path}\n")

    simulator = Simulator(scene)
    viewer = SimulationViewer("human", 0.5 if slow else 1.0) if visual else None
    try:
        if viewer is None:
            result = run_until_settled(simulator)
        else:
            from physical_ai_mujoco.simulation.settling import SettlingResult

            viewer.attach(simulator)
            viewer.begin_settle()
            outcome = simulator.step_until_settled(on_step=viewer.on_step)
            result = SettlingResult(
                outcome.settled,
                simulator.time,
                outcome.steps,
                simulator.get_object_states(),
            )
        print_result(result)
    finally:
        if viewer is not None:
            viewer.close()
        simulator.close()


def run_tests() -> None:
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=PROJECT_ROOT,
        check=False,
    )


# ------------------------------------------------------------------ supporto


def make_env(**kwargs):
    import gymnasium as gym

    import physical_ai_mujoco.envs  # noqa: F401  (registra l'environment)

    return gym.make("TargetExtraction-v0", **kwargs)


def ask(question: str, default: str) -> str:
    try:
        answer = input(f"{question} [{default}]: ").strip()
    except EOFError:
        # Ctrl-D: l'utente ha chiuso l'ingresso. Uscire in silenzio, non
        # stampargli addosso una traccia di errore.
        print()
        raise SystemExit(0) from None
    return answer or default


def ask_int(question: str, default: int, low: int, high: int) -> int:
    while True:
        answer = ask(question, str(default))
        if answer.isdigit() and low <= int(answer) <= high:
            return int(answer)
        print(f"Inserire un numero fra {low} e {high}.")


def ask_yes_no(question: str, default: bool) -> bool:
    marker = "S/n" if default else "s/N"
    while True:
        try:
            answer = input(f"{question} [{marker}]: ").strip().lower()
        except EOFError:
            print()
            raise SystemExit(0) from None
        if not answer:
            return default
        if answer in {"s", "si", "y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Rispondere s oppure n.")


def print_banner() -> None:
    print("=" * 64)
    print("  Physical AI con MuJoCo")
    print("=" * 64)
    try:
        import gymnasium
        from importlib.metadata import version

        print(f"  MuJoCo {version('mujoco')} | Gymnasium {gymnasium.__version__}")
    except ImportError:
        print("  Dipendenze mancanti. Esegui:")
        print('    python -m pip install -e ".[test]"')
        raise SystemExit(1)
    print(f"  Python {sys.version.split()[0]}")


if __name__ == "__main__":
    raise SystemExit(main())
