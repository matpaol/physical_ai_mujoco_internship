"""Environment Gymnasium per Phase 0B: estrazione di un oggetto target.

Il task
-------
La scena viene generata e lasciata assestare. Uno degli oggetti e' designato
come **target**. A ogni step l'agente sceglie un oggetto da rimuovere; dopo
ogni rimozione la scena viene lasciata riassestare fino alla quiete, e si
misura di quanto si e' spostato il resto del mucchio. L'obiettivo e' arrivare
a rimuovere il target **disturbando il meno possibile** gli altri oggetti.

Due modi di finire, scelti da `configs/phase_0b/env.json`
---------------------------------------------------------
- `terminate_on_target` / `terminate_on_collapse` a **False** (il default):
  l'episodio prosegue finche' la scena non e' vuota. E' la modalita' di
  **misura**: ogni rimozione produce un dato — quanto si e' mosso il mucchio,
  con quanti oggetti ancora in scena — invece di chiudere la partita alla
  prima mossa. Serve a conoscere la distribuzione dei disturbi, che e' quello
  che poi permette di scegliere una soglia con un criterio invece che a occhio.
- entrambi a **True**: l'episodio e' un task, e finisce quando si rimuove il
  target (successo) o quando il disturbo supera la soglia (crollo).

In entrambi i casi il disturbo viene calcolato e riportato allo stesso modo:
cambia solo se termina l'episodio. Nessun modulo sa in quale delle due
modalita' si trova.

Teacher-student
---------------
L'env supporta tre modalita' di osservazione, selezionate da `obs_mode`:

- ``"state"``   : stato privilegiato (pose, velocita', masse, attriti, maschera).
                  E' l'osservazione del **teacher**, che si allena con RL.
- ``"stereo"``  : solo la coppia di immagini RGB dal rig stereo.
                  E' l'osservazione dello **student**.
- ``"both"``    : entrambe, nello stesso dizionario.

``"both"`` e' la modalita' che serve per la **distillazione**: un unico rollout
produce sia lo stato su cui il teacher decide, sia le immagini su cui lo
student deve imparare a riprodurre quella decisione. Raccogliere i due flussi
da rollout separati non funzionerebbe, perche' la fisica dei contatti e'
caotica e le due traiettorie divergerebbero.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from physical_ai_mujoco.envs import compat  # noqa: F401  (patch MuJoCo >= 3.13)
from physical_ai_mujoco.scene.dataset_loader import (
    load_ground_dataset,
    load_object_dataset,
    load_scene_rules,
    load_simulation_config,
)
from physical_ai_mujoco.scene.scene_description import Pose, bounding_radius
from physical_ai_mujoco.scene.scene_generator import (
    generate_scene,
    resample_object_poses,
)
from physical_ai_mujoco.simulation.simulator import SettlingOutcome, Simulator

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Valori per oggetto nel vettore di stato del teacher.
STATE_FEATURES_PER_OBJECT = 17

# Giallo acceso per il target, quando highlight_target e' attivo.
TARGET_COLOUR = (1.0, 0.85, 0.1, 1.0)

# Titolo della finestra del viewer di Gymnasium, che altrimenti si
# chiamerebbe solo "mujoco".
WINDOW_TITLE = "Gymnasium viewer - TargetExtraction-v0"


def _choose(override, configured) -> bool:
    """`override` se e' stato dato, altrimenti il valore del file di config."""
    return bool(configured if override is None else override)


def _largest_shift(before: dict, after: dict) -> float:
    """Spostamento massimo fra gli oggetti presenti in entrambe le letture."""
    largest = 0.0
    for name, start_position in before.items():
        if name in after:
            largest = max(
                largest, float(np.linalg.norm(after[name] - start_position))
            )
    return largest


class TargetExtractionEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        obs_mode: str = "state",
        render_mode: str | None = None,
        env_config_path: str | Path | None = None,
        scene_rules_path: str | Path | None = None,
        fixed_scene_seed: int | None = None,
        capture_camera: str | None = None,
        capture_stride: int = 16,
        object_count: int | None = None,
        resample_shapes: bool = True,
        highlight_target: bool = False,
        realtime_factor: float = 1.0,
        terminate_on_target: bool | None = None,
        terminate_on_collapse: bool | None = None,
        scene_pool_size: int = 0,
        fresh_scene_probability: float = 0.1,
    ):
        super().__init__()

        if obs_mode not in {"state", "stereo", "both"}:
            raise ValueError(
                f"obs_mode deve essere 'state', 'stereo' o 'both': {obs_mode!r}"
            )
        self.obs_mode = obs_mode
        self.render_mode = render_mode
        self.fixed_scene_seed = fixed_scene_seed
        # Cattura frame per la visualizzazione. Non influenza ne' la fisica
        # ne' l'osservazione: e' puro costo di rendering, quindi resta spenta
        # durante l'allenamento.
        self.capture_camera = capture_camera
        self.capture_stride = capture_stride
        # Colora il target di giallo per renderlo riconoscibile a colpo
        # d'occhio. Cambia cio' che vedono LE CAMERE, quindi va lasciato
        # spento quando le immagini sono l'osservazione dello student.
        self.highlight_target = highlight_target
        # Quanto veloce far scorrere il viewer interattivo: 1 = tempo reale,
        # 0.5 = rallentatore, 0 = piu' veloce possibile.
        self.realtime_factor = realtime_factor

        # Magazzino di scene gia' costruite e assestate. Con `scene_pool_size`
        # a 0 e' spento e ogni reset costruisce da capo, che e' il
        # comportamento giusto per misurare. Serve per ALLENARE: costruire una
        # scena costa 0,73 s contro i 0,045 s di una mossa, cioe' l'84% del
        # tempo se ne va a far cadere oggetti invece che a imparare.
        #
        # `fresh_scene_probability` tiene aperta la porta alle scene nuove
        # anche a magazzino pieno: pescare sempre e solo dal magazzino
        # congelerebbe la varieta' al momento in cui si e' riempito, mentre
        # cosi' continua a crescere per tutto l'allenamento.
        self.scene_pool_size = int(scene_pool_size)
        self.fresh_scene_probability = float(fresh_scene_probability)
        self._scene_pool: list[tuple] = []

        env_config_path = Path(
            env_config_path or PROJECT_ROOT / "configs/phase_0b/env.json"
        )
        self._env_config = json.loads(env_config_path.read_text(encoding="utf-8"))
        self._stereo_config = self._env_config["stereo_camera"]
        self._task = self._env_config["task"]

        self._scene_rules = load_scene_rules(
            scene_rules_path or PROJECT_ROOT / "configs/phase_0b/scene_rules.json"
        )
        # I dataset sono configurazione, non percorsi fissi: Fase 0B usa
        # oggetti con attrito volvente realistico, mentre Fase 0A resta sui
        # suoi e continua a produrre scene byte-identiche a parita' di seed.
        datasets = self._env_config.get("datasets", {})
        self._object_dataset = load_object_dataset(
            PROJECT_ROOT
            / datasets.get("objects", "datasets/object_dataset/geometric_objects.json")
        )
        self._ground_dataset = load_ground_dataset(
            PROJECT_ROOT
            / datasets.get("grounds", "datasets/ground_dataset/basic_grounds.json")
        )
        # Fase 0B ha un timeout di assestamento piu' lungo: con rilascio
        # sequenziale ogni caduta viene attesa singolarmente, e i 5 s di Fase
        # 0A lasciavano una scena su tre ancora in movimento — cioe' misurata
        # male. Le soglie di quiete restano identiche: cambiare quelle
        # cambierebbe il significato del disturbo.
        self._simulation_config = load_simulation_config(
            PROJECT_ROOT
            / datasets.get("simulation", "configs/phase_0a/simulation.json")
        )

        if object_count is not None:
            self._scene_rules = self._rules_with_object_count(object_count)

        # Con resample_shapes=False la scena viene costruita una volta sola e
        # i reset successivi rimescolano soltanto le pose. Il modello MuJoCo
        # non viene ricompilato, quindi una finestra interattiva aperta resta
        # valida per tutta la sessione invece di chiudersi a ogni episodio.
        # In cambio si perde la randomizzazione di forme e dimensioni, che
        # durante l'allenamento invece serve.
        self.resample_shapes = resample_shapes

        self._release_mode = self._scene_rules["release_mode"]
        self.object_count = len(
            self._scene_rules["object_selection"]["required_type_ids"]
        )
        self.action_space = spaces.Discrete(self.object_count)
        self.observation_space = self._build_observation_space()

        # Quando finisce un episodio. Entrambi a False (il default in
        # `env.json`) l'episodio prosegue finche' la scena non e' vuota: e' la
        # modalita' di MISURA, in cui ogni rimozione produce un dato invece di
        # chiudere la partita. Rimetterli a True restituisce il task.
        # Il file di configurazione fissa la modalita' dell'esperimento; questi
        # due argomenti la scavalcano per un singolo environment, che serve a
        # mettere a confronto le due letture sulla stessa scena.
        self._terminate_on_target = _choose(
            terminate_on_target, self._task.get("terminate_on_target", True)
        )
        self._terminate_on_collapse = _choose(
            terminate_on_collapse, self._task.get("terminate_on_collapse", True)
        )

        self.simulator: Simulator | None = None
        self.target_id: str | None = None
        self._object_ids: tuple[str, ...] = ()
        self._reference_positions: dict[str, np.ndarray] = {}
        self._total_disturbance = 0.0
        self._ever_collapsed = False
        self._release_log: list[dict] = []
        self._mujoco_renderer = None
        self._window_named = False
        # Ancora (tempo reale, tempo simulato) da cui si misura se la
        # riproduzione e' in anticipo o in ritardo. Si riazzera a ogni
        # assestamento.
        self._pacing_origin: tuple[float, float] | None = None
        # Aggancio opzionale per chi disegna per conto proprio: viene
        # chiamato con la stessa cadenza del rendering interno.
        self.step_callback = None

    def _rules_with_object_count(self, object_count: int) -> dict:
        """Regole di scena con `object_count` oggetti invece di quelli in config.

        La zona di rilascio cresce **solo in altezza**. Allargarla anche in
        pianta, come si faceva prima, disperde gli oggetti su tutto il
        pavimento: cadono lontani, non si toccano, e il problema dell'ordine
        di rimozione sparisce insieme alla pila. La colonna stretta del
        config e' cio' che li fa impilare, e resta quella.

        Limite noto di questo schema: il campionamento rifiuta due posizioni
        i cui ingombri si sovrappongono, quindi con molti oggetti nella
        stessa colonna serve molta quota, e da alto l'impatto e' violento —
        proprio il regime che i simulatori riproducono peggio. La soluzione
        vera e' rilasciare gli oggetti **uno alla volta** da poco sopra la
        cima della pila, invece che tutti insieme da altezze diverse.
        """
        if object_count < 1:
            raise ValueError("Serve almeno 1 oggetto")

        # Si ciclano i tipi ELENCATI NEL CONFIG, non tutto il dataset: e' il
        # config a dire quali forme entrano in scena, e --objects solo quante.
        # (Ciclare tutto il dataset tirerebbe dentro le barre lunghe, che con
        # il loro raggio d'ingombro non stanno in una pila e rendono
        # l'assestamento caotico.)
        rules = json.loads(json.dumps(self._scene_rules))
        available = list(rules["object_selection"]["required_type_ids"])
        rules["object_selection"]["required_type_ids"] = [
            available[index % len(available)] for index in range(object_count)
        ]

        spawn = rules["spawn"]
        radius = self._largest_bounding_radius(
            rules["object_selection"]["required_type_ids"]
        )
        if rules["release_mode"] == "simultaneous":
            # Cadendo tutti insieme, due oggetti nella stessa colonna vanno
            # separati di almeno la somma dei raggi d'ingombro. Con rilascio
            # sequenziale la quota iniziale viene ricalcolata a ogni caduta e
            # questo intervallo non conta.
            lowest = float(spawn["z_range"][0])
            spawn["z_range"] = [lowest, lowest + 2.0 * radius * object_count]
        spawn["max_attempts_per_object"] = max(
            int(spawn["max_attempts_per_object"]), 300 * object_count
        )
        return rules

    def _largest_bounding_radius(self, type_ids) -> float:
        """Raggio della sfera che contiene il piu' grande oggetto possibile."""
        import math

        definitions = {
            item["id"]: item for item in self._object_dataset["object_types"]
        }
        largest = 0.0
        for type_id in set(type_ids):
            definition = definitions[type_id]
            upper = {
                name: float(bounds[1])
                for name, bounds in definition["size_range"].items()
            }
            shape = definition["shape"]
            if shape == "box":
                radius = math.sqrt(
                    upper["x"] ** 2 + upper["y"] ** 2 + upper["z"] ** 2
                ) / 2.0
            elif shape == "cylinder":
                radius = math.sqrt(
                    upper["radius"] ** 2 + (upper["height"] / 2.0) ** 2
                )
            elif shape == "sphere":
                radius = upper["radius"]
            else:
                raise ValueError(f"Forma non supportata: {shape}")
            largest = max(largest, radius)
        return largest

    @property
    def task(self) -> dict:
        """Parametri del task: soglie, pesi della reward, regole di fine."""
        return dict(self._task)

    # --------------------------------------------------------------- Gymnasium

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        ripresa = self._prendi_dal_magazzino()
        if ripresa is not None:
            return ripresa

        scene_seed = (
            self.fixed_scene_seed
            if self.fixed_scene_seed is not None
            else int(self.np_random.integers(0, 2**31 - 1))
        )

        if self.simulator is not None and not self.resample_shapes:
            # Modello invariato: si rimescolano solo le pose iniziali.
            scene = self.simulator.scene
            poses = resample_object_poses(scene, self._scene_rules, self.np_random)
            self.simulator.reset_to_poses(poses)
        else:
            scene = generate_scene(
                self._object_dataset,
                self._ground_dataset,
                self._scene_rules,
                self._simulation_config,
                scene_seed,
                stereo_config=self._stereo_config,
            )
            self._close_simulator()
            self.simulator = Simulator(scene)

        self._object_ids = tuple(item.instance_id for item in scene.objects)

        # Il target si sceglie PRIMA di far cadere qualcosa. Con rilascio
        # sequenziale la posizione nell'ordine di caduta e' la cosa che
        # DECIDE se il problema esiste: n oggetti, poi il target, poi altri m.
        # Pescando l'indice fra 0 e N-2 si garantisce m >= 1, cioe' che
        # qualcosa finisca sopra al target. Sceglierlo a caso dopo la caduta,
        # come si faceva prima, lasciava il target in cima in una frazione
        # notevole delle scene, e in quelle scene l'ordine non conta.
        last_useful_position = max(self.object_count - 1, 1)
        target_index = int(self.np_random.integers(0, last_useful_position))
        self.target_id = self._object_ids[target_index]
        if self.highlight_target:
            self.simulator.set_object_color(self.target_id, TARGET_COLOUR)

        # Il viewer va creato PRIMA dell'assestamento, altrimenti la caduta
        # iniziale del primo episodio non si vede.
        if self.render_mode == "human" and (
            self._mujoco_renderer is None or self.resample_shapes
        ):
            self._reset_viewer()

        if self._release_mode == "sequential":
            outcome = self._release_sequentially(scene)
        else:
            outcome = self._settle()

        # Le posizioni di riferimento si registrano DOPO: il disturbo va
        # misurato rispetto a una configurazione gia' stabile.
        self._reference_positions = self._current_positions()
        self._total_disturbance = 0.0
        self._ever_collapsed = False

        info = self._build_info(
            disturbance=0.0, invalid_action=False, settling=outcome
        )
        info["scene_seed"] = scene_seed
        # Come e' stata costruita la scena, un oggetto per riga. Serve a
        # controllare la FISICA, non l'agente: tempi di assestamento, altezze
        # di caduta, e quanto ogni caduta ha smosso cio' che c'era gia'.
        info["release_log"] = list(self._release_log)
        info["from_pool"] = False
        self._forse_archivia(scene_seed, outcome, info)
        return self._observation(), info

    # ------------------------------------------------------- magazzino scene

    def _prendi_dal_magazzino(self):
        """Riparte da una scena gia' costruita, se ce n'e' una e si pesca cosi'.

        Restituisce `(osservazione, info)` come `reset`, oppure `None` per
        dire «costruiscila».
        """
        # Finche' il magazzino non e' pieno si costruisce sempre: riempirlo
        # al ritmo di `fresh_scene_probability` vorrebbe dire rigiocare le
        # prime quattro scene per centinaia di episodi prima di averne
        # cinquanta. Si paga il prezzo pieno all'inizio, una volta.
        if len(self._scene_pool) < self.scene_pool_size:
            return None
        if not self._scene_pool:
            return None
        if self.np_random.random() < self.fresh_scene_probability:
            return None

        indice = int(self.np_random.integers(len(self._scene_pool)))
        scene, istantanea, target_id, esito, riga = self._scene_pool[indice]

        # Il modello va ricompilato, ma e' l'operazione economica: il costo
        # della costruzione sta nel far CADERE gli oggetti, non nel
        # compilarli.
        if self.simulator is None or self.simulator.scene.scene_id != scene.scene_id:
            self._close_simulator()
            self.simulator = Simulator(scene)
        self.simulator.restore(istantanea)

        self._object_ids = tuple(item.instance_id for item in scene.objects)
        self.target_id = target_id
        if self.highlight_target:
            self.simulator.set_object_color(self.target_id, TARGET_COLOUR)
        if self.render_mode == "human" and self._mujoco_renderer is None:
            self._reset_viewer()

        self._reference_positions = self._current_positions()
        self._total_disturbance = 0.0
        self._ever_collapsed = False
        self._release_log = list(riga)

        info = self._build_info(disturbance=0.0, invalid_action=False, settling=esito)
        info["scene_seed"] = scene.seed
        info["release_log"] = list(riga)
        info["from_pool"] = True
        return self._observation(), info

    def _forse_archivia(self, scene_seed: int, outcome, info) -> None:
        """Mette in magazzino la scena appena costruita, se c'e' posto.

        Si archivia DOPO l'assestamento e prima di qualsiasi rimozione: e'
        quello lo stato da cui un episodio deve ripartire.
        """
        if self.scene_pool_size <= 0:
            return

        voce = (
            self.simulator.scene,
            self.simulator.snapshot(),
            self.target_id,
            outcome,
            list(self._release_log),
        )
        if len(self._scene_pool) < self.scene_pool_size:
            self._scene_pool.append(voce)
            return

        # Magazzino pieno: la scena nuova ne SOSTITUISCE una vecchia. E'
        # quello che fa crescere la varieta' per tutto l'allenamento invece di
        # congelarla al momento in cui il magazzino si e' riempito.
        self._scene_pool[int(self.np_random.integers(self.scene_pool_size))] = voce

    def _settle(self):
        """Lascia assestare la scena e restituisce l'esito."""
        self._pacing_origin = None
        return self.simulator.step_until_settled(
            capture_camera=self.capture_camera,
            capture_stride=self.capture_stride,
            on_step=self._live_render,
        )

    def _release_sequentially(self, scene) -> SettlingOutcome:
        """Fa cadere gli oggetti uno alla volta, ciascuno da sopra la pila.

        Rilasciarli tutti insieme obbliga a partire gia' distanti — altrimenti
        si compenetrano — e quindi o si allarga la scena, e allora non si
        forma nessuna pila, o si alza la colonna, e allora cadono da metri e
        l'impatto li disperde. Uno alla volta nessuno dei due problemi esiste:
        la posa iniziale puo' anche sovrapporsi (l'oggetto e' ancora fuori
        scena) e l'altezza di caduta resta di pochi centimetri.
        """
        # Quanto sopra la cima della pila lasciar cadere il prossimo oggetto.
        # Poca: l'energia d'impatto e' cio' che disperde il mucchio.
        clearance = float(self._scene_rules["spawn"].get("release_clearance", 0.04))
        # Le pose di partenza si leggono dal SIMULATORE, non dalla descrizione
        # della scena: con `resample_shapes=False` i reset rimescolano le pose
        # senza ricompilare, e la descrizione resta ferma a quelle originali.
        # Leggendo da li' ogni episodio ripartirebbe identico al primo.
        start = {
            item.instance_id: self.simulator.get_object_state(item.instance_id)
            for item in scene.objects
        }
        for instance_id in self._object_ids:
            self.simulator.remove_object(instance_id)

        frames: list[np.ndarray] = []
        steps = 0
        settled = True
        target_released = False
        self._release_log = []

        for index, item in enumerate(scene.objects):
            radius = bounding_radius(item.shape, item.size)
            summit_x, summit_y, summit_z = self.simulator.pile_summit()

            if index == 0:
                # Il primo oggetto tocca terra dove dicono le regole di scena:
                # e' quel punto a decidere dove nasce la pila.
                aim_x, aim_y, aim_z = (*start[item.instance_id].position[:2], 0.0)
            elif target_released:
                # Dopo il target si mira al TARGET, non alla cima generica.
                # E' cio' che rende la sepoltura una proprieta' garantita per
                # costruzione invece di un esito fortunato: puntare alla pila
                # in generale lascia il target scoperto ogni volta che e'
                # rotolato di lato, e in quelle scene l'ordine non conta.
                aim_x, aim_y, aim_z = self._object_summit(self.target_id)
            else:
                aim_x, aim_y, aim_z = summit_x, summit_y, summit_z

            # Scarto dell'ordine di mezzo raggio: senza, si otterrebbe una
            # torre perfetta e irreale; molto piu' grande, si manca la pila e
            # si finisce per terra accanto.
            offset = self.np_random.uniform(-radius / 2.0, radius / 2.0, size=2)
            if index == 0:
                offset[:] = 0.0

            drop_z = aim_z + clearance + radius
            # Pose degli oggetti gia' in scena, per misurare quanto li smuove
            # questa caduta: e' il controllo che dice se la pila regge o se
            # ogni aggiunta la rimescola.
            before = self._current_positions()

            self.simulator.restore_object(
                item.instance_id,
                Pose(
                    position=(aim_x + offset[0], aim_y + offset[1], drop_z),
                    quaternion=start[item.instance_id].quaternion,
                ),
            )
            target_released = target_released or item.instance_id == self.target_id

            wall_start = time.perf_counter()
            outcome = self._settle()
            wall_elapsed = time.perf_counter() - wall_start

            frames.extend(outcome.frames)
            steps += outcome.steps
            settled = settled and outcome.settled

            final = self.simulator.get_object_state(item.instance_id).position
            self._release_log.append(
                {
                    "index": index,
                    "instance_id": item.instance_id,
                    "type_id": item.type_id,
                    "is_target": item.instance_id == self.target_id,
                    "mass": item.mass,
                    "radius": radius,
                    "drop_height": drop_z,
                    # Quanto e' sceso il baricentro: la caduta vera, non la
                    # quota assoluta. E' questa a dire quanta energia arriva
                    # al mucchio.
                    "fall_distance": drop_z - float(final[2]),
                    "settling_seconds": outcome.duration,
                    "settling_steps": outcome.steps,
                    "wall_seconds": wall_elapsed,
                    "settled": outcome.settled,
                    "final_position": tuple(float(v) for v in final),
                    "resting_on": self.simulator.supported_by(item.instance_id),
                    "moved_others": _largest_shift(
                        before, self._current_positions()
                    ),
                }
            )

        return SettlingOutcome(
            settled=settled,
            steps=steps,
            duration=steps * self.simulator.model.opt.timestep,
            frames=frames,
        )

    def step(self, action):
        if self.simulator is None:
            raise RuntimeError("Chiama reset() prima di step()")

        action = int(action)
        if not 0 <= action < self.object_count:
            raise ValueError(f"Azione fuori range: {action}")

        instance_id = self._object_ids[action]

        # Azione non valida: l'oggetto e' gia' stato rimosso. Non si solleva
        # un'eccezione (bloccherebbe l'allenamento di una policy senza
        # maschera): si applica una penalita' e si lascia proseguire.
        if not self.simulator.is_present(instance_id):
            reward = -float(self._task["removal_cost"])
            info = self._build_info(disturbance=0.0, invalid_action=True)
            return self._observation(), reward, False, False, info

        self.simulator.remove_object(instance_id)
        outcome = self._settle()

        disturbance = self._disturbance()
        self._total_disturbance += disturbance
        # Il riferimento si sposta a ogni mossa: cosi' `disturbance` misura
        # quanto il mucchio si e' spostato PER QUESTA rimozione, non quanto si
        # e' spostato da inizio episodio. Cumulando, il valore cresce e satura
        # e non dice piu' quale mossa abbia fatto danno.
        self._reference_positions = self._current_positions()

        target_removed = not self.simulator.is_present(self.target_id)
        collapsed = disturbance > float(self._task["disturbance_threshold"])
        self._ever_collapsed = self._ever_collapsed or collapsed
        scene_is_empty = not self.simulator.present_objects()

        reward = -float(self._task["removal_cost"])
        reward -= float(self._task["disturbance_penalty"]) * disturbance
        # Il premio si paga sulla MOSSA che estrae il target, non finche' il
        # target risulta estratto: senza questa distinzione, un episodio che
        # continua dopo l'estrazione incassa il premio a ogni passo
        # successivo, e la ricompensa smette di dire qualcosa sull'ordine.
        if instance_id == self.target_id:
            reward += float(self._task["target_reward"])
        if collapsed:
            reward -= float(self._task["failure_penalty"])

        # Quando termina l'episodio e' una scelta di configurazione, non una
        # proprieta' della scena. In campagna di misura si svuota la scena e
        # si guarda ogni rimozione; come task, si chiude al target o al
        # crollo. I due flag qui sotto commutano fra le due letture senza
        # toccare il codice.
        terminated = scene_is_empty
        if target_removed and self._terminate_on_target:
            terminated = True
        if collapsed and self._terminate_on_collapse:
            terminated = True

        info = self._build_info(
            disturbance=disturbance, invalid_action=False, settling=outcome
        )
        # `target_removed` e `collapsed` descrivono QUESTO passo; i due flag
        # con `_ever` descrivono l'episodio. La distinzione conta solo da
        # quando l'episodio prosegue oltre l'estrazione: prima coincidevano, e
        # confonderle faceva risultare "successo" un episodio in cui la pila
        # era crollata a meta' strada.
        info["target_just_removed"] = instance_id == self.target_id
        info["target_removed"] = target_removed
        info["collapsed"] = collapsed
        info["collapsed_ever"] = self._ever_collapsed
        info["scene_is_empty"] = scene_is_empty
        info["is_success"] = bool(target_removed and not self._ever_collapsed)

        if self.render_mode == "human":
            self.render()

        return self._observation(), reward, terminated, False, info

    def _object_summit(self, instance_id: str) -> tuple[float, float, float]:
        """Il punto piu' alto di un singolo oggetto: `(x, y, z)`."""
        item = next(
            entry
            for entry in self.simulator.scene.objects
            if entry.instance_id == instance_id
        )
        centre = self.simulator.get_object_state(instance_id).position
        return (
            centre[0],
            centre[1],
            centre[2] + bounding_radius(item.shape, item.size),
        )

    def _live_render(self) -> None:
        """Aggiorna la visualizzazione DURANTE l'assestamento.

        Senza questo si vedrebbe un aggiornamento per step, cioe' uno per
        rimozione: dei salti fra configurazioni statiche invece degli oggetti
        che si muovono.

        Due destinatari possibili:
        - il viewer di Gymnasium, quando render_mode == "human";
        - `step_callback`, un aggancio per chi disegna per conto proprio.
        """
        if self.step_callback is None and (
            self.render_mode != "human" or self._mujoco_renderer is None
        ):
            return

        # Alla massima velocita' non c'e' niente da sincronizzare: si disegna
        # ogni volta che capita e si va avanti.
        if self.realtime_factor <= 0:
            self._draw()
            return

        now = time.perf_counter()
        if self._pacing_origin is None:
            self._pacing_origin = (now, self.simulator.time)
        wall_origin, sim_origin = self._pacing_origin

        # Il tempo simulato che a quest'ora dovremmo aver raggiunto, se la
        # riproduzione andasse a `realtime_factor` volte il tempo reale.
        due = sim_origin + (now - wall_origin) * self.realtime_factor
        ahead = self.simulator.time - due

        if ahead < 0:
            # Siamo in ritardo: si SALTA il fotogramma e si lascia correre la
            # fisica. E' cio' che distingue una riproduzione a velocita' reale
            # da una al rallentatore — disegnare costa decine di millisecondi,
            # molto piu' del tempo simulato che quel fotogramma rappresenta, e
            # disegnarli tutti significa per forza andare piu' piano del reale.
            # Meno fotogrammi, stessa durata: e' quello che fa qualsiasi
            # riproduzione video quando la macchina non ce la fa.
            return

        self._draw()
        # Si dorme cio' che resta DOPO aver disegnato, non il budget intero.
        remaining = ahead - (time.perf_counter() - now)
        if remaining > 0:
            time.sleep(remaining)

    def _draw(self) -> None:
        if self.step_callback is not None:
            self.step_callback()
        if self.render_mode == "human" and self._mujoco_renderer is not None:
            self._mujoco_renderer.render("human")
            self._name_the_window()

    def _name_the_window(self) -> None:
        """Rinomina la finestra del viewer di Gymnasium.

        Gymnasium crea la sua finestra GLFW con il titolo letterale "mujoco"
        (gymnasium/envs/mujoco/mujoco_rendering.py, glfw.create_window). Qui
        le si da' un nome che dice cosa si sta guardando.
        """
        if self._window_named:
            return
        viewer = getattr(self._mujoco_renderer, "viewer", None)
        window = getattr(viewer, "window", None)
        if window is None:
            return
        try:
            import glfw

            glfw.set_window_title(window, WINDOW_TITLE)
            self._window_named = True
        except Exception:  # noqa: BLE001  (il titolo e' un dettaglio estetico)
            self._window_named = True

    def render(self):
        if self.render_mode == "human":
            if self._mujoco_renderer is not None:
                return self._mujoco_renderer.render("human")
            return None
        if self.render_mode == "rgb_array":
            return self.simulator.render_camera("cam_left")
        return None

    def close(self):
        self._close_simulator()

    # ------------------------------------------------------------ action mask

    def action_masks(self) -> np.ndarray:
        """Maschera booleana delle azioni valide.

        Il nome segue la convenzione di sb3-contrib `MaskablePPO`, che cerca
        proprio un metodo `action_masks()` sull'environment. La stessa maschera
        e' anche in `info["action_mask"]` a ogni step.
        """
        if self.simulator is None:
            return np.ones(self.object_count, dtype=bool)
        return np.array(
            [self.simulator.is_present(name) for name in self._object_ids],
            dtype=bool,
        )

    # ---------------------------------------------------------- osservazioni

    def _build_observation_space(self) -> spaces.Space:
        state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.object_count * STATE_FEATURES_PER_OBJECT,),
            dtype=np.float32,
        )
        image_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                int(self._stereo_config["height"]),
                int(self._stereo_config["width"]),
                3,
            ),
            dtype=np.uint8,
        )

        if self.obs_mode == "state":
            return state_space
        if self.obs_mode == "stereo":
            return spaces.Dict(
                {"rgb_left": image_space, "rgb_right": image_space}
            )
        return spaces.Dict(
            {
                "state": state_space,
                "rgb_left": image_space,
                "rgb_right": image_space,
            }
        )

    def _observation(self):
        if self.obs_mode == "state":
            return self._state_observation()

        left, right = self.simulator.render_stereo()
        if self.obs_mode == "stereo":
            return {"rgb_left": left, "rgb_right": right}
        return {
            "state": self._state_observation(),
            "rgb_left": left,
            "rgb_right": right,
        }

    def _state_observation(self) -> np.ndarray:
        """Stato privilegiato del teacher.

        Per ogni oggetto, nell'ordine fisso di `self._object_ids`:
        posizione (3), quaternione (4), velocita' lineare (3), angolare (3),
        massa (1), attrito radente (1), presente (1), e' il target (1).
        """
        features = []
        for item in self.simulator.scene.objects:
            state = self.simulator.get_object_state(item.instance_id)
            present = self.simulator.is_present(item.instance_id)
            features.extend(
                [
                    *state.position,
                    *state.quaternion,
                    *state.linear_velocity,
                    *state.angular_velocity,
                    item.mass,
                    item.friction[0],
                    1.0 if present else 0.0,
                    1.0 if item.instance_id == self.target_id else 0.0,
                ]
            )
        return np.asarray(features, dtype=np.float32)

    # ------------------------------------------------------------- meccaniche

    def _current_positions(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(
                self.simulator.get_object_state(name).position, dtype=float
            )
            for name in self.simulator.present_objects()
        }

    def _disturbance(self) -> float:
        """Spostamento massimo dall'ULTIMA configurazione stabile.

        Si usa il massimo e non la media perche' cio' che conta e' che UN
        oggetto si muova troppo: una media lo diluirebbe fra tutti gli altri
        rimasti fermi, e con molti oggetti non emergerebbe mai.

        Il target e' escluso: e' l'oggetto da recuperare, e il suo spostamento
        e' un'altra grandezza, non un disturbo causato all'ambiente.
        """
        maximum = 0.0
        for name, position in self._current_positions().items():
            if name == self.target_id:
                continue
            reference = self._reference_positions.get(name)
            if reference is None:
                continue
            maximum = max(maximum, float(np.linalg.norm(position - reference)))
        return maximum

    def _build_info(
        self,
        disturbance: float,
        invalid_action: bool,
        settling=None,
    ) -> dict:
        return {
            "action_mask": self.action_masks(),
            "target_id": self.target_id,
            "target_index": (
                self._object_ids.index(self.target_id)
                if self.target_id in self._object_ids
                else -1
            ),
            "present_objects": self.simulator.present_objects(),
            # Due grandezze diverse, entrambe utili: la prima dice quale mossa
            # ha fatto danno, la seconda quanto se n'e' accumulato.
            "disturbance_step": disturbance,
            "disturbance_total": self._total_disturbance,
            "invalid_action": invalid_action,
            # False significa che il timeout dell'assestamento e' scaduto con
            # qualcosa ancora in movimento: le pose lette sopra non
            # descrivono una configurazione stabile.
            "settled": True if settling is None else settling.settled,
            "settling_duration": 0.0 if settling is None else settling.duration,
            "frames": [] if settling is None else settling.frames,
        }

    # --------------------------------------------------------------- interno

    def _reset_viewer(self) -> None:
        """(Ri)crea il viewer interattivo di Gymnasium.

        Il modello MuJoCo viene ricompilato a ogni reset (le dimensioni degli
        oggetti sono campionate per scena), quindi anche il viewer va
        ricreato: con render_mode="human" la finestra si richiude e riapre a
        ogni episodio. E' accettabile per ispezionare, non per allenare.
        """
        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        if self._mujoco_renderer is not None:
            self._mujoco_renderer.close()
        self._window_named = False
        self._mujoco_renderer = MujocoRenderer(
            self.simulator.model,
            self.simulator.data,
            default_cam_config={
                "distance": 2.0,
                "azimuth": 110.0,
                "elevation": -25.0,
                "lookat": np.array([0.0, 0.0, 0.1]),
            },
        )

    def _close_simulator(self) -> None:
        if self._mujoco_renderer is not None:
            self._mujoco_renderer.close()
            self._mujoco_renderer = None
        if self.simulator is not None:
            self.simulator.close()
            self.simulator = None
