"""Ciclo di vita della scena: costruzione, rilascio, pool e assestamento.

Gli algoritmi di generazione/rilascio sono quelli della baseline 30e5667.
"""

from dataclasses import dataclass
import json
import time
import numpy as np
from physical_ai_mujoco.scene.scene_description import Pose, bounding_radius
from physical_ai_mujoco.scene.scene_generator import (
    generate_scene,
    resample_object_poses,
)
from physical_ai_mujoco.simulation.simulator import Simulator, SettlingOutcome
from physical_ai_mujoco.simulation.viewer import SimulationViewer

TARGET_COLOUR = (1.0, 0.85, 0.1, 1.0)


def _largest_shift(before, after):
    return max(
        (float(np.linalg.norm(after[k] - v)) for k, v in before.items() if k in after),
        default=0.0,
    )


@dataclass
class SceneReset:
    settling: SettlingOutcome
    scene_seed: int
    from_pool: bool


class SceneSession:
    def __init__(
        self,
        inputs,
        task,
        *,
        object_count=None,
        render_mode=None,
        fixed_scene_seed=None,
        capture_camera=None,
        capture_stride=16,
        resample_shapes=True,
        highlight_target=False,
        realtime_factor=1.0,
        scene_pool_size=0,
        fresh_scene_probability=0.1,
    ):
        self._scene_rules = inputs.scene_rules
        self._object_dataset = inputs.objects
        self._ground_dataset = inputs.grounds
        self._simulation_config = inputs.simulation
        self._stereo_config = inputs.env["stereo_camera"]
        self.select_target = task.select_target
        self.fixed_scene_seed = fixed_scene_seed
        self.capture_camera = capture_camera
        self.capture_stride = capture_stride
        self.resample_shapes = resample_shapes
        self.highlight_target = highlight_target
        self.render_mode = render_mode
        self.scene_pool_size = int(scene_pool_size)
        self.fresh_scene_probability = float(fresh_scene_probability)
        self._scene_pool = []
        if object_count is not None:
            self._scene_rules = self._rules_with_object_count(object_count)
        self._release_mode = self._scene_rules["release_mode"]
        self.object_count = len(
            self._scene_rules["object_selection"]["required_type_ids"]
        )
        self.simulator = None
        self.target_id = None
        self._object_ids = ()
        self._release_log = []
        self.viewer = SimulationViewer(render_mode, realtime_factor)

    @property
    def _mujoco_renderer(self):
        return self.viewer.renderer

    def _reset_viewer(self):
        self.viewer.attach(self.simulator)

    def _close_simulator(self):
        self.viewer.close()
        if self.simulator is not None:
            self.simulator.close()
            self.simulator = None

    def settle(self):
        self.viewer.begin_settle()
        return self.simulator.step_until_settled(
            capture_camera=self.capture_camera,
            capture_stride=self.capture_stride,
            on_step=self.viewer.on_step,
        )

    @property
    def object_ids(self):
        return self._object_ids

    @property
    def scene_rules(self):
        return self._scene_rules

    @property
    def release_log(self):
        return list(self._release_log)

    def close(self):
        self._close_simulator()

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
                radius = (
                    math.sqrt(upper["x"] ** 2 + upper["y"] ** 2 + upper["z"] ** 2) / 2.0
                )
            elif shape == "cylinder":
                radius = math.sqrt(upper["radius"] ** 2 + (upper["height"] / 2.0) ** 2)
            elif shape == "sphere":
                radius = upper["radius"]
            else:
                raise ValueError(f"Forma non supportata: {shape}")
            largest = max(largest, radius)
        return largest

    def current_positions(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(
                self.simulator.get_object_state(name).position, dtype=float
            )
            for name in self.simulator.present_objects()
        }

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
                # Si favorisce la sepoltura, senza garantirla: puntare alla pila
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
            before = self.current_positions()

            self.simulator.restore_object(
                item.instance_id,
                Pose(
                    position=(aim_x + offset[0], aim_y + offset[1], drop_z),
                    quaternion=start[item.instance_id].quaternion,
                ),
            )
            target_released = target_released or item.instance_id == self.target_id

            wall_start = time.perf_counter()
            outcome = self.settle()
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
                    "moved_others": _largest_shift(before, self.current_positions()),
                }
            )

        return SettlingOutcome(
            settled=settled,
            steps=steps,
            duration=steps * self.simulator.model.opt.timestep,
            frames=frames,
        )

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

    def reset(self, rng):
        self.np_random = rng

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
        self.target_id = self.select_target(self._object_ids, self.np_random)
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
            outcome = self.settle()

        self._forse_archivia(scene_seed, outcome, None)
        return SceneReset(outcome, scene_seed, False)

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

        self._release_log = list(riga)
        return SceneReset(esito, scene.seed, True)
