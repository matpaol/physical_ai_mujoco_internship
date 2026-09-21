from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from physical_ai_mujoco.scene.scene_description import (
    SceneDescription,
    bounding_radius,
)
from physical_ai_mujoco.simulation.mujoco_builder import _camera_axes, build_model
from physical_ai_mujoco.simulation.visual_conditions import (
    BackgroundAppearance,
    CameraPerturbation,
    GroundAppearance,
    LightingConditions,
    VisualConditions,
)


# Nomi degli elementi visivi dell'MJCF (vedi mujoco_builder.build_mjcf).
MAIN_LIGHT = "sole"
GROUND_GEOM = "ground_geom"
GROUND_MATERIAL = "mat_piano"
GROUND_TEXTURE = "griglia"
BACKGROUND_TEXTURE = "skybox"
STEREO_CAMERAS = (("cam_left", -1.0), ("cam_right", +1.0))

# Posizione dove vengono "parcheggiati" gli oggetti rimossi: fuori dal campo
# visivo di qualsiasi camera ragionevole e senza contatti attivi.
PARKING_POSITION = (0.0, 0.0, -50.0)

# Nome del corpo del terreno nell'MJCF (vedi mujoco_builder._add_ground).
GROUND_BODY = "ground"


@dataclass(frozen=True)
class ObjectState:
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]


@dataclass(frozen=True)
class SettlingOutcome:
    """Esito di un assestamento.

    `settled` a False significa che il timeout e' scaduto con qualcosa ancora
    in movimento. Non e' un errore — un oggetto che rotola via puo' non
    fermarsi mai — ma le pose lette dopo non descrivono una configurazione
    stabile, e chi misura deve saperlo.
    """

    settled: bool
    steps: int
    duration: float
    frames: list[np.ndarray] = field(default_factory=list)


class Simulator:
    """Avvolge un modello MuJoCo costruito da una SceneDescription.

    Oltre alla simulazione di Phase 0A offre le operazioni che servono a un
    environment Gymnasium: rimozione di oggetti, reset completo e rendering
    offscreen dalle camere stereo.
    """

    def __init__(self, scene: SceneDescription):
        self.scene = scene
        self.model, self.data = build_model(scene)
        self._joint_ids = {
            item.instance_id: mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{item.instance_id}_joint",
            )
            for item in scene.objects
        }
        self._body_ids = {
            item.instance_id: mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                item.instance_id,
            )
            for item in scene.objects
        }
        self._removed: set[str] = set()
        # Oggetti tenuti fermi nella posa corrente mentre il resto si assesta.
        self._held: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._renderer: mujoco.Renderer | None = None

        # Copie di riferimento per poter annullare le rimozioni in reset().
        self._initial_qpos = self.data.qpos.copy()
        self._initial_contype = self.model.geom_contype.copy()
        self._initial_conaffinity = self.model.geom_conaffinity.copy()
        self._initial_rgba = self.model.geom_rgba.copy()
        # Aspetto nominale salvato al primo cambio visivo, per poterlo ripristinare.
        self._visual_nominal: dict = {}

    # ------------------------------------------------------------ simulazione

    @property
    def time(self) -> float:
        return float(self.data.time)

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)
        self._hold_removed_objects()
        for instance_id, (position, quaternion) in self._held.items():
            self._write_pose(instance_id, position, quaternion)

    def hold_object(self, instance_id: str) -> None:
        """Blocca un oggetto nella posa attuale finche' non viene rilasciato.

        Serve quando una posa e' imposta dall'esterno e non deve essere
        corretta dalla fisica, per esempio un target interrato in un terreno
        che in simulazione e' rigido: gli altri oggetti possono cadergli
        addosso e assestarsi, lui resta dov'e'.
        """
        state = self.get_object_state(instance_id)
        self._held[instance_id] = (
            np.asarray(state.position, dtype=float),
            np.asarray(state.quaternion, dtype=float),
        )

    def release_object(self, instance_id: str) -> None:
        self._held.pop(instance_id, None)

    def _hold_removed_objects(self) -> None:
        """Tiene fermi gli oggetti rimossi al punto di parcheggio.

        Non avendo piu' contatti, la loro dinamica e' completamente separata
        dal resto della scena: riscriverne lo stato a ogni passo non altera la
        fisica degli oggetti ancora presenti, ed evita che cadano all'infinito
        accumulando velocita'. (`body_gravcomp` non compensa la gravita' in
        questo caso, verificato sperimentalmente.)
        """
        for instance_id in self._removed:
            self._park(instance_id)

    def step_for(
        self,
        duration: float,
        capture_camera: str | None = None,
        capture_stride: int = 0,
        on_step=None,
    ) -> list[np.ndarray]:
        """Avanza la simulazione per `duration` secondi.

        Con `capture_camera` viene renderizzato un frame ogni `capture_stride`
        passi e la lista dei frame viene restituita.

        `on_step` viene chiamata con la stessa cadenza. Serve a chi disegna da
        solo, per esempio il viewer interattivo: senza, si vedrebbe solo lo
        stato finale di ogni assestamento, cioe' dei salti invece del
        movimento.
        """
        step_count = int(np.ceil(duration / self.model.opt.timestep))
        frames: list[np.ndarray] = []
        stride = max(int(capture_stride), 1)
        capturing = capture_camera is not None and capture_stride > 0

        for index in range(step_count):
            self.step()
            if index % stride:
                continue
            if capturing:
                frames.append(self.render_camera(capture_camera))
            if on_step is not None:
                on_step()

        return frames

    def step_until_settled(
        self,
        capture_camera: str | None = None,
        capture_stride: int = 0,
        on_step=None,
    ) -> SettlingOutcome:
        """Avanza finche' la scena non e' ferma, o finche' scade il timeout.

        Avanzare per una durata FISSA e' sbagliato: il tempo di caduta dipende
        dall'altezza di rilascio, e leggere le pose mentre gli oggetti si
        muovono ancora fa misurare come "disturbo" del moto che nessuno ha
        causato. Qui si guarda invece la condizione che interessa davvero —
        che tutto sia fermo — con le soglie dichiarate nella scena.

        Contano solo gli oggetti PRESENTI: quelli rimossi sono parcheggiati e
        tenuti fermi a ogni passo, quindi sarebbero sempre "stabili" e non
        aggiungono informazione.
        """
        settings = self.scene.simulation
        required_stable_steps = math.ceil(
            settings.stable_duration / settings.timestep
        )
        maximum_steps = math.ceil(settings.timeout / settings.timestep)
        stride = max(int(capture_stride), 1)
        capturing = capture_camera is not None and capture_stride > 0

        frames: list[np.ndarray] = []
        reference = self._pose_snapshot()
        stable_steps = 0
        step_number = 0

        while step_number < maximum_steps:
            self.step()
            step_number += 1

            if step_number % stride == 0:
                if capturing:
                    frames.append(self.render_camera(capture_camera))
                if on_step is not None:
                    on_step()

            current = self._pose_snapshot()
            if self._poses_coincide(reference, current):
                stable_steps += 1
                if stable_steps >= required_stable_steps:
                    break
            else:
                # La finestra riparte da QUI: e' cio' che rende il criterio
                # uno "non si e' mosso nell'ultimo mezzo secondo" invece di un
                # confronto con una posa vecchia e ormai irrilevante.
                reference = current
                stable_steps = 0

        return SettlingOutcome(
            settled=stable_steps >= required_stable_steps,
            steps=step_number,
            duration=step_number * settings.timestep,
            frames=frames,
        )

    def _pose_snapshot(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            instance_id: (
                np.asarray(state.position, dtype=float),
                np.asarray(state.quaternion, dtype=float),
            )
            for instance_id, state in self.get_present_object_states().items()
        }

    def _poses_coincide(
        self,
        reference: dict[str, tuple[np.ndarray, np.ndarray]],
        current: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> bool:
        """True se nessun oggetto si e' mosso oltre le tolleranze di quiete.

        Il confronto e' sulle POSE, non sulle velocita'. Un oggetto appoggiato
        riceve dai contatti picchi di velocita' angolare che durano pochi
        passi e non lo spostano: con una soglia sulla velocita' quei picchi
        azzerano continuamente il conteggio, e una scena ferma risulta "mai
        assestata" fino allo scadere del timeout. Guardando lo spostamento il
        rumore si media da solo, e resta solo il moto vero.
        """
        settings = self.scene.simulation
        for instance_id, (position, quaternion) in current.items():
            previous = reference.get(instance_id)
            if previous is None:
                # Un oggetto comparso o riapparso durante la finestra: la
                # finestra non e' piu' confrontabile.
                return False
            if np.linalg.norm(position - previous[0]) >= (
                settings.linear_settle_tolerance
            ):
                return False
            if _rotation_angle(previous[1], quaternion) >= (
                settings.angular_settle_tolerance
            ):
                return False
        return len(current) == len(reference)

    def snapshot(self) -> dict:
        """Fotografia dello stato: pose, velocita', tempo e rimozioni.

        Serve a ripartire da una configurazione gia' assestata senza doverla
        ricostruire. Costruire una scena costa quasi un secondo — la caduta e
        l'assestamento di ogni oggetto — mentre ripristinarla costa un
        millesimo, e per chi deve provare molte volte la stessa scena (una
        ricerca sugli ordini, un allenamento) e' la differenza fra ore e
        minuti.

        Vale **solo per questo modello**: il vettore delle pose ha una
        lunghezza che dipende da quanti corpi ci sono. Ripristinarlo su un
        modello diverso non e' un errore che MuJoCo segnala, quindi lo
        segnaliamo noi.
        """
        return {
            "scene_id": self.scene.scene_id,
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "time": float(self.data.time),
            "contype": self.model.geom_contype.copy(),
            "conaffinity": self.model.geom_conaffinity.copy(),
            "removed": set(self._removed),
            "rgba": self.model.geom_rgba.copy(),
        }

    def restore(self, snapshot: dict) -> None:
        """Riporta il simulatore allo stato di `snapshot`."""
        if snapshot["scene_id"] != self.scene.scene_id:
            raise ValueError(
                "La fotografia viene da un'altra scena "
                f"({snapshot['scene_id']!r} invece di {self.scene.scene_id!r}): "
                "le pose non descrivono questi corpi."
            )
        self.data.qpos[:] = snapshot["qpos"]
        self.data.qvel[:] = snapshot["qvel"]
        self.data.time = snapshot["time"]
        self.model.geom_contype[:] = snapshot["contype"]
        self.model.geom_conaffinity[:] = snapshot["conaffinity"]
        self._removed = set(snapshot["removed"])
        if "rgba" in snapshot:
            self.model.geom_rgba[:] = snapshot["rgba"]
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        """Riporta il modello allo stato iniziale, rimozioni comprese."""
        self.model.geom_contype[:] = self._initial_contype
        self.model.geom_conaffinity[:] = self._initial_conaffinity
        self.model.geom_rgba[:] = self._initial_rgba
        self._removed.clear()

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._initial_qpos
        mujoco.mj_forward(self.model, self.data)

    def reset_to_poses(self, poses) -> None:
        """Reset con nuove pose iniziali, senza ricompilare il modello.

        `poses` e' una sequenza di Pose nello stesso ordine di
        `scene.objects`. Il modello resta lo stesso oggetto Python, quindi un
        viewer gia' agganciato continua a funzionare.
        """
        self.reset()
        for item, pose in zip(self.scene.objects, poses, strict=True):
            joint_id = self._joint_ids[item.instance_id]
            address = self.model.jnt_qposadr[joint_id]
            self.data.qpos[address : address + 3] = pose.position
            self.data.qpos[address + 3 : address + 7] = pose.quaternion
        mujoco.mj_forward(self.model, self.data)

    # -------------------------------------------------------------- rimozione

    @property
    def removed_objects(self) -> frozenset[str]:
        return frozenset(self._removed)

    def present_objects(self) -> tuple[str, ...]:
        return tuple(
            item.instance_id
            for item in self.scene.objects
            if item.instance_id not in self._removed
        )

    def is_present(self, instance_id: str) -> bool:
        return instance_id not in self._removed

    def remove_object(self, instance_id: str) -> None:
        """Toglie un oggetto dalla scena senza ricompilare il modello.

        MuJoCo non permette di cancellare un corpo da un modello compilato.
        Invece di ricostruire il modello a ogni rimozione (costoso, e
        cambierebbe la dimensione degli spazi di osservazione e azione a
        meta' episodio) l'oggetto viene:

          1. escluso dai contatti (contype/conaffinity a zero);
          2. reso invisibile (alpha a zero), cosi' sparisce anche dalle camere;
          3. spostato lontano e tenuto fermo li' a ogni passo.

        Il risultato osservabile e' identico alla cancellazione, ma gli spazi
        dell'environment restano di dimensione costante.
        """
        if instance_id not in self._body_ids:
            raise KeyError(f"Oggetto sconosciuto: {instance_id}")
        if instance_id in self._removed:
            raise ValueError(f"Oggetto gia' rimosso: {instance_id}")

        body_id = self._body_ids[instance_id]
        first_geom = self.model.body_geomadr[body_id]
        geom_count = self.model.body_geomnum[body_id]
        for geom_id in range(first_geom, first_geom + geom_count):
            self.model.geom_contype[geom_id] = 0
            self.model.geom_conaffinity[geom_id] = 0
            self.model.geom_rgba[geom_id, 3] = 0.0

        self._removed.add(instance_id)
        self._park(instance_id)
        mujoco.mj_forward(self.model, self.data)

    def restore_object(self, instance_id: str, pose=None) -> None:
        """Rimette in scena un oggetto rimosso, opzionalmente in una nuova posa.

        E' l'inverso esatto di `remove_object`. Serve al **rilascio
        sequenziale**: gli oggetti partono tutti fuori scena e vengono
        immessi uno alla volta, cosi' non possono mai partire compenetrati e
        ciascuno cade da poco sopra la pila invece che da metri di quota.
        """
        if instance_id not in self._body_ids:
            raise KeyError(f"Oggetto sconosciuto: {instance_id}")
        if instance_id not in self._removed:
            raise ValueError(f"Oggetto gia' in scena: {instance_id}")

        body_id = self._body_ids[instance_id]
        first_geom = self.model.body_geomadr[body_id]
        geom_count = self.model.body_geomnum[body_id]
        for geom_id in range(first_geom, first_geom + geom_count):
            self.model.geom_contype[geom_id] = self._initial_contype[geom_id]
            self.model.geom_conaffinity[geom_id] = self._initial_conaffinity[geom_id]
            self.model.geom_rgba[geom_id, 3] = self._initial_rgba[geom_id, 3]

        self._removed.discard(instance_id)
        if pose is not None:
            self.set_object_pose(instance_id, pose.position, pose.quaternion)
        mujoco.mj_forward(self.model, self.data)

    def set_object_pose(self, instance_id: str, position, quaternion) -> None:
        """Teletrasporta un oggetto, azzerandone la velocita'."""
        self._write_pose(instance_id, position, quaternion)
        # qpos e le pose cartesiane dei geom sono cache separate in MuJoCo.
        # Senza il forward, raycast e renderer continuano a vedere la posa
        # precedente fino al passo fisico successivo.
        mujoco.mj_forward(self.model, self.data)

    def _write_pose(self, instance_id: str, position, quaternion) -> None:
        joint_id = self._joint_ids[instance_id]
        qpos_address = self.model.jnt_qposadr[joint_id]
        dof_address = self.model.jnt_dofadr[joint_id]
        self.data.qpos[qpos_address : qpos_address + 3] = position
        self.data.qpos[qpos_address + 3 : qpos_address + 7] = quaternion
        self.data.qvel[dof_address : dof_address + 6] = 0.0

    def contact_pairs(self) -> list[tuple[str, str]]:
        """Coppie di corpi attualmente a contatto, senza duplicati.

        `data.contact` viene riempito da MuJoCo a ogni passo: sono i contatti
        ATTIVI adesso. Su una scena assestata sono gli appoggi reali.
        """
        seen = set()
        pairs = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first = self._body_name(contact.geom1)
            second = self._body_name(contact.geom2)
            key = tuple(sorted((first, second)))
            if key not in seen:
                seen.add(key)
                pairs.append(key)
        return pairs

    def _body_name(self, geom_id: int) -> str:
        body_id = int(self.model.geom_bodyid[geom_id])
        return (
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "?"
        )

    def support_graph(self) -> dict[str, list[str]]:
        """`{oggetto: [cosa lo sostiene]}`, con "terreno" per il pavimento.

        Fra due corpi a contatto si assume che il piu' basso sostenga il piu'
        alto. E' un'approssimazione — la normale del contatto sarebbe piu'
        precisa — ma per oggetti appoggiati sotto gravita' e' fedele.
        """
        supports: dict[str, list[str]] = {}
        for first, second in self.contact_pairs():
            if GROUND_BODY in (first, second):
                item = second if first == GROUND_BODY else first
                supports.setdefault(item, []).append("terreno")
                continue
            if first not in self._body_ids or second not in self._body_ids:
                continue
            lower, upper = sorted(
                (first, second),
                key=lambda name: self.get_object_state(name).position[2],
            )
            supports.setdefault(upper, []).append(lower)
        return supports

    def supported_by(self, instance_id: str) -> list[str]:
        """Su cosa poggia un singolo oggetto."""
        return self.support_graph().get(instance_id, [])

    def pile_summit(self) -> tuple[float, float, float]:
        """Il punto piu' alto della pila: `(x, y, z)`. `(0, 0, 0)` se e' vuota.

        E' il bersaglio su cui rilasciare l'oggetto successivo. Servono anche
        x e y, non solo la quota: lasciando cadere sempre sulla stessa colonna
        nominale gli oggetti si affiancano invece di impilarsi, perche' la
        pila si sposta man mano che cresce. Mirare alla sua cima e' cio' che
        produce una pila invece di un mucchietto sparso.

        Si somma il raggio d'ingombro perche' `position` e' il centro del
        corpo, e un oggetto sporge sopra il proprio centro.
        """
        summit = (0.0, 0.0, 0.0)
        for item in self.scene.objects:
            if item.instance_id in self._removed:
                continue
            centre = self.get_object_state(item.instance_id).position
            top = centre[2] + bounding_radius(item.shape, item.size)
            if top > summit[2]:
                summit = (centre[0], centre[1], top)
        return summit

    def _park(self, instance_id: str) -> None:
        joint_id = self._joint_ids[instance_id]
        qpos_address = self.model.jnt_qposadr[joint_id]
        dof_address = self.model.jnt_dofadr[joint_id]
        self.data.qpos[qpos_address : qpos_address + 3] = PARKING_POSITION
        self.data.qpos[qpos_address + 3 : qpos_address + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[dof_address : dof_address + 6] = 0.0

    def set_object_color(self, instance_id: str, rgba) -> None:
        """Cambia il colore di un oggetto (solo aspetto, nessun effetto fisico).

        Serve a evidenziare il target nelle visualizzazioni. Attenzione: le
        camere vedono il colore cambiato, quindi va usato solo quando le
        immagini NON sono osservazioni dell'agente.
        """
        body_id = self._body_ids[instance_id]
        first_geom = self.model.body_geomadr[body_id]
        geom_count = self.model.body_geomnum[body_id]
        for geom_id in range(first_geom, first_geom + geom_count):
            self.model.geom_rgba[geom_id] = rgba

    # ------------------------------------------------------------------ stato

    def get_object_state(self, instance_id: str) -> ObjectState:
        joint_id = self._joint_ids[instance_id]
        qpos_address = self.model.jnt_qposadr[joint_id]
        dof_address = self.model.jnt_dofadr[joint_id]
        qpos = self.data.qpos[qpos_address : qpos_address + 7]
        qvel = self.data.qvel[dof_address : dof_address + 6]

        return ObjectState(
            position=tuple(float(value) for value in qpos[:3]),
            quaternion=tuple(float(value) for value in qpos[3:7]),
            linear_velocity=tuple(float(value) for value in qvel[:3]),
            angular_velocity=tuple(float(value) for value in qvel[3:6]),
        )

    def get_object_states(self) -> dict[str, ObjectState]:
        """Stato di TUTTI gli oggetti, rimossi compresi (parcheggiati)."""
        return {
            item.instance_id: self.get_object_state(item.instance_id)
            for item in self.scene.objects
        }

    def get_present_object_states(self) -> dict[str, ObjectState]:
        return {
            instance_id: self.get_object_state(instance_id)
            for instance_id in self.present_objects()
        }

    def object_projection_bounds(
        self, instance_id: str, axis
    ) -> tuple[float, float]:
        """Limiti dell'AABB MuJoCo dell'oggetto proiettati su un asse mondo."""
        if instance_id not in self._body_ids:
            raise KeyError(f"Oggetto sconosciuto: {instance_id}")
        direction = np.asarray(axis, dtype=float)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(direction).all() or norm <= 0:
            raise ValueError("L'asse deve essere un vettore finito non nullo")
        direction /= norm
        mujoco.mj_forward(self.model, self.data)
        body_id = self._body_ids[instance_id]
        first = int(self.model.body_geomadr[body_id])
        count = int(self.model.body_geomnum[body_id])
        bounds = []
        for geom_id in range(first, first + count):
            aabb = np.asarray(self.model.geom_aabb[geom_id], dtype=float)
            rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
            center = self.data.geom_xpos[geom_id] + rotation @ aabb[:3]
            radius = float(np.sum(np.abs(direction @ rotation) * aabb[3:]))
            projection = float(direction @ center)
            bounds.append((projection - radius, projection + radius))
        if not bounds:
            raise ValueError(f"L'oggetto {instance_id} non contiene geom")
        return min(item[0] for item in bounds), max(item[1] for item in bounds)

    # -------------------------------------------------------------- rendering

    def render_camera(self, camera_name: str) -> np.ndarray:
        """Un frame RGB (H, W, 3) uint8 da una camera definita nell'MJCF."""
        renderer = self._get_renderer()
        renderer.update_scene(self.data, camera=camera_name)
        return renderer.render()

    def render_stereo(self) -> tuple[np.ndarray, np.ndarray]:
        """La coppia (sinistra, destra) dal rig stereo della scena."""
        if self.scene.stereo_camera is None:
            raise RuntimeError(
                "La scena non ha un rig stereo: passa stereo_config a "
                "generate_scene()"
            )
        return self.render_camera("cam_left"), self.render_camera("cam_right")

    def render_depth_camera(self, camera_name: str) -> np.ndarray:
        """Profondita' metrica del renderer; canale RGB-D distinto dalla stereo."""
        renderer = self._get_renderer()
        renderer.enable_depth_rendering()
        try:
            renderer.update_scene(self.data, camera=camera_name)
            return np.asarray(renderer.render(), dtype=float).copy()
        finally:
            renderer.disable_depth_rendering()

    def render_depth_stereo(self) -> tuple[np.ndarray, np.ndarray]:
        if self.scene.stereo_camera is None:
            raise RuntimeError("La scena non ha un rig stereo")
        return self.render_depth_camera("cam_left"), self.render_depth_camera("cam_right")

    def raycast(self, origin, direction) -> float | None:
        """Distanza al primo hit; nessun ID privilegiato esce dal simulatore."""
        p = np.asarray(origin, dtype=float)
        d = np.asarray(direction, dtype=float)
        if p.shape != (3,) or d.shape != (3,) or not np.isfinite(p).all() or not np.isfinite(d).all():
            raise ValueError("Origine o direzione del raggio non valide")
        length = np.linalg.norm(d)
        if length <= 0:
            raise ValueError("Direzione del raggio nulla")
        distance = mujoco.mj_ray(self.model, self.data, p, d / length, None, True, -1, None)
        return None if distance < 0 else float(distance)

    def render_instance_masks(self, camera_name: str) -> dict[str, np.ndarray]:
        """Maschere visibili per istanza dal renderer di segmentazione MuJoCo.

        Gli ID sono etichette ideali di simulazione. Questa API serve a creare
        dati sintetici e a valutare OSSERVA, non a simulare un detector RGB.
        """
        renderer = self._get_renderer()
        renderer.enable_segmentation_rendering()
        try:
            renderer.update_scene(self.data, camera=camera_name)
            segmentation = renderer.render()
        finally:
            renderer.disable_segmentation_rendering()

        geom_ids = segmentation[:, :, 0]
        geom_pixels = segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM.value
        body_to_instance = {body: name for name, body in self._body_ids.items()}
        masks: dict[str, np.ndarray] = {}
        for geom_id in np.unique(geom_ids[geom_pixels]):
            body_id = int(self.model.geom_bodyid[int(geom_id)])
            instance_id = body_to_instance.get(body_id)
            if instance_id is None or instance_id in self._removed:
                continue
            if instance_id not in masks:
                masks[instance_id] = np.zeros(geom_ids.shape, dtype=bool)
            masks[instance_id] |= geom_pixels & (geom_ids == geom_id)
        return masks

    def stereo_calibration(self) -> tuple[np.ndarray, np.ndarray, float]:
        """K, world_from_left (frame camera CV) e baseline del rig simulato."""
        rig = self.scene.stereo_camera
        if rig is None:
            raise RuntimeError("La scena non ha un rig stereo")
        camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_left"
        )
        # Il fov si legge dal modello e non dalla descrizione: la domain
        # randomization puo' cambiarlo, e una calibrazione ferma al valore
        # nominale darebbe punti 3D sbagliati. Senza randomizzazione i due
        # valori coincidono.
        fovy = float(self.model.cam_fovy[camera_id])
        focal = (rig.height / 2.0) / math.tan(math.radians(fovy) / 2.0)
        intrinsics = np.array(
            [[focal, 0.0, rig.width / 2.0],
             [0.0, focal, rig.height / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=float,
        )
        world_from_left = np.eye(4, dtype=float)
        # MuJoCo: x destra, y alto, -z avanti. CV: x destra, y basso, z avanti.
        world_from_left[:3, :3] = (
            self.data.cam_xmat[camera_id].reshape(3, 3)
            @ np.diag([1.0, -1.0, -1.0])
        )
        world_from_left[:3, 3] = self.data.cam_xpos[camera_id]
        return intrinsics, world_from_left, rig.baseline

    # ------------------------------------------------------ aspetto visivo

    def apply_visual_conditions(self, conditions: VisualConditions) -> None:
        """Cambia camera, luci, terreno e colori senza toccare la fisica.

        Serve alla domain randomization del visore: pose e contatti restano
        quelli della scena, cambia solo come la si vede. Le maschere di
        segmentazione e la calibrazione seguono la camera spostata. Un campo
        `None` delle condizioni lascia quell'aspetto com'e'.
        """
        if not isinstance(conditions, VisualConditions):
            raise TypeError("Servono VisualConditions")
        if conditions.camera is not None:
            self._apply_camera(conditions.camera)
        if conditions.lighting is not None:
            self._apply_lighting(conditions.lighting)
        if conditions.ground is not None:
            self._apply_ground(conditions.ground)
        if conditions.background is not None:
            self._apply_background(conditions.background)
        for instance_id, rgb in conditions.object_rgb.items():
            self._tint_object(instance_id, rgb)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def _apply_camera(self, perturbation: CameraPerturbation) -> None:
        rig = self.scene.stereo_camera
        if rig is None:
            raise RuntimeError("La scena non ha un rig stereo da spostare")
        eye = np.asarray(rig.position, dtype=float) + perturbation.position_offset
        target = np.asarray(rig.target, dtype=float) + perturbation.target_offset
        right, up = _camera_axes(eye, target)
        roll = math.radians(perturbation.roll_deg)
        right, up = (
            math.cos(roll) * right + math.sin(roll) * up,
            -math.sin(roll) * right + math.cos(roll) * up,
        )
        # Colonne: assi x, y, z del frame camera MuJoCo (z = -avanti).
        rotation = np.column_stack((right, up, np.cross(right, up)))
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, rotation.flatten())
        fovy = rig.fovy if perturbation.fovy_deg is None else perturbation.fovy_deg
        half = rig.baseline / 2.0
        for name, side in STEREO_CAMERAS:
            camera_id = self._named_id(mujoco.mjtObj.mjOBJ_CAMERA, name)
            self.model.cam_pos[camera_id] = eye + right * side * half
            self.model.cam_quat[camera_id] = quaternion
            self.model.cam_fovy[camera_id] = fovy

    def _apply_lighting(self, lighting: LightingConditions) -> None:
        light_id = self._named_id(mujoco.mjtObj.mjOBJ_LIGHT, MAIN_LIGHT)
        direction = np.asarray(lighting.direction, dtype=float)
        self.model.light_dir[light_id] = direction / np.linalg.norm(direction)
        self.model.light_diffuse[light_id] = (lighting.diffuse,) * 3
        self.model.light_castshadow[light_id] = bool(lighting.cast_shadow)
        self.model.vis.headlight.diffuse[:] = lighting.headlight
        self.model.vis.headlight.ambient[:] = lighting.ambient

    def _apply_ground(self, ground: GroundAppearance) -> None:
        geom_id = self._named_id(mujoco.mjtObj.mjOBJ_GEOM, GROUND_GEOM)
        material_id = self._named_id(mujoco.mjtObj.mjOBJ_MATERIAL, GROUND_MATERIAL)
        nominal = self._visual_nominal.setdefault(
            "ground",
            {
                "matid": int(self.model.geom_matid[geom_id]),
                "rgba": self.model.geom_rgba[geom_id].copy(),
                "reflectance": float(self.model.mat_reflectance[material_id]),
            },
        )
        if ground.kind == "flat":
            self.model.geom_matid[geom_id] = -1
            rgba = nominal["rgba"].copy()
            rgba[:3] = ground.base_gray
            self.model.geom_rgba[geom_id] = rgba
            self._initial_rgba[geom_id] = rgba
            self._write_texture(GROUND_TEXTURE, None)
        else:
            self.model.geom_matid[geom_id] = nominal["matid"]
            self.model.geom_rgba[geom_id] = nominal["rgba"]
            self._initial_rgba[geom_id] = nominal["rgba"]
            self._write_texture(
                GROUND_TEXTURE,
                None
                if ground.kind == "checker"
                else (ground.base_gray, ground.contrast, ground.grain_px, ground.seed),
            )
        self.model.mat_reflectance[material_id] = (
            nominal["reflectance"] if ground.reflectance is None else ground.reflectance
        )

    def _apply_background(self, background: BackgroundAppearance) -> None:
        # Grana ampia: lo sfondo deve variare, non diventare un altro pattern.
        self._write_texture(
            BACKGROUND_TEXTURE,
            (background.base_gray, background.contrast, 256, background.seed),
        )

    def _write_texture(self, name: str, noise) -> None:
        """Riscrive una texture: `None` ripristina l'originale, altrimenti rumore.

        `noise` = (grigio medio, contrasto, grana in pixel, seed).
        """
        texture_id = self._named_id(mujoco.mjtObj.mjOBJ_TEXTURE, name)
        start = int(self.model.tex_adr[texture_id])
        height = int(self.model.tex_height[texture_id])
        width = int(self.model.tex_width[texture_id])
        channels = int(self.model.tex_nchannel[texture_id])
        end = start + height * width * channels
        original = self._visual_nominal.setdefault(
            f"texture:{name}", self.model.tex_data[start:end].copy()
        )
        self.model.tex_data[start:end] = (
            original if noise is None else _noise_texture(*noise, height, width, channels)
        )
        # Un renderer gia' creato ha la texture nella memoria della GPU: va
        # ricaricata. Se non esiste ancora, la prendera' dal modello aggiornato.
        if self._renderer is not None:
            mujoco.mjr_uploadTexture(
                self.model, self._renderer._mjr_context, texture_id
            )

    def _tint_object(self, instance_id: str, rgb) -> None:
        if instance_id not in self._body_ids:
            raise KeyError(f"Oggetto sconosciuto: {instance_id}")
        body_id = self._body_ids[instance_id]
        first_geom = self.model.body_geomadr[body_id]
        geom_count = self.model.body_geomnum[body_id]
        for geom_id in range(first_geom, first_geom + geom_count):
            # Solo RGB: l'alpha resta quello che codifica "rimosso/presente".
            self.model.geom_rgba[geom_id, :3] = rgb
            self._initial_rgba[geom_id, :3] = rgb

    def _named_id(self, kind, name: str) -> int:
        identifier = mujoco.mj_name2id(self.model, kind, name)
        if identifier < 0:
            raise KeyError(f"Elemento MJCF assente: {name}")
        return identifier

    def _get_renderer(self) -> mujoco.Renderer:
        if self._renderer is None:
            rig = self.scene.stereo_camera
            height = rig.height if rig is not None else 480
            width = rig.width if rig is not None else 640
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        return self._renderer

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def _rotation_angle(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Angolo, in radianti, della rotazione che porta `first` su `second`.

    Per due quaternioni unitari vale |cos(theta/2)| = |<q1, q2>|. Il valore
    assoluto serve perche' q e -q sono la stessa rotazione: senza, una
    rotazione nulla potrebbe risultare di 2*pi.
    """
    cosine = abs(float(np.dot(first, second)))
    return 2.0 * math.acos(min(1.0, cosine))


def _noise_texture(
    base_gray: float,
    contrast: float,
    grain_px: int,
    seed: int,
    height: int,
    width: int,
    channels: int,
) -> np.ndarray:
    """Texture a macchie in scala di grigio, riproducibile dal seed.

    Un rumore a bassa risoluzione ingrandito con interpolazione lineare da'
    macchie morbide di circa `grain_px` pixel: abbastanza da togliere al
    detector l'appiglio del pavimento a scacchi, senza pretendere di imitare
    un terreno reale specifico.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.uniform(
        -1.0,
        1.0,
        (max(2, math.ceil(height / grain_px) + 1), max(2, math.ceil(width / grain_px) + 1)),
    )
    rows = np.linspace(0.0, coarse.shape[0] - 1.0, height)
    columns = np.linspace(0.0, coarse.shape[1] - 1.0, width)
    # Interpolazione bilineare separabile, senza dipendere da OpenCV.
    row_index = np.arange(coarse.shape[0])
    column_index = np.arange(coarse.shape[1])
    along_rows = np.stack(
        [np.interp(rows, row_index, coarse[:, j]) for j in column_index], axis=1
    )
    field = np.stack(
        [np.interp(columns, column_index, along_rows[i]) for i in range(height)], axis=0
    )
    gray = np.clip(base_gray + 0.5 * contrast * field, 0.0, 1.0)
    values = np.rint(gray * 255.0).astype(np.uint8)
    return np.repeat(values[:, :, None], channels, axis=2).ravel()
