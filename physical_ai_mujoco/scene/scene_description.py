from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


def bounding_radius(shape: str, size: dict[str, float]) -> float:
    """Raggio della sfera che contiene l'oggetto, comunque sia orientato.

    E' la stima conservativa dell'ingombro: non dipende dall'orientamento,
    quindi vale anche per un oggetto che ruota mentre cade.
    """
    if shape in {"box", "mesh"}:
        return math.sqrt(size["x"] ** 2 + size["y"] ** 2 + size["z"] ** 2) / 2.0
    if shape == "cylinder":
        return math.sqrt(size["radius"] ** 2 + (size["height"] / 2.0) ** 2)
    if shape == "sphere":
        return size["radius"]
    raise ValueError(f"Unsupported shape: {shape}")


def half_extents(shape: str, size: dict[str, float]) -> tuple[float, float, float]:
    """Semi-dimensioni lungo gli assi del corpo."""
    if shape in {"box", "mesh"}:
        return (size["x"] / 2.0, size["y"] / 2.0, size["z"] / 2.0)
    if shape == "cylinder":
        return (size["radius"], size["radius"], size["height"] / 2.0)
    if shape == "sphere":
        return (size["radius"],) * 3
    raise ValueError(f"Unsupported shape: {shape}")


def principal_inertia(
    shape: str, size: dict[str, float], mass: float
) -> tuple[float, float, float]:
    """Momenti principali d'inerzia di un solido UNIFORME, attorno al suo centro.

    Sono le stesse formule che il compilatore MuJoCo applica quando un geom
    dichiara la propria massa — verificato: coincidono a meno di 1e-10. Servono
    scritte qui perche' spostare il centro di massa obbliga a dichiarare
    esplicitamente l'inerzia nell'MJCF, e a quel punto il valore deve venire
    da qualche parte.

    L'ipotesi e' che gli assi principali coincidano con gli assi del corpo, che
    per box, cilindro e sfera e' vero.
    """
    if shape in {"box", "mesh"}:
        x, y, z = size["x"], size["y"], size["z"]
        return (
            mass * (y * y + z * z) / 12.0,
            mass * (x * x + z * z) / 12.0,
            mass * (x * x + y * y) / 12.0,
        )
    if shape == "cylinder":
        radius, height = size["radius"], size["height"]
        transverse = mass * (3.0 * radius * radius + height * height) / 12.0
        return (transverse, transverse, mass * radius * radius / 2.0)
    if shape == "sphere":
        value = 2.0 * mass * size["radius"] ** 2 / 5.0
        return (value, value, value)
    raise ValueError(f"Unsupported shape: {shape}")


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]


@dataclass(frozen=True)
class ObjectDescription:
    instance_id: str
    type_id: str
    shape: str
    size: dict[str, float]
    density: float
    mass: float
    friction: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    pose: Pose
    # Scostamento del centro di massa dal centro geometrico, in metri, negli
    # assi del corpo. `(0, 0, 0)` e' il solido uniforme. Un oggetto reale non
    # ha quasi mai la massa distribuita in modo uniforme, e per un compito che
    # dipende da cosa si ribalta e cosa no la differenza non e' un dettaglio.
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Solo per shape="mesh". Il file viene risolto dal loader rispetto al
    # dataset, mentre la scala converte le unita' del file in metri MuJoCo.
    mesh_file: str | None = None
    mesh_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @property
    def has_offset_mass(self) -> bool:
        return any(value != 0.0 for value in self.center_of_mass)


@dataclass(frozen=True)
class GroundDescription:
    """Il piano di appoggio della scena.

    Due forme, con semantiche diverse e non intercambiabili:

    - ``"box"``   : un tavolo. Ha spessore e **bordi**: `size` e' l'ingombro
                    reale, e un oggetto che scivola oltre il bordo cade nel
                    vuoto. La superficie e' a z=0, quindi il corpo sta a
                    -size.z/2.
    - ``"plane"`` : un pavimento. In MuJoCo un plane e' **infinito per le
                    collisioni**: `size` riguarda solo il disegno (meta'
                    estensione x, meta' estensione y, passo della griglia).
                    Non ci sono bordi, quindi l'area di rilascio la
                    decidono soltanto le regole di scena. La superficie e'
                    l'origine del corpo, a z=0.
    """

    type_id: str
    shape: str
    size: dict[str, float]
    friction: tuple[float, float, float]
    roughness_type: str
    rgba: tuple[float, float, float, float]
    pose: Pose

    @property
    def is_bounded(self) -> bool:
        """True se il terreno ha bordi da cui un oggetto puo' cadere."""
        return self.shape != "plane"

    @property
    def surface_thickness(self) -> float:
        """Spessore fra l'origine del corpo e la superficie di appoggio."""
        return 0.0 if self.shape == "plane" else self.size["z"] / 2.0


@dataclass(frozen=True)
class SimulationDescription:
    gravity: tuple[float, float, float]
    timestep: float
    integrator: str
    # Quanto un oggetto puo' spostarsi e ruotare, nella finestra
    # `stable_duration`, e continuare a contare come fermo. La quiete si
    # giudica sullo SPOSTAMENTO e non sulla velocita' istantanea: il contatto
    # produce picchi di velocita' angolare di frazioni di millisecondo su
    # oggetti che restano dove sono, e una soglia sulla velocita' non riesce a
    # distinguerli da una rotazione vera (misurato: sugli assestamenti
    # dichiarati falliti gli oggetti si spostavano 0,14 mm al secondo).
    linear_settle_tolerance: float
    angular_settle_tolerance: float
    stable_duration: float
    timeout: float
    # Rapporto fra la rigidezza del vincolo d'attrito e quella del vincolo
    # normale, per il cono ellittico. Col default 1 l'attrito statico e'
    # "morbido" e gli oggetti appoggiati strisciano lentamente per secondi.
    impratio: float = 1.0
    # "pyramidal" e' l'approssimazione veloce del cono d'attrito, "elliptic"
    # quella esatta. La seconda costa di piu' per passo ma, non sbagliando la
    # direzione della forza d'attrito, fa assestare prima. `impratio` ha
    # effetto solo con il cono ellittico.
    cone: str = "pyramidal"
    # Passata PGS aggiuntiva sulle sole direzioni d'attrito, a vincoli rigidi,
    # dopo che il solutore principale ha converso. E' il rimedio documentato
    # allo scivolamento residuo dei vincoli morbidi. La documentazione avverte
    # che puo' destabilizzare i contatti multipli: qui e' stato misurato, non
    # supposto (vedi MODIFICHE.md).
    noslip_iterations: int = 0
    # Costante di tempo del contatto, in secondi (il primo valore di `solref`;
    # lo smorzamento resta critico). E' quanto il contatto e' "molle": due
    # corpi appoggiati si compenetrano di una quantita' che cresce con il suo
    # quadrato. Il minimo consigliato dalla documentazione MuJoCo e' due passi
    # di integrazione; qui se ne usano cinque, che e' il punto in cui la
    # compenetrazione e' scesa a 0,2 mm senza perdere stabilita'.
    contact_timeconst: float = 0.02


@dataclass(frozen=True)
class StereoCameraDescription:
    """Rig stereo a camere parallele (rettificate).

    Le due camere condividono l'orientamento e sono separate di `baseline`
    lungo l'asse orizzontale del rig. Con camere parallele vale la relazione
    classica profondita' = focal_length_px * baseline / disparita', dove
    focal_length_px e' dato da `focal_length_px()`.
    """

    baseline: float
    width: int
    height: int
    fovy: float
    position: tuple[float, float, float]
    target: tuple[float, float, float]

    def focal_length_px(self) -> float:
        return (self.height / 2.0) / math.tan(math.radians(self.fovy) / 2.0)


@dataclass(frozen=True)
class SceneDescription:
    schema_version: int
    scene_id: str
    seed: int
    ground: GroundDescription
    objects: tuple[ObjectDescription, ...]
    simulation: SimulationDescription
    # Opzionale: assente nelle scene di Phase 0A, che restano invariate.
    stereo_camera: StereoCameraDescription | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )
