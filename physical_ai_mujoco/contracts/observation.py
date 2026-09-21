"""Prodotti dati del ramo operativo di OSSERVA.

I campi opzionali rappresentano dati non osservati. Nessun prodotto di questo
modulo contiene informazioni fisiche riservate al simulatore.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PerceptualObject:
    object_id: str
    type_id: str | None
    position: tuple[float, float, float] | None
    quaternion: tuple[float, float, float, float] | None
    shape: str | None
    size: tuple[float, float, float] | None
    detected: bool
    confidence: float | None
    classification_confidence: float | None = None


@dataclass(frozen=True)
class PerceptualState:
    objects: tuple[PerceptualObject, ...]
    frame: str
    timestamp: float


@dataclass(frozen=True)
class SensorEvidence:
    """Evidenza spaziale interna a OSSERVA, eventualmente parziale."""

    frame: str
    timestamp: float
    observed_fraction: float | None = None
    point_cloud: np.ndarray | None = None
    instance_ids: tuple[str, ...] = ()
    point_instance_ids: np.ndarray | None = None
    depth_map: np.ndarray | None = None
    height_map: np.ndarray | None = None
    instance_masks: dict[str, np.ndarray] | None = None
    semantic_mask: np.ndarray | None = None
    camera_intrinsics: np.ndarray | None = None
    camera_extrinsics: np.ndarray | None = None
    lidar: np.ndarray | None = None
    observed_space: np.ndarray | None = None
    rgb_left: np.ndarray | None = None
    rgb_right: np.ndarray | None = None


@dataclass(frozen=True)
class SensorBundle:
    """Ingresso stereo normalizzato con maschere associate nelle due viste.

    ``world_from_left`` trasforma coordinate camera CV (z in avanti) nel
    frame canonico. Gli ID possono essere track ID di un detector o etichette
    ideali del renderer simulato; la provenienza va dichiarata dall'adapter.
    """

    rgb_left: np.ndarray
    rgb_right: np.ndarray
    intrinsics: np.ndarray
    world_from_left: np.ndarray
    baseline: float
    timestamp: float
    frame: str
    left_masks: dict[str, np.ndarray]
    right_masks: dict[str, np.ndarray]
    type_ids: dict[str, str] | None = None
    depth_left: np.ndarray | None = None
    depth_right: np.ndarray | None = None
    detection_source: str = "unspecified"

    @property
    def gray_left(self) -> np.ndarray:
        return self.rgb_left[:, :, 0]

    @property
    def gray_right(self) -> np.ndarray:
        return self.rgb_right[:, :, 0]


@dataclass(frozen=True)
class TaskContext:
    target_id: str | None
    ground_height: float
    bounds: tuple[float, float, float, float] | None = None
    protected_zone: tuple[float, ...] | None = None
    target_safe_zone: tuple[float, ...] | None = None
    obstacle_drop_zone: tuple[float, ...] | None = None
    target_type_id: str | None = None


@dataclass(frozen=True)
class SceneObject:
    object_id: str
    role: str
    type_id: str | None
    position: tuple[float, float, float] | None
    quaternion: tuple[float, float, float, float] | None
    shape: str | None
    size: tuple[float, float, float] | None
    detected: bool
    perception_quality: float | None

    @property
    def present(self) -> bool:
        return self.detected

    @property
    def is_target(self) -> bool:
        return self.role == "target"


@dataclass(frozen=True)
class SceneState:
    objects: tuple[SceneObject, ...]
    target_id: str | None
    ground_height: float
    bounds: tuple[float, float, float, float] | None
    frame: str
    timestamp: float
    protected_zone: tuple[float, ...] | None = None
    target_safe_zone: tuple[float, ...] | None = None
    obstacle_drop_zone: tuple[float, ...] | None = None


@dataclass(frozen=True)
class PhysicalRelation:
    source_id: str
    target_id: str
    relation_type: str
    relation_score: float


@dataclass(frozen=True)
class PhysicalRelationState:
    object_ids: tuple[str, ...]
    relations: tuple[PhysicalRelation, ...]
    estimator: str
    available: bool
    frame: str
    timestamp: float

    @property
    def dependency_graph(self) -> dict[str, tuple[str, ...]]:
        """Adiacenza diretta derivata dalle relazioni; non e' un piano."""

        return {
            object_id: tuple(
                relation.target_id
                for relation in self.relations
                if relation.source_id == object_id
            )
            for object_id in self.object_ids
        }


@dataclass(frozen=True)
class ObjectUncertainty:
    object_id: str
    visible: bool
    detection_quality: float | None
    pose_quality: float | None


@dataclass(frozen=True)
class UncertaintyState:
    objects: tuple[ObjectUncertainty, ...]
    unknown_space: bool
    provider: str
    frame: str
    timestamp: float

    @property
    def unseen_object_ids(self) -> tuple[str, ...]:
        return tuple(item.object_id for item in self.objects if not item.visible)


@dataclass(frozen=True)
class Observation:
    """Unico input normale di DECIDE, indipendente dal suo algoritmo."""

    scene: SceneState
    relations: PhysicalRelationState
    uncertainty: UncertaintyState

    @property
    def objects(self) -> tuple[SceneObject, ...]:
        """Accesso pratico agli oggetti, senza duplicare SceneState."""

        return self.scene.objects


class ObservationInvariantError(ValueError):
    """Un'Observation viola il contratto strutturale tra i macro-moduli."""


def validate_observation(observation: Observation) -> None:
    """Controlla il contratto senza ricostruire o modificare l'Observation."""
    scene = observation.scene
    relations = observation.relations
    uncertainty = observation.uncertainty
    ids = tuple(obj.object_id for obj in scene.objects)
    if len(ids) != len(set(ids)):
        raise ObservationInvariantError("Object ID duplicati nella scena")
    if tuple(relations.object_ids) != ids:
        raise ObservationInvariantError("ID del grafo non allineati alla scena")
    uncertainty_ids = tuple(item.object_id for item in uncertainty.objects)
    if len(uncertainty_ids) != len(set(uncertainty_ids)):
        raise ObservationInvariantError("ID duplicati nell'incertezza")
    if uncertainty_ids[:len(ids)] != ids:
        raise ObservationInvariantError("ID dell'incertezza non allineati alla scena")
    if any(item.visible for item in uncertainty.objects[len(ids):]):
        raise ObservationInvariantError("Un oggetto assente dalla scena non puo' essere visibile")
    if len({scene.frame, relations.frame, uncertainty.frame}) != 1:
        raise ObservationInvariantError("Frame non allineati")
    if len({scene.timestamp, relations.timestamp, uncertainty.timestamp}) != 1:
        raise ObservationInvariantError("Timestamp non allineati")
    for relation in relations.relations:
        if relation.source_id not in ids or relation.target_id not in ids:
            raise ObservationInvariantError("Relazione riferita a un oggetto assente")
