"""Dati pubblici. Nessuna dipendenza da MuJoCo, Gymnasium o ROS."""

from dataclasses import dataclass
import numpy as np

STATE_FEATURES_PER_OBJECT = 17


@dataclass(frozen=True)
class ObjectObservation:
    object_id: str
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    mass: float
    friction: float
    present: bool
    is_target: bool


@dataclass(frozen=True)
class Observation:
    """Stima disponibile a DECIDE, ordinata secondo gli slot delle azioni.

    In 0B/1A coincide con la lettura esatta. In 1B questi campi saranno
    stimati: il contratto percettivo definitivo richiede la fase 1B.
    """

    objects: tuple[ObjectObservation, ...]

    def as_vector(self) -> np.ndarray:
        return np.asarray(
            [
                v
                for o in self.objects
                for v in (
                    *o.position,
                    *o.quaternion,
                    *o.linear_velocity,
                    *o.angular_velocity,
                    o.mass,
                    o.friction,
                    float(o.present),
                    float(o.is_target),
                )
            ],
            dtype=np.float32,
        )

    def action_index(self, decision: "ObjectDecision") -> int:
        return tuple(o.object_id for o in self.objects).index(decision.object_id)


@dataclass(frozen=True)
class PrivilegedState:
    """Verita simulata per supervisione e valutazione, distinta da Observation."""

    objects: tuple[ObjectObservation, ...]


@dataclass(frozen=True)
class ObjectDecision:
    object_id: str


@dataclass(frozen=True)
class ExecutionOutcome:
    object_id: str
    removed: bool
    failure_reason: str | None = None


@dataclass(frozen=True)
class TaskOutcome:
    reward: float
    terminated: bool
    disturbance: float
    target_removed: bool
    target_just_removed: bool
    collapsed: bool
    collapsed_ever: bool
    scene_is_empty: bool
    is_success: bool
