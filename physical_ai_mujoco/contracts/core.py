"""Dati pubblici. Nessuna dipendenza da MuJoCo, Gymnasium o ROS."""

from dataclasses import dataclass
import numpy as np

from .observation import Observation

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
class PrivilegedState:
    """Verita' simulata per teacher, supervisione e valutazione."""

    objects: tuple[ObjectObservation, ...]

    def as_vector(self) -> np.ndarray:
        """Codifica storica a 17 valori, riservata al teacher 1A."""
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
