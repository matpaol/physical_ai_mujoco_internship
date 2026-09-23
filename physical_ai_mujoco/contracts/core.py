"""Dati pubblici. Nessuna dipendenza da MuJoCo, Gymnasium o ROS."""

from dataclasses import dataclass
import numpy as np

from .observation import Observation

STATE_FEATURES_PER_OBJECT = 17


@dataclass(frozen=True)
class ObjectObservation:
    """Simulator truth about one object, part of the privileged branch.

    The first nine fields form the legacy 17-value teacher vector and must not
    change. The optional fields below extend the truth available to teachers
    without altering that vector.
    """

    object_id: str
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    mass: float
    friction: float  # sliding friction coefficient
    present: bool
    is_target: bool
    type_id: str | None = None
    shape: str | None = None
    size: tuple[float, float, float] | None = None  # full extents x, y, z [m]
    center_of_mass: tuple[float, float, float] | None = None  # offset from the geometric centre, body frame [m]
    torsional_friction: float | None = None
    rolling_friction: float | None = None


@dataclass(frozen=True)
class PrivilegedState:
    """Simulator truth for teachers, oracles, labels and evaluation.

    ``contact_supports`` holds ``(lower_id, upper_id)`` pairs between present
    objects, read from the active simulator contacts on a settled scene. It is
    the contact support graph, not the causal dependency ground truth.
    """

    objects: tuple[ObjectObservation, ...]
    contact_supports: tuple[tuple[str, str], ...] = ()

    def as_vector(self) -> np.ndarray:
        """Legacy 17-values-per-object encoding, reserved to the 1A teacher."""
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
