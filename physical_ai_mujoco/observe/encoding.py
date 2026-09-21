"""Codifica a slot fissi dell'Observation per policy vettoriali."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physical_ai_mujoco.contracts import Observation


OBJECT_FEATURE_NAMES = (
    "occupied",
    "visible",
    "is_target",
    "detected",
    "position_known",
    "position_x",
    "position_y",
    "position_z",
    "orientation_known",
    "quaternion_w",
    "quaternion_x",
    "quaternion_y",
    "quaternion_z",
    "size_known",
    "size_small",
    "size_medium",
    "size_large",
    "perception_quality",
    "detection_quality",
    "pose_quality",
    "relation_out_degree",
    "relation_in_degree",
)


@dataclass(frozen=True)
class EncodedObservation:
    vector: np.ndarray
    slot_ids: tuple[str | None, ...]
    action_mask: np.ndarray

    def __post_init__(self):
        if self.vector.ndim != 1 or self.vector.dtype != np.float32:
            raise ValueError("Il vettore codificato deve essere float32 monodimensionale")
        if self.action_mask.shape != (len(self.slot_ids),):
            raise ValueError("Action mask e slot non sono allineati")


class ObservationEncoder:
    """Mantiene uno slot per track ID e produce una shape PPO costante.

    Uno slot libero viene mantenuto per gli ID temporaneamente occlusi.
    Solo quando la capacita' e esaurita, un nuovo ID puo sostituire uno slot
    non attivo. Se le detection correnti superano la capacita', vengono
    conservate quelle di qualita' maggiore, dando priorita' al target.
    """

    def __init__(self, max_objects: int):
        if max_objects < 1:
            raise ValueError("max_objects deve essere positivo")
        self.max_objects = int(max_objects)
        self.reset()

    @property
    def feature_count(self) -> int:
        return len(OBJECT_FEATURE_NAMES)

    @property
    def vector_size(self) -> int:
        return (
            self.max_objects * self.feature_count
            + self.max_objects * self.max_objects
            + 1
        )

    def reset(self, seed=None):
        self._slot_by_id: dict[str, int] = {}
        self._slot_ids: list[str | None] = [None] * self.max_objects

    def _allocate(self, object_id: str, active_ids: set[str]) -> int | None:
        existing = self._slot_by_id.get(object_id)
        if existing is not None:
            return existing
        try:
            slot = self._slot_ids.index(None)
        except ValueError:
            stale = next(
                (
                    index for index, old_id in enumerate(self._slot_ids)
                    if old_id not in active_ids
                ),
                None,
            )
            if stale is None:
                return None
            old_id = self._slot_ids[stale]
            del self._slot_by_id[old_id]
            slot = stale
        self._slot_ids[slot] = object_id
        self._slot_by_id[object_id] = slot
        return slot

    def encode(self, observation: Observation) -> EncodedObservation:
        ordered = sorted(
            observation.scene.objects,
            key=lambda obj: (
                not obj.is_target,
                -(obj.perception_quality or 0.0),
                obj.object_id,
            ),
        )
        selected = ordered[:self.max_objects]
        active_ids = {obj.object_id for obj in selected}
        uncertainty = {
            item.object_id: item for item in observation.uncertainty.objects
        }
        features = np.zeros(
            (self.max_objects, self.feature_count), dtype=np.float32
        )
        action_mask = np.zeros(self.max_objects, dtype=bool)

        for obj in selected:
            slot = self._allocate(obj.object_id, active_ids)
            if slot is None:
                continue
            item = uncertainty.get(obj.object_id)
            position = (0.0, 0.0, 0.0) if obj.position is None else obj.position
            quaternion = (
                (0.0, 0.0, 0.0, 0.0)
                if obj.quaternion is None else obj.quaternion
            )
            size = (
                (0.0, 0.0, 0.0)
                if obj.size is None else tuple(sorted(obj.size))
            )
            out_degree = sum(
                relation.source_id == obj.object_id
                for relation in observation.relations.relations
            )
            in_degree = sum(
                relation.target_id == obj.object_id
                for relation in observation.relations.relations
            )
            features[slot] = np.asarray(
                (
                    1.0,
                    float(item.visible) if item is not None else float(obj.detected),
                    float(obj.is_target),
                    float(obj.detected),
                    float(obj.position is not None),
                    *position,
                    float(obj.quaternion is not None),
                    *quaternion,
                    float(obj.size is not None),
                    *size,
                    0.0 if obj.perception_quality is None else obj.perception_quality,
                    0.0 if item is None or item.detection_quality is None else item.detection_quality,
                    0.0 if item is None or item.pose_quality is None else item.pose_quality,
                    out_degree / self.max_objects,
                    in_degree / self.max_objects,
                ),
                dtype=np.float32,
            )
            action_mask[slot] = bool(obj.detected and obj.position is not None)

        adjacency = np.zeros(
            (self.max_objects, self.max_objects), dtype=np.float32
        )
        for relation in observation.relations.relations:
            source = self._slot_by_id.get(relation.source_id)
            target = self._slot_by_id.get(relation.target_id)
            if source is not None and target is not None:
                adjacency[source, target] = float(relation.relation_score)

        vector = np.concatenate(
            (
                features.ravel(),
                adjacency.ravel(),
                np.asarray([
                    float(observation.uncertainty.unknown_space or len(ordered) > self.max_objects)
                ], dtype=np.float32),
            )
        ).astype(np.float32, copy=False)
        return EncodedObservation(vector, tuple(self._slot_ids), action_mask)
