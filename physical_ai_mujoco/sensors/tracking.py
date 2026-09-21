"""Tracking leggero di istanze segmentate tra frame consecutivi."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detector import StereoDetections


@dataclass(frozen=True)
class ObjectTrackerConfig:
    minimum_iou: float = 0.05
    maximum_centroid_distance_px: float = 80.0
    iou_weight: float = 0.65
    max_missed_frames: int = 3

    def __post_init__(self):
        if not 0 <= self.minimum_iou <= 1:
            raise ValueError("minimum_iou deve essere in [0, 1]")
        if self.maximum_centroid_distance_px <= 0:
            raise ValueError("maximum_centroid_distance_px deve essere positivo")
        if not 0 <= self.iou_weight <= 1:
            raise ValueError("iou_weight deve essere in [0, 1]")
        if self.max_missed_frames < 0:
            raise ValueError("max_missed_frames deve essere non negativo")


@dataclass
class _Track:
    track_id: str
    mask: np.ndarray
    centroid: tuple[float, float]
    class_id: str | None
    missed_frames: int = 0


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    rows, cols = np.nonzero(mask)
    if not len(rows):
        raise ValueError("Una detection da tracciare deve contenere almeno un pixel")
    return float(cols.mean()), float(rows.mean())


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.count_nonzero(first & second)
    union = np.count_nonzero(first | second)
    return float(intersection / union) if union else 0.0


class ObjectTracker:
    """Assegna ID persistenti con gating su classe, IoU e centroide.

    L'associazione e globale rispetto alle coppie ammissibili: le coppie sono
    ordinate per punteggio e ogni track/detection viene usata al massimo una
    volta. Le track scomparse restano riassociabili per un numero limitato di
    frame, senza essere emesse come detection correnti.
    """

    def __init__(self, config: ObjectTrackerConfig | None = None):
        self.config = config or ObjectTrackerConfig()
        self.reset()

    def reset(self, seed=None):
        self._tracks: dict[str, _Track] = {}
        self._next_id = 0

    def _new_track_id(self) -> str:
        result = f"track_{self._next_id:06d}"
        self._next_id += 1
        return result

    def update(self, detections: StereoDetections) -> StereoDetections:
        current = list(detections.detections)
        candidates = []
        centroid_by_input = {
            item.track_id: _centroid(np.asarray(item.left_mask, dtype=bool))
            for item in current
        }
        for track_id, track in self._tracks.items():
            for item in current:
                if (
                    track.class_id is not None
                    and item.class_id is not None
                    and track.class_id != item.class_id
                ):
                    continue
                overlap = _iou(track.mask, item.left_mask)
                center = centroid_by_input[item.track_id]
                distance = float(np.linalg.norm(np.asarray(track.centroid) - center))
                if (
                    overlap < self.config.minimum_iou
                    and distance > self.config.maximum_centroid_distance_px
                ):
                    continue
                distance_score = max(
                    0.0, 1.0 - distance / self.config.maximum_centroid_distance_px
                )
                score = (
                    self.config.iou_weight * overlap
                    + (1.0 - self.config.iou_weight) * distance_score
                )
                candidates.append((score, overlap, -distance, track_id, item.track_id))

        assigned_tracks: set[str] = set()
        assigned_inputs: dict[str, str] = {}
        for _, _, _, track_id, input_id in sorted(candidates, reverse=True):
            if track_id in assigned_tracks or input_id in assigned_inputs:
                continue
            assigned_tracks.add(track_id)
            assigned_inputs[input_id] = track_id

        for track in self._tracks.values():
            track.missed_frames += 1

        for item in current:
            stable_id = assigned_inputs.get(item.track_id)
            if stable_id is None:
                stable_id = self._new_track_id()
                assigned_inputs[item.track_id] = stable_id
            self._tracks[stable_id] = _Track(
                stable_id,
                np.asarray(item.left_mask, dtype=bool).copy(),
                centroid_by_input[item.track_id],
                item.class_id,
                0,
            )

        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track.missed_frames <= self.config.max_missed_frames
        }

        left_masks = {}
        right_masks = {}
        type_ids = {}
        confidences = {}
        for item in current:
            stable_id = assigned_inputs[item.track_id]
            left_masks[stable_id] = item.left_mask
            if item.right_mask is not None:
                right_masks[stable_id] = item.right_mask
            if item.class_id is not None:
                type_ids[stable_id] = item.class_id
            if item.class_confidence is not None:
                confidences[stable_id] = item.class_confidence
        return StereoDetections(
            left_masks,
            right_masks,
            type_ids or None,
            detections.source,
            confidences or None,
        )
