"""Rilevazioni per le due viste; solo l'adapter oracle conosce il simulatore."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from physical_ai_mujoco.contracts import Detection

from .capture import StereoFrame


@dataclass(frozen=True)
class StereoDetections:
    """Output stereo retrocompatibile, esposto anche come tuple di Detection."""

    left_masks: dict[str, np.ndarray]
    right_masks: dict[str, np.ndarray]
    type_ids: dict[str, str] | None
    source: str
    class_confidences: dict[str, float] | None = None

    @property
    def detections(self) -> tuple[Detection, ...]:
        identifiers = sorted(self.left_masks.keys() | self.right_masks.keys())
        result = []
        for track_id in identifiers:
            left = self.left_masks.get(track_id)
            if left is None:
                # La pipeline 3D usa la camera sinistra come riferimento.
                continue
            result.append(
                Detection(
                    track_id=track_id,
                    left_mask=left,
                    right_mask=self.right_masks.get(track_id),
                    class_id=None if self.type_ids is None else self.type_ids.get(track_id),
                    class_confidence=(
                        None
                        if self.class_confidences is None
                        else self.class_confidences.get(track_id)
                    ),
                )
            )
        return tuple(result)


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: StereoFrame) -> StereoDetections:
        """Riceve solo misure; non ha accesso allo stato simulato."""


class OracleDetector(Detector):
    """Adapter di test: segmentazione e identita' ideali del simulatore."""

    def __init__(self, simulator):
        self.simulator = simulator

    def detect(self, frame: StereoFrame) -> StereoDetections:
        left = self.simulator.render_instance_masks("cam_left")
        right = self.simulator.render_instance_masks("cam_right")
        expected = frame.gray_left.shape
        if any(mask.shape != expected for mask in (*left.values(), *right.values())):
            raise ValueError("Le maschere oracle non coincidono con il frame")
        types = {item.instance_id: item.type_id for item in self.simulator.scene.objects}
        confidences = {track_id: 1.0 for track_id in left.keys() | right.keys()}
        return StereoDetections(left, right, types, "mujoco_oracle", confidences)
