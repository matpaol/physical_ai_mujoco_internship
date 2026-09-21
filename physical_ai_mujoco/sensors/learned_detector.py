"""Adapter opzionale per un segmenter appreso, senza dipendenze nel core."""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Protocol

import cv2
import numpy as np

from .detector import Detector, StereoDetections
from .tracking import ObjectTracker


def resolve_detector_weights(
    requested: str | Path | None = None,
    *,
    directory: str | Path | None = None,
) -> Path:
    """Risolve un modello esplicito, canonico o il checkpoint PFM-1 piu' recente."""
    if requested is not None:
        selected = Path(requested).expanduser().resolve()
        if selected.is_file():
            return selected
        raise FileNotFoundError(f"Pesi detector non trovati: {selected}")

    if directory is None:
        directory = Path(__file__).resolve().parents[2] / "outputs/detector_weights"
    directory = Path(directory)
    canonical = directory / "pfm_1_seg.pt"
    if canonical.is_file():
        return canonical
    candidates = sorted(
        directory.glob("pfm_1_seg*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"Nessun modello PFM-1 trovato in {directory}. "
        "Genera il dataset e addestra il detector dal laboratorio OSSERVA."
    )


@dataclass(frozen=True)
class SegmentationPrediction:
    mask: np.ndarray
    class_index: int
    confidence: float


class SegmentationBackend(Protocol):
    def predict(self, image: np.ndarray) -> tuple[SegmentationPrediction, ...]: ...


class UltralyticsSegmentationBackend:
    def __init__(self, weights_path: str | Path, confidence_threshold: float = 0.25):
        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Pesi detector non trovati: {path}. "
                "Genera il dataset e addestra il detector dal laboratorio OSSERVA."
            )
        try:
            from ultralytics import YOLO
        except ImportError as error:
            project_root = Path(__file__).resolve().parents[2]
            raise RuntimeError(
                "Il detector appreso non e' installato nell'interprete corrente. "
                f"Esegui: {sys.executable} -m pip install -e "
                f"'{project_root}[detector]'"
            ) from error
        self.model = YOLO(str(path))
        self.confidence_threshold = float(confidence_threshold)

    def predict(self, image: np.ndarray) -> tuple[SegmentationPrediction, ...]:
        result = self.model.predict(
            source=np.asarray(image),
            conf=self.confidence_threshold,
            verbose=False,
        )[0]
        if result.masks is None or result.boxes is None:
            return ()
        height, width = image.shape[:2]
        masks = result.masks.data.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy()
        predictions = []
        for mask, class_index, confidence in zip(masks, classes, confidences):
            resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            predictions.append(
                SegmentationPrediction(
                    mask=resized >= 0.5,
                    class_index=int(class_index),
                    confidence=float(confidence),
                )
            )
        return tuple(predictions)


class LearnedDetector(Detector):
    """Segmenta la vista sinistra e stabilizza gli ID tra frame consecutivi."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        *,
        backend: SegmentationBackend | None = None,
        class_names: tuple[str, ...] = ("obstacle", "pfm_1_target"),
        confidence_threshold: float = 0.25,
        tracker: ObjectTracker | None = None,
    ):
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold deve essere in [0, 1]")
        if not class_names or len(set(class_names)) != len(class_names):
            raise ValueError("class_names deve contenere classi uniche")
        if backend is None:
            if weights_path is None:
                raise ValueError("Fornire weights_path oppure un backend")
            backend = UltralyticsSegmentationBackend(weights_path, confidence_threshold)
        self.backend = backend
        self.class_names = class_names
        self.confidence_threshold = float(confidence_threshold)
        self.tracker = tracker or ObjectTracker()

    def reset(self, seed=None):
        self.tracker.reset(seed)

    def detect(self, frame) -> StereoDetections:
        predictions = self.backend.predict(frame.gray_left)
        left_masks = {}
        type_ids = {}
        confidences = {}
        accepted = [
            item for item in predictions
            if item.confidence >= self.confidence_threshold and np.asarray(item.mask).any()
        ]
        accepted.sort(
            key=lambda item: (
                item.class_index,
                float(np.argwhere(np.asarray(item.mask))[..., 1].mean()),
            )
        )
        for index, item in enumerate(accepted):
            if not 0 <= item.class_index < len(self.class_names):
                raise ValueError(f"Classe predetta fuori catalogo: {item.class_index}")
            mask = np.asarray(item.mask, dtype=bool)
            if mask.shape != frame.gray_left.shape:
                raise ValueError("Maschera learned non allineata all'immagine")
            track_id = f"learned_{index:03d}"
            left_masks[track_id] = mask
            type_ids[track_id] = self.class_names[item.class_index]
            confidences[track_id] = float(item.confidence)
        temporary = StereoDetections(
            left_masks=left_masks,
            right_masks={},
            type_ids=type_ids,
            source="learned_left_segmentation",
            class_confidences=confidences,
        )
        return self.tracker.update(temporary)
