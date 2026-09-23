"""Scrittura del dataset: immagini, maschere lossless, JSON e poligoni YOLO-seg.

Lo stesso formato vale per immagini sintetiche e reali: un campione e' una
immagine, un file di etichette YOLO e un JSON di annotazione con le istanze.
Le immagini reali etichettate a mano potranno entrare nello stesso schema e
passare per la stessa valutazione.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


SPLITS = ("train", "val", "test")
FOLDERS = ("images", "labels", "masks", "annotations")


def mask_to_yolo_segments(mask: np.ndarray) -> tuple[tuple[float, ...], ...]:
    """Converte le componenti visibili in poligoni YOLO-seg normalizzati."""
    region = np.asarray(mask, dtype=np.uint8)
    if region.ndim != 2:
        raise ValueError("La maschera deve essere bidimensionale")
    height, width = region.shape
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if len(contour) < 3 or cv2.contourArea(contour) <= 0:
            continue
        points = contour[:, 0, :].astype(float)
        normalized = np.column_stack((points[:, 0] / width, points[:, 1] / height))
        segments.append(tuple(float(value) for value in normalized.ravel()))
    return tuple(segments)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Impossibile salvare {path}")


def prepare_directory(destination: Path) -> None:
    """Crea la struttura del dataset; rifiuta una cartella gia' usata."""
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"La destinazione non e' vuota: {destination}. Scegli un nuovo nome."
        )
    destination.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for folder in FOLDERS:
            (destination / folder / split).mkdir(parents=True, exist_ok=True)


def write_sample(
    destination: Path,
    split: str,
    sample_id: str,
    image: np.ndarray,
    masks: dict[str, np.ndarray],
    type_ids: dict[str, str],
    class_names: tuple[str, ...],
    minimum_visible_pixels: int,
    extra: dict | None = None,
) -> dict:
    """Scrive un campione e restituisce la sua annotazione.

    `extra` sono campi aggiuntivi dell'annotazione (visibilita', condizioni
    visive, ...), scritti nello stesso JSON del campione.

    `class_names[0]` e' la classe degli ostacoli; ogni altra classe coincide
    con un type_id. Le istanze sotto `minimum_visible_pixels` non vengono
    etichettate: nemmeno un annotatore umano le segnerebbe.
    """
    if split not in SPLITS:
        raise ValueError(f"Split sconosciuto: {split}")
    height, width = image.shape[:2]
    image_path = destination / "images" / split / f"{sample_id}.png"
    label_path = destination / "labels" / split / f"{sample_id}.txt"
    annotation_path = destination / "annotations" / split / f"{sample_id}.json"
    write_image(image_path, image)

    instances = []
    labels = []
    for instance_id, raw_mask in sorted(masks.items()):
        mask = np.asarray(raw_mask, dtype=bool)
        visible_pixels = int(mask.sum())
        if visible_pixels < minimum_visible_pixels:
            continue
        type_id = type_ids.get(instance_id)
        class_name = type_id if type_id in class_names[1:] else class_names[0]
        class_index = class_names.index(class_name)
        rows, columns = np.nonzero(mask)
        bbox = [int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())]
        mask_path = destination / "masks" / split / f"{sample_id}_{instance_id}.png"
        write_image(mask_path, mask.astype(np.uint8) * 255)
        segments = mask_to_yolo_segments(mask)
        # YOLO-seg rappresenta una istanza con un solo poligono: si esporta la
        # componente visibile maggiore, per non trasformare frammenti occlusi
        # in oggetti distinti. La maschera lossless resta completa.
        for segment in segments[:1]:
            labels.append(" ".join((str(class_index), *(f"{value:.8f}" for value in segment))))
        instances.append(
            {
                "instance_id": instance_id,
                "class_id": class_name,
                "class_index": class_index,
                "visible_pixels": visible_pixels,
                "bbox_xyxy": bbox,
                "mask": str(mask_path.relative_to(destination)),
                "visible_components": len(segments),
                "yolo_polygon_exported": bool(segments),
            }
        )

    label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    annotation = {
        "sample_id": sample_id,
        "split": split,
        "image": str(image_path.relative_to(destination)),
        "width": width,
        "height": height,
        "instances": instances,
        **(extra or {}),
    }
    annotation_path.write_text(json.dumps(annotation, indent=2) + "\n", encoding="utf-8")
    return annotation


def write_data_yaml(path: Path, dataset_root: Path, class_names: tuple[str, ...]) -> Path:
    """File YOLO con percorso assoluto, riscritto al momento dell'uso.

    Un percorso fissato alla generazione smette di valere appena il dataset
    viene copiato altrove: e' successo con `pfm_1_immersed_v2`.
    """
    splits = [split for split in SPLITS if any((dataset_root / "images" / split).glob("*.png"))]
    lines = [f"path: {dataset_root.resolve()}"]
    for split in splits:
        lines.append(f"{split}: images/{split}")
    lines.append("names:")
    lines.extend(f"  {index}: {name}" for index, name in enumerate(class_names))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
