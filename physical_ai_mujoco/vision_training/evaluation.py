"""Valutazione del visore per fasce di percentuale visibile del target.

Risponde alla domanda da cui e' nato il modulo: *sotto quale % di sagoma
visibile il detector smette di riconoscere il target?* La risposta non si
sceglie a tavolino, si misura qui, e finisce nella scheda del modello, da cui
il benchmark di OSSERVA la legge per decidere quando un target conta come
"osservabile".

Funziona su qualunque dataset nel formato di `labels.py`: sintetico (con la %
visibile in ogni annotazione) o reale (senza, e allora le fasce diventano una
sola, "sconosciuta").
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from .dataset import visibility_band
from .recipe import PROJECT_ROOT, EvaluationSettings, evaluation_settings_from_dict


EVALUATIONS_DIR = PROJECT_ROOT / "outputs/detector_evaluations"
UNKNOWN_BAND = "sconosciuta"


def dataset_class_names(dataset_dir: Path) -> tuple[str, ...]:
    """Classi del dataset: dal riassunto se c'e', altrimenti da data.yaml."""
    summary = dataset_dir / "summary.json"
    if summary.is_file():
        names = json.loads(summary.read_text()).get("class_names")
        if names:
            return tuple(names)
    data_yaml = dataset_dir / "data.yaml"
    if data_yaml.is_file():
        names = {}
        inside = False
        for line in data_yaml.read_text().splitlines():
            if line.strip() == "names:":
                inside = True
                continue
            if inside and ":" in line and line.startswith(" "):
                key, value = line.split(":", 1)
                names[int(key.strip())] = value.strip()
            elif inside and line and not line.startswith(" "):
                inside = False
        if names:
            return tuple(names[index] for index in sorted(names))
    raise FileNotFoundError(f"Classi del dataset non trovate in {dataset_dir}")


def load_annotations(dataset_dir: Path, split: str) -> list[dict]:
    folder = dataset_dir / "annotations" / split
    paths = sorted(folder.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"Nessuna annotazione nello split '{split}' di {dataset_dir}")
    return [json.loads(path.read_text()) for path in paths]


def _visible_fraction(annotation: dict) -> float | None:
    # I dataset precedenti al modulo usavano un altro nome per lo stesso dato.
    value = annotation.get("target_visible_fraction", annotation.get("target_camera_visible_fraction"))
    return None if value is None else float(value)


def _band_limits(band: str) -> tuple[float, float] | None:
    try:
        low, high = (float(value) for value in band.split("-"))
    except ValueError:
        return None
    return low, high


def minimum_recognizable_fraction(
    bands: list[dict], required_recall: float, min_samples: int
) -> float | None:
    """Limite inferiore della fascia piu' bassa da cui in su il richiamo regge.

    Si scende dalla fascia piu' visibile verso le meno visibili, considerando
    solo quelle con abbastanza campioni; ci si ferma alla prima che non
    raggiunge `required_recall`. `None` se gia' la fascia piu' alta fallisce.
    """
    eligible = sorted(
        (item for item in bands if item["lower"] is not None and item["samples"] >= min_samples),
        key=lambda item: item["lower"],
        reverse=True,
    )
    result = None
    for item in eligible:
        if item["recall"] < required_recall:
            break
        result = item["lower"]
    return result


def evaluate(
    weights: str | Path,
    dataset_dir: str | Path,
    settings: EvaluationSettings = EvaluationSettings(),
    *,
    backend=None,
    output: str | Path | None = None,
    update_card: bool = True,
) -> dict:
    """Valuta i pesi su uno split e scrive il report (e la scheda, se esiste)."""
    weights = Path(weights).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    class_names = dataset_class_names(dataset_dir)
    if len(class_names) < 2:
        raise ValueError("Il dataset deve avere la classe ostacolo e la classe target")
    target_index = 1
    target_class = class_names[target_index]
    if backend is None:
        from physical_ai_mujoco.sensors.learned_detector import UltralyticsSegmentationBackend

        backend = UltralyticsSegmentationBackend(weights, settings.confidence)

    samples = []
    for annotation in load_annotations(dataset_dir, settings.split):
        image = cv2.imread(str(dataset_dir / annotation["image"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Immagine illeggibile: {annotation['image']}")
        predictions = [
            item for item in backend.predict(image)
            if item.class_index == target_index and item.confidence >= settings.confidence
        ]
        truth = np.zeros(image.shape, dtype=bool)
        labelled = False
        for instance in annotation["instances"]:
            if instance["class_index"] == target_index:
                mask = cv2.imread(str(dataset_dir / instance["mask"]), cv2.IMREAD_GRAYSCALE)
                truth |= mask > 0
                labelled = True
        overlaps = []
        for item in predictions:
            mask = np.asarray(item.mask, dtype=bool)
            union = int(np.logical_or(mask, truth).sum())
            overlaps.append(float(np.logical_and(mask, truth).sum() / union) if union else 0.0)
        best_iou = max(overlaps, default=0.0)
        fraction = _visible_fraction(annotation)
        present = bool(annotation.get("target_present", True))
        if labelled:
            kind = "etichettato"
            band = UNKNOWN_BAND if fraction is None else visibility_band(fraction, settings.band_width)
        elif present:
            kind = "non_etichettato"  # sepolto, o sotto la soglia minima di pixel
            band = None
        else:
            kind = "assente"
            band = None
        samples.append({
            "sample_id": annotation["sample_id"],
            "kind": kind,
            "band": band,
            "visible_fraction": fraction,
            "object_count": annotation.get("object_count"),
            "recognized": labelled and best_iou >= settings.iou_threshold,
            "best_iou": best_iou,
            "target_predictions": len(predictions),
        })

    labelled_samples = [item for item in samples if item["kind"] == "etichettato"]
    bands = []
    for band in sorted({item["band"] for item in labelled_samples}, key=lambda name: (_band_limits(name) is None, name)):
        members = [item for item in labelled_samples if item["band"] == band]
        limits = _band_limits(band)
        bands.append({
            "band": band,
            "lower": None if limits is None else limits[0],
            "upper": None if limits is None else limits[1],
            "samples": len(members),
            "recognized": sum(item["recognized"] for item in members),
            "recall": sum(item["recognized"] for item in members) / len(members),
            "mean_best_iou": float(np.mean([item["best_iou"] for item in members])),
        })
    by_count: dict[str, dict] = {}
    for item in labelled_samples:
        key = str(item["object_count"])
        entry = by_count.setdefault(key, {"samples": 0, "recognized": 0})
        entry["samples"] += 1
        entry["recognized"] += int(item["recognized"])
    for entry in by_count.values():
        entry["recall"] = entry["recognized"] / entry["samples"]
    unlabelled = [item for item in samples if item["kind"] == "non_etichettato"]
    negatives = [item for item in samples if item["kind"] == "assente"]
    minimum = minimum_recognizable_fraction(bands, settings.required_recall, settings.min_samples_per_band)

    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "weights": str(weights),
        "dataset": str(dataset_dir),
        "target_class": target_class,
        "settings": settings.__dict__,
        "sample_count": len(samples),
        "labelled_target_samples": len(labelled_samples),
        "target_recall": (
            sum(item["recognized"] for item in labelled_samples) / len(labelled_samples)
            if labelled_samples else None
        ),
        "bands": bands,
        "recall_by_object_count": dict(sorted(by_count.items(), key=lambda pair: int(pair[0]) if pair[0].isdigit() else 0)),
        "unlabelled_target_samples": len(unlabelled),
        "unlabelled_target_predicted": sum(item["target_predictions"] > 0 for item in unlabelled),
        "negative_samples": len(negatives),
        "false_positive_rate": (
            sum(item["target_predictions"] > 0 for item in negatives) / len(negatives)
            if negatives else None
        ),
        "minimum_recognizable_visible_fraction": minimum,
        "minimum_rule": (
            f"fascia piu' bassa da cui in su il richiamo e' >= {settings.required_recall:.0%} "
            f"(IoU >= {settings.iou_threshold}), contando le fasce con almeno "
            f"{settings.min_samples_per_band} campioni"
        ),
    }

    destination = Path(output) if output is not None else (
        EVALUATIONS_DIR / f"{weights.stem}__{dataset_dir.name}__{settings.split}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({**report, "samples": samples}, indent=2) + "\n")
    report["report_path"] = str(destination)
    if update_card:
        _update_card(weights, dataset_dir, settings, report)
    return report


def _update_card(weights: Path, dataset_dir: Path, settings: EvaluationSettings, report: dict) -> None:
    card_path = weights.with_suffix(".json")
    if not card_path.is_file():
        return
    card = json.loads(card_path.read_text())
    card.setdefault("evaluations", {})[f"{dataset_dir.name}:{settings.split}"] = report
    has_visibility = any(item["lower"] is not None for item in report["bands"])
    if settings.split == "test" and has_visibility:
        card["observability"] = {
            "minimum_recognizable_visible_fraction": report["minimum_recognizable_visible_fraction"],
            "required_recall": settings.required_recall,
            "iou_threshold": settings.iou_threshold,
            "dataset": dataset_dir.name,
            "split": settings.split,
        }
    card_path.write_text(json.dumps(card, indent=2) + "\n")


def format_report(report: dict) -> str:
    """Tabella leggibile da terminale."""
    lines = [
        f"Split: {report['settings']['split']} | campioni: {report['sample_count']} | "
        f"target etichettati: {report['labelled_target_samples']}",
    ]
    if report["target_recall"] is not None:
        lines.append(f"Richiamo del target complessivo: {report['target_recall']:.1%}")
    lines.append("  % visibile   campioni   richiamo   IoU medio")
    for item in report["bands"]:
        lines.append(
            f"  {item['band']:>11}   {item['samples']:>8}   {item['recall']:>8.1%}   {item['mean_best_iou']:>9.2f}"
        )
    if report["false_positive_rate"] is not None:
        lines.append(
            f"Scene senza target: {report['negative_samples']}, falsi positivi {report['false_positive_rate']:.1%}"
        )
    minimum = report["minimum_recognizable_visible_fraction"]
    lines.append(
        "Soglia minima misurata: "
        + ("nessuna fascia regge il richiamo richiesto" if minimum is None else f"{minimum:.0%} di sagoma visibile")
    )
    lines.append(f"Report completo: {report['report_path']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Valuta il visore per fasce di % visibile del target")
    parser.add_argument("weights", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--band-width", type=float, default=0.1)
    parser.add_argument("--required-recall", type=float, default=0.8)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    settings = evaluation_settings_from_dict({
        "split": args.split, "iou_threshold": args.iou, "confidence": args.confidence,
        "band_width": args.band_width, "required_recall": args.required_recall,
        "min_samples_per_band": args.min_samples,
    })
    report = evaluate(args.weights, args.dataset, settings, output=args.output)
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
