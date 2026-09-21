"""Training opzionale del segmenter; il core non importa Ultralytics."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DetectorTrainingConfig:
    dataset_yaml: Path
    destination: Path
    base_model: str = "yolo11n-seg.pt"
    epochs: int = 100
    image_size: int = 640
    batch_size: int = 8
    seed: int = 0

    def validate(self) -> None:
        if not self.dataset_yaml.is_file():
            raise FileNotFoundError(f"Dataset YAML non trovato: {self.dataset_yaml}")
        if self.destination.exists():
            raise FileExistsError(
                f"I pesi esistono gia': {self.destination}. "
                "Scegli un nuovo nome per non sovrascrivere un esperimento."
            )
        metadata = self.destination.with_suffix(".json")
        if metadata.exists():
            raise FileExistsError(
                f"I metadati del modello esistono gia': {metadata}. "
                "Scegli un nuovo nome per non mescolare esperimenti."
            )
        if self.epochs < 1 or self.image_size < 32 or self.batch_size < 1 or self.seed < 0:
            raise ValueError("Parametri di training non validi")


def train_detector(config: DetectorTrainingConfig) -> Path:
    config.validate()
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Training detector non disponibile. Installa esplicitamente: "
            "pip install -e '.[detector]'"
        ) from error

    run_root = PROJECT_ROOT / "outputs/detector_training"
    model = YOLO(config.base_model)
    result = model.train(
        data=str(config.dataset_yaml),
        epochs=config.epochs,
        imgsz=config.image_size,
        batch=config.batch_size,
        seed=config.seed,
        project=str(run_root),
        name=config.destination.stem,
        exist_ok=False,
    )
    save_dir = Path(result.save_dir)
    best = save_dir / "weights/best.pt"
    if not best.is_file():
        raise RuntimeError(f"Il training non ha prodotto {best}")
    config.destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, config.destination)
    metrics = getattr(result, "results_dict", {})
    metadata = {
        "training_config": {
            **asdict(config),
            "dataset_yaml": str(config.dataset_yaml),
            "destination": str(config.destination),
        },
        "ultralytics_run_directory": str(save_dir),
        "best_source_weights": str(best),
        "exported_weights": str(config.destination),
        "metrics": {
            str(key): float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) or hasattr(value, "item")
        },
    }
    config.destination.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return config.destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Addestra il detector PFM-1 YOLO-seg")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "datasets/generated/pfm_1/data.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/detector_weights/pfm_1_seg.pt",
    )
    parser.add_argument("--base-model", default="yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    destination = train_detector(
        DetectorTrainingConfig(
            dataset_yaml=args.data.resolve(),
            destination=args.output.resolve(),
            base_model=args.base_model,
            epochs=args.epochs,
            image_size=args.image_size,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    )
    print(f"Pesi migliori salvati in: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
