"""Addestramento del visore da una ricetta, con scheda del modello obbligatoria.

La scheda (`<nome>.json` accanto a `<nome>.pt`) dice da quale ricetta, da quale
dataset e da quale modello di partenza vengono i pesi, su che macchina sono
stati addestrati e come vanno sul test. Il modello `pfm_1_seg_immersed_v2.pt`
e' arrivato senza scheda: non si sa piu' con quante epoche ne su quali dati.

Il modello di partenza puo' essere un modello Ultralytics generico
(pre-training in simulazione) o un nostro `.pt` gia' addestrato (fine-tuning
sulle immagini reali): in quel caso la scheda registra il genitore.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from .evaluation import dataset_class_names, evaluate, format_report
from .labels import write_data_yaml
from .recipe import PROJECT_ROOT, TrainingRecipe, load_recipe


WEIGHTS_DIR = PROJECT_ROOT / "outputs/detector_weights"
RUNS_DIR = PROJECT_ROOT / "outputs/detector_training"


def select_device(requested: str) -> str:
    """'auto' sceglie CUDA, poi Apple MPS, poi CPU; altrimenti rispetta la richiesta."""
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    backends = getattr(torch, "backends", None)
    if backends is not None and getattr(backends, "mps", None) is not None and backends.mps.is_available():
        return "mps"
    return "cpu"


def _base_model(recipe: TrainingRecipe) -> tuple[str, dict | None]:
    """Modello di partenza e, se e' uno dei nostri, la sua scheda."""
    candidate = Path(recipe.base_model).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if candidate.suffix == ".pt" and candidate.is_file():
        card = candidate.with_suffix(".json")
        parent = json.loads(card.read_text()) if card.is_file() else None
        return str(candidate), parent
    if "/" in recipe.base_model or "\\" in recipe.base_model:
        raise FileNotFoundError(f"Modello di partenza non trovato: {candidate}")
    # Nome di un modello Ultralytics (es. yolo11n-seg.pt): lo scarica la libreria.
    return recipe.base_model, None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=5, check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() + ("+modifiche" if dirty else "")


def _versions() -> dict:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for module in ("ultralytics", "torch"):
        try:
            versions[module] = __import__(module).__version__
        except ImportError:
            versions[module] = None
    return versions


def train(recipe: TrainingRecipe, *, yolo_factory=None, run_evaluation: bool = True) -> Path:
    """Addestra, salva pesi e scheda, valuta sul test se c'e'. Ritorna la scheda.

    `yolo_factory` sostituisce `ultralytics.YOLO` nei test.
    """
    dataset = recipe.dataset.resolve()
    if not (dataset / "images" / "train").is_dir():
        raise FileNotFoundError(f"Dataset non trovato o senza split train: {dataset}")
    if not any((dataset / "images" / "val").glob("*.png")):
        raise ValueError(f"Il dataset {dataset.name} non ha immagini di validazione")
    weights = WEIGHTS_DIR / f"{recipe.name}.pt"
    card_path = weights.with_suffix(".json")
    run_directory = RUNS_DIR / recipe.name
    for existing in (weights, card_path, run_directory):
        if existing.exists():
            raise FileExistsError(
                f"Esiste gia' {existing}. Cambia 'name' nella ricetta: "
                "un esperimento non sovrascrive mai un altro."
            )
    base_model, parent_card = _base_model(recipe)
    class_names = dataset_class_names(dataset)
    data_yaml = write_data_yaml(run_directory / "data.yaml", dataset, class_names)
    device = select_device(recipe.device)

    if yolo_factory is None:
        try:
            from ultralytics import YOLO as yolo_factory
        except ImportError as error:
            raise RuntimeError(
                "Training non disponibile: installa il detector con "
                f"{sys.executable} -m pip install -e '.[detector]'"
            ) from error
    started = datetime.now().astimezone()
    model = yolo_factory(base_model)
    result = model.train(
        data=str(data_yaml),
        epochs=recipe.epochs,
        imgsz=recipe.image_size,
        batch=recipe.batch,
        patience=recipe.patience,
        seed=recipe.seed,
        device=device,
        project=str(RUNS_DIR),
        name=recipe.name,
        exist_ok=True,  # la cartella l'abbiamo creata noi per data.yaml
        **recipe.ultralytics,
    )
    save_dir = Path(getattr(result, "save_dir", None) or getattr(model.trainer, "save_dir"))
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Il training non ha prodotto {best}")
    weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, weights)

    summary_path = dataset / "summary.json"
    metrics = getattr(result, "results_dict", None) or {}
    card = {
        "name": recipe.name,
        "created_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "weights": str(weights),
        "class_names": list(class_names),
        "training_recipe": recipe.source,
        "resolved": {
            "dataset": str(dataset),
            "base_model": base_model,
            "device": device,
            "ultralytics_run_directory": str(save_dir),
        },
        "parent": None if parent_card is None else {
            "name": parent_card.get("name"),
            "weights": parent_card.get("weights"),
            "dataset": parent_card.get("resolved", {}).get("dataset"),
        },
        "dataset_summary": json.loads(summary_path.read_text()) if summary_path.is_file() else None,
        "environment": {**_versions(), "git_commit": _git_commit()},
        "metrics": {
            str(key): float(value) for key, value in metrics.items()
            if isinstance(value, (int, float)) or hasattr(value, "item")
        },
    }
    card_path.write_text(json.dumps(card, indent=2) + "\n")

    split = recipe.evaluation.split
    if run_evaluation and any((dataset / "annotations" / split).glob("*.json")):
        report = evaluate(weights, dataset, recipe.evaluation)
        print(format_report(report))
    return card_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Addestra il visore da una ricetta")
    parser.add_argument("recipe", type=Path, help="Ricetta JSON con kind='training'")
    parser.add_argument("--no-evaluation", action="store_true", help="Non valutare sul test a fine training")
    args = parser.parse_args(argv)
    recipe = load_recipe(args.recipe)
    if not isinstance(recipe, TrainingRecipe):
        parser.error("La ricetta indicata non e' di tipo 'training'")
    card = train(recipe, run_evaluation=not args.no_evaluation)
    print(f"Pesi e scheda salvati: {card.with_suffix('.pt')} , {card}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
