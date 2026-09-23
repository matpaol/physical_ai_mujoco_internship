"""Menu leggero del laboratorio OSSERVA; la logica vive nei moduli dedicati."""

from datetime import datetime
import json
from pathlib import Path
import sys
import traceback

# Con l'avvio tramite percorso del file, Python non aggiunge la radice del
# progetto a sys.path. L'avvio come modulo non richiede questo passaggio.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from physical_ai_mujoco.evaluation.observe_benchmark import (
    _resolve_detector_weights,
    main as benchmark_main,
)
from physical_ai_mujoco.vision_training import available_recipes, load_recipe
from physical_ai_mujoco.vision_training.dataset import (
    DATASETS_DIR,
    generate_dataset,
    main as dataset_main,
)
from physical_ai_mujoco.vision_training.evaluation import (
    evaluate as evaluate_detector,
    format_report,
    main as evaluation_main,
)
from physical_ai_mujoco.vision_training.recipe import evaluation_settings_from_dict
from physical_ai_mujoco.vision_training.training import (
    main as training_main,
    train as train_detector,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DETECTOR_WEIGHTS_DIR = PROJECT_ROOT / "outputs/detector_weights"
ERROR_LOG_DIR = PROJECT_ROOT / "outputs/logs"


def _ask_int(
    label: str, default: int, minimum: int = 1, maximum: int | None = None
) -> int:
    while True:
        answer = input(f"{label} [{default}]: ").strip()
        if not answer:
            return default
        if (
            answer.isdigit()
            and int(answer) >= minimum
            and (maximum is None or int(answer) <= maximum)
        ):
            return int(answer)
        suffix = "" if maximum is None else f" e minore o uguale a {maximum}"
        print(f"Inserire un intero maggiore o uguale a {minimum}{suffix}.")


def _choose(label: str, options: list[str]) -> int:
    """Stampa un elenco numerato e restituisce l'indice scelto (da 0)."""
    for index, text in enumerate(options, 1):
        print(f"  {index}. {text}")
    return _ask_int(label, 1, 1, len(options)) - 1


def _choose_recipe(kind: str):
    recipes = available_recipes(kind)
    if not recipes:
        print(f"Nessuna ricetta di tipo '{kind}' in configs/vision_training.")
        return None
    print(f"\nRicette di {'dataset' if kind == 'dataset' else 'training'} disponibili:")
    index = _choose(
        "Ricetta",
        [f"{path.stem} — {data.get('description', '')}" for path, data in recipes],
    )
    return load_recipe(recipes[index][0])


def _generate_interactive() -> int:
    recipe = _choose_recipe("dataset")
    if recipe is None:
        return 1
    destination = DATASETS_DIR / recipe.name
    print(f"\nGenero {recipe.scene_count} scene in {destination} ...")
    summary = generate_dataset(recipe, destination)
    print(f"\nDataset creato: {destination}")
    for split, stats in summary["splits"].items():
        print(
            f"  {split}: {stats['images']} immagini, target etichettato "
            f"{stats['target_labelled']}, senza target {stats['target_absent']}"
        )
    return 0


def _train_interactive() -> int:
    recipe = _choose_recipe("training")
    if recipe is None:
        return 1
    try:
        card = train_detector(recipe)
    except (RuntimeError, FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"\nImpossibile avviare il training: {error}")
        return 1
    print(f"Pesi: {card.with_suffix('.pt')}\nScheda: {card}")
    return 0


def _evaluate_interactive() -> int:
    weights = sorted(DETECTOR_WEIGHTS_DIR.glob("*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    datasets = sorted(
        (path.parent.parent for path in DATASETS_DIR.glob("*/annotations/val")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not weights or not datasets:
        print("Servono almeno un modello in outputs/detector_weights e un dataset generato.")
        return 1
    print("\nModelli:")
    model = weights[_choose("Modello", [path.name for path in weights])]
    print("\nDataset:")
    dataset = datasets[_choose("Dataset", [path.name for path in datasets])]
    splits = [split for split in ("test", "val", "train") if any((dataset / "annotations" / split).glob("*.json"))]
    print("\nSplit (test = scene mai viste in addestramento):")
    split = splits[_choose("Split", splits)]
    report = evaluate_detector(model, dataset, evaluation_settings_from_dict({"split": split}))
    print()
    print(format_report(report))
    return 0


def _detector_status() -> tuple[str, Path | None]:
    try:
        weights = _resolve_detector_weights(None, DETECTOR_WEIGHTS_DIR)
    except FileNotFoundError:
        return "non disponibile", None
    return f"pronto ({weights.name})", weights


def _interactive_menu() -> int:
    detector_status, detector_weights = _detector_status()
    print("\nLaboratorio OSSERVA\n")
    print(f"Detector: {detector_status}")
    print(f"Cartella pesi: {DETECTOR_WEIGHTS_DIR}")
    if detector_weights is not None:
        print(f"Modello attivo: {detector_weights}\n")
    else:
        print("Modello attivo: nessuno; fusion_learned non e' disponibile\n")
    print("  1. Test visivo e confronto sorgenti OSSERVA")
    print("  2. Genera un dataset del visore (da ricetta)")
    print("  3. Addestra il detector (da ricetta)")
    print("  4. Benchmark e confronti")
    print("  5. Valuta il detector per % di target visibile")
    print("  0. Esci\n")
    choice = _ask_int("Scelta", 1, 0, 5)
    if choice == 0:
        return 0
    if choice == 1:
        # Lascia al benchmark la scelta esplicita di scene, oggetti e seed.
        # Forzare sempre 1 scena/3 oggetti replicava la stessa anteprima.
        return benchmark_main(["--mode", "fusion_oracle"])
    if choice == 2:
        return _generate_interactive()
    if choice == 3:
        return _train_interactive()
    if choice == 4:
        return benchmark_main([])
    if choice == 5:
        return _evaluate_interactive()
    print("Scelta non valida.")
    return 1


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "dataset":
        return dataset_main(arguments[1:])
    if arguments and arguments[0] == "train-detector":
        return training_main(arguments[1:])
    if arguments and arguments[0] == "evaluate-detector":
        return evaluation_main(arguments[1:])
    if arguments and arguments[0] == "benchmark":
        return benchmark_main(arguments[1:])
    if arguments:
        # Compatibilita' con il vecchio avvio: --mode, --config, ...
        return benchmark_main(arguments)
    if sys.stdin.isatty():
        return _interactive_menu()
    return benchmark_main([])


def _write_error_log() -> Path | None:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    path = ERROR_LOG_DIR / f"osserva_error_{stamp}.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(traceback.format_exc(), encoding="utf-8")
    except OSError:
        return None
    return path


def run_cli(argv: list[str] | None = None) -> int:
    """Frontiera CLI: messaggi leggibili, traceback conservati nei log."""
    try:
        return main(argv)
    except KeyboardInterrupt:
        print("\nOperazione annullata dall'utente.")
        return 130
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError,
            json.JSONDecodeError) as error:
        print(f"\nOperazione non completata: {error}")
        return 1
    except Exception as error:  # La UI non deve esporre traceback grezzi.
        log_path = _write_error_log()
        print(f"\nErrore interno OSSERVA: {type(error).__name__}: {error}")
        if log_path is not None:
            print(f"Dettagli tecnici salvati in: {log_path}")
        else:
            print("Impossibile salvare il log tecnico.")
        return 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
