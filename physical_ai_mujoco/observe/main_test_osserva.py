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
from physical_ai_mujoco.experiments.synthetic_dataset import (
    DatasetGenerationConfig,
    generate_synthetic_dataset,
    main as dataset_main,
)
from physical_ai_mujoco.experiments.train_detector import (
    DetectorTrainingConfig,
    train_detector,
    main as training_main,
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


def _generate_interactive() -> int:
    scenes = _ask_int("Quante scene generare", 100)
    objects = _ask_int("Quanti oggetti per scena", 6)
    seed = _ask_int("Seed", 0, 0)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    destination = PROJECT_ROOT / "datasets/generated" / f"pfm_1_{stamp}"
    summary = generate_synthetic_dataset(
        DatasetGenerationConfig(scene_count=scenes, object_count=objects, seed=seed),
        destination,
    )
    print(f"\nDataset creato: {destination}")
    print(
        f"Immagini: {summary['image_count']} | "
        f"target visibile: {summary['target_visible_images']} | "
        f"target nascosto: {summary['target_hidden_images']}"
    )
    return 0


def _train_interactive() -> int:
    datasets = sorted(
        (PROJECT_ROOT / "datasets/generated").glob("pfm_1*/data.yaml"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not datasets:
        print("Nessun dataset disponibile. Usa prima 'Genera dataset sintetico'.")
        return 1
    print("\nDataset disponibili:")
    for index, path in enumerate(datasets, 1):
        print(f"  {index}. {path.parent.name}")
    selected = datasets[_ask_int("Dataset", 1, 1, len(datasets)) - 1]
    epochs = _ask_int("Epoche", 100)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    destination = PROJECT_ROOT / "outputs/detector_weights" / f"pfm_1_seg_{stamp}.pt"
    try:
        result = train_detector(
            DetectorTrainingConfig(selected, destination, epochs=epochs)
        )
    except (RuntimeError, FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"\nImpossibile avviare il training: {error}")
        return 1
    print(f"Pesi salvati in: {result}")
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
    print("  2. Genera dataset sintetico della PFM-1")
    print("  3. Addestra il detector")
    print("  4. Benchmark e confronti")
    print("  0. Esci\n")
    choice = _ask_int("Scelta", 1, 0, 4)
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
    print("Scelta non valida.")
    return 1


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "dataset":
        return dataset_main(arguments[1:])
    if arguments and arguments[0] == "train-detector":
        return training_main(arguments[1:])
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
        path.write_text(traceback.format_exc())
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
