"""Moduli logici della suite; un test compare in un solo modulo principale."""

from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parent
SUITES = {
    "scena": (
        "test_phase_0a.py",
        "test_mesh_target.py",
    ),
    "osserva": (
        "test_observe_pipeline.py",
        "test_sensor_observer_boundary.py",
        "test_sensor_pipeline_completion.py",
        "test_sensors_extended.py",
        "test_simulated_stereo.py",
    ),
    "validazione_sensori": (
        "test_sensor_validation.py",
    ),
    "pipeline_dati": (
        "test_learned_detector.py",
        "test_observe_menu.py",
    ),
    "decidi": (
        "test_phase_0b.py",
    ),
    "architettura": (
        "test_architecture.py",
        "test_test_runner.py",
    ),
}


def suite_paths(name: str) -> tuple[Path, ...]:
    if name == "tutti":
        return (TEST_ROOT,)
    if name not in SUITES:
        raise ValueError(f"Modulo di test sconosciuto: {name}")
    paths = tuple(TEST_ROOT / filename for filename in SUITES[name])
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"File di test mancanti: {', '.join(missing)}")
    return paths


def unassigned_test_files() -> tuple[str, ...]:
    assigned = {filename for paths in SUITES.values() for filename in paths}
    return tuple(
        sorted(path.name for path in TEST_ROOT.glob("test_*.py") if path.name not in assigned)
    )
