"""Il menu dei test deve coprire tutti i file ed eseguire il modulo scelto."""

import main_test
from tests.suites import SUITES, suite_paths, unassigned_test_files


def test_all_tests_are_assigned_to_one_module():
    files = [filename for group in SUITES.values() for filename in group]
    assert len(files) == len(set(files))
    assert unassigned_test_files() == ()


def test_suite_paths_are_existing_files():
    for name in SUITES:
        assert all(path.is_file() for path in suite_paths(name))


def test_menu_runs_only_selected_suite(monkeypatch):
    monkeypatch.setattr(main_test, "unassigned_test_files", lambda: ())
    calls = []
    monkeypatch.setattr(main_test.subprocess, "call", lambda command: calls.append(command) or 0)
    assert main_test.main(["--suite", "osserva"]) == 0
    assert len(calls) == 1
    assert calls[0][1:4] == ["-m", "pytest", "-q"]
    assert "test_sensor_pipeline_completion.py" in " ".join(calls[0])
