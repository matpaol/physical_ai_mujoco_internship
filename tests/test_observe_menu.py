"""Copertura dei percorsi del menu e della frontiera CLI di OSSERVA."""

from pathlib import Path

import pytest

import physical_ai_mujoco.observe.main_test_osserva as menu
import physical_ai_mujoco.evaluation.observe_benchmark as benchmark


@pytest.mark.parametrize(
    ("choice", "expected", "return_code"),
    [
        (0, "exit", 0),
        (1, "visual", 11),
        (2, "dataset", 12),
        (3, "training", 13),
        (4, "benchmark", 14),
        (5, "evaluation", 15),
    ],
)
def test_every_main_menu_branch(monkeypatch, choice, expected, return_code):
    calls = []
    monkeypatch.setattr(menu, "_detector_status", lambda: ("non disponibile", None))
    monkeypatch.setattr(menu, "_ask_int", lambda *args, **kwargs: choice)
    monkeypatch.setattr(
        menu,
        "benchmark_main",
        lambda args: calls.append(("benchmark", args)) or (11 if args else 14),
    )
    monkeypatch.setattr(
        menu, "_generate_interactive", lambda: calls.append(("dataset", None)) or 12
    )
    monkeypatch.setattr(
        menu, "_train_interactive", lambda: calls.append(("training", None)) or 13
    )
    monkeypatch.setattr(
        menu, "_evaluate_interactive", lambda: calls.append(("evaluation", None)) or 15
    )

    assert menu._interactive_menu() == return_code
    if expected == "exit":
        assert calls == []
    elif expected == "visual":
        assert calls == [
            ("benchmark", ["--mode", "fusion_oracle"])
        ]
    elif expected == "benchmark":
        assert calls == [("benchmark", [])]
    else:
        assert calls == [(expected, None)]


@pytest.mark.parametrize(
    ("choice", "expected_modes", "asks_stereo"),
    [
        (1, benchmark.DEFAULT_MODES, True),
        (2, ("exact",), False),
        (3, ("degraded",), False),
        (4, ("stereo",), True),
        (5, ("rgbd",), False),
        (6, ("fusion_oracle",), False),
        (7, ("fusion_learned",), False),
    ],
)
def test_every_sensor_selection_branch(monkeypatch, choice, expected_modes, asks_stereo):
    profile = Path("clean.json")
    monkeypatch.setattr(
        benchmark,
        "_interactive_profiles",
        lambda: [(profile, {"name": "Clean"})],
    )
    answers = iter([1, choice] + ([3] if asks_stereo else []))
    monkeypatch.setattr(benchmark, "_ask_int", lambda *args, **kwargs: next(answers))

    selected, modes, stereo_profile = benchmark._interactive_selection()

    assert selected == profile
    assert modes == expected_modes
    assert stereo_profile == ("random" if asks_stereo else None)


def test_real_profile_directory_contains_only_three_menu_profiles():
    profiles = benchmark._interactive_profiles(benchmark.CONFIG_DIR)
    assert [path.stem for path, _ in profiles] == ["clean", "fixed", "random"]


def test_visual_seed_is_random_by_default_and_reproducible_when_entered(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    monkeypatch.setattr(benchmark.secrets, "randbelow", lambda _: 123456)
    assert benchmark._ask_visual_seed(42) == 123456
    monkeypatch.setattr("builtins.input", lambda _: "42")
    assert benchmark._ask_visual_seed(99) == 42


def test_cli_handles_expected_error_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(menu, "main", lambda _argv=None: (_ for _ in ()).throw(
        FileNotFoundError("modello assente")
    ))
    assert menu.run_cli([]) == 1
    output = capsys.readouterr().out
    assert "modello assente" in output
    assert "Traceback" not in output


def test_cli_logs_unexpected_error_without_printing_traceback(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(menu, "ERROR_LOG_DIR", tmp_path)
    monkeypatch.setattr(menu, "main", lambda _argv=None: (_ for _ in ()).throw(
        KeyError("campo")
    ))
    assert menu.run_cli([]) == 1
    output = capsys.readouterr().out
    assert "Errore interno OSSERVA" in output
    assert "Traceback" not in output
    logs = list(tmp_path.glob("osserva_error_*.log"))
    assert len(logs) == 1
    assert "Traceback" in logs[0].read_text()


def test_cli_handles_keyboard_interrupt(monkeypatch, capsys):
    monkeypatch.setattr(menu, "main", lambda _argv=None: (_ for _ in ()).throw(
        KeyboardInterrupt()
    ))
    assert menu.run_cli([]) == 130
    assert "annullata" in capsys.readouterr().out
