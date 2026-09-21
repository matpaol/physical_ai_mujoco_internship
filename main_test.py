"""Menu unico dei test: python main_test.py oppure --suite osserva."""

from __future__ import annotations

import argparse
import subprocess
import sys

from tests.suites import SUITES, suite_paths, unassigned_test_files


def run_suite(name: str) -> int:
    missing = unassigned_test_files()
    if missing:
        print(
            "Attenzione: test non assegnati a un modulo: " + ", ".join(missing)
        )
        if name != "tutti":
            return 2
    paths = suite_paths(name)
    print(f"\nEseguo modulo {name}: {', '.join(path.name for path in paths)}\n", flush=True)
    try:
        return subprocess.call(
            [sys.executable, "-m", "pytest", "-q", *(str(path) for path in paths)]
        )
    except KeyboardInterrupt:
        print("\nTest interrotti.")
        return 130


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Menu dei moduli di test")
    parser.add_argument("--suite", choices=("tutti", *SUITES))
    parser.add_argument("--list", action="store_true", help="Elenca i moduli")
    args = parser.parse_args(argv)
    if args.list:
        for name, paths in SUITES.items():
            print(f"{name}: {', '.join(paths)}")
        return 0
    if args.suite:
        return run_suite(args.suite)
    names = ("tutti", *SUITES)
    print("\nTest del progetto\n")
    for index, name in enumerate(names, 1):
        print(f"  {index}. {name}")
    print("  0. Esci")
    while True:
        answer = input("Scelta [1]: ").strip() or "1"
        if answer == "0":
            return 0
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return run_suite(names[int(answer) - 1])
        print("Scegli un numero del menu.")


if __name__ == "__main__":
    raise SystemExit(main())
