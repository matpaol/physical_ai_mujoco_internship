"""Comando compatibile; implementazione in physical_ai_mujoco.experiments.rollout."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physical_ai_mujoco.experiments.rollout import *  # noqa: F401,F403

from physical_ai_mujoco.experiments.rollout import main as main

if __name__ == "__main__":
    raise SystemExit(main())
