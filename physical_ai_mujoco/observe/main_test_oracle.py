"""Main of the synthetic OBSERVE pipeline: oracle observation on random scenes.

    python physical_ai_mujoco/observe/main_test_oracle.py
    python physical_ai_mujoco/observe/main_test_oracle.py --seed 1234   # replay

Every scene is random; the tool only asks whether to open the viewer and
whether to go on. The logic lives in ``evaluation/oracle_preview.py``.
"""

from pathlib import Path
import sys

# Launching by file path does not put the project root on sys.path.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from physical_ai_mujoco.evaluation.oracle_preview import run_cli

if __name__ == "__main__":
    raise SystemExit(run_cli())
