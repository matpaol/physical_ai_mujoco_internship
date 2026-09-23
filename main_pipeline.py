"""Full loop, step by step: python main_pipeline.py [--profile PATH] [--seed N].

scene -> OBSERVE -> DECIDE -> EXECUTE -> TASK, with every component chosen by
the experiment profile (default: configs/experiments/oracolo.json).
"""

from physical_ai_mujoco.experiments.pipeline_main import run_cli

if __name__ == "__main__":
    raise SystemExit(run_cli())
