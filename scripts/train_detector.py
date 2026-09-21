"""Comando leggero: addestra il visore da una ricetta.

    python -m scripts.train_detector configs/vision_training/training_sim_dr_v1_m3.json
"""

from physical_ai_mujoco.vision_training.training import main


if __name__ == "__main__":
    raise SystemExit(main())
