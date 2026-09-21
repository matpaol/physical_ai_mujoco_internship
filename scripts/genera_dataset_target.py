"""Comando leggero: genera un dataset del visore da una ricetta.

    python -m scripts.genera_dataset_target configs/vision_training/dataset_sim_dr_v1.json
"""

from physical_ai_mujoco.vision_training.dataset import main


if __name__ == "__main__":
    raise SystemExit(main())
