"""Environment Gymnasium del progetto (Phase 0B)."""

from __future__ import annotations

import json
from pathlib import Path

from gymnasium.envs.registration import register

from physical_ai_mujoco.envs.target_extraction import TargetExtractionEnv

_ENV_CONFIG = Path(__file__).resolve().parents[2] / "configs/phase_0b/env.json"
_MAX_STEPS = json.loads(_ENV_CONFIG.read_text(encoding="utf-8"))["task"][
    "max_episode_steps"
]

register(
    id="TargetExtraction-v0",
    entry_point="physical_ai_mujoco.envs.target_extraction:TargetExtractionEnv",
    max_episode_steps=_MAX_STEPS,
)

__all__ = ["TargetExtractionEnv"]
