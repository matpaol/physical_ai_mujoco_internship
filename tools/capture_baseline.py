"""Uso: python tools/capture_baseline.py RADICE_PROGETTO FILE_OUTPUT.json.

Registra 60 episodi e tre scene 0A, senza rendering. Confrontare con stesso
interprete e stesse librerie; esclude tempi reali e immagini.
"""

import sys, json
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import numpy as np
import gymnasium as gym
import physical_ai_mujoco.envs
from scripts.run_phase_0a import create_scene
from dataclasses import asdict


def clean(x):
    if isinstance(x, dict):
        return {
            k: clean(v) for k, v in x.items() if k not in ("frames", "wall_seconds")
        }
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


result = {"phase0a": [asdict(create_scene(i)) for i in [0, 42, 67]], "episodes": []}
for options in [
    {},
    {"terminate_on_target": False, "terminate_on_collapse": False},
    {"terminate_on_target": True, "terminate_on_collapse": True},
    {"resample_shapes": False},
    {"scene_pool_size": 2, "fresh_scene_probability": 0.0},
]:
    for policy in ["random", "top", "target"]:
        env = gym.make("TargetExtraction-v0", object_count=6, **options)
        try:
            for seed in range(4):
                obs, info = env.reset(seed=seed)
                trace = [clean((obs, info))]
                rng = np.random.default_rng(seed)
                for step in range(64):
                    valid = np.flatnonzero(info["action_mask"])
                    if policy == "top":
                        sim = env.unwrapped.simulator
                        action = int(
                            max(
                                valid,
                                key=lambda a: sim.get_object_state(
                                    sim.scene.objects[a].instance_id
                                ).position[2],
                            )
                        )
                    elif policy == "target":
                        action = (
                            int(info["target_index"])
                            if info["target_index"] in valid
                            else int(rng.choice(valid))
                        )
                    else:
                        action = int(rng.choice(valid))
                    obs, reward, term, trunc, info = env.step(action)
                    trace.append(clean((action, obs, reward, term, trunc, info)))
                    if term or trunc:
                        break
                    if step == 0:
                        trace.append(clean(("invalid", env.step(action))))
                result["episodes"].append(
                    dict(options=options, policy=policy, seed=seed, trace=trace)
                )
        finally:
            env.close()
Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[2]).write_text(json.dumps(clean(result), sort_keys=True), encoding="utf-8")
print(len(result["episodes"]), "episodes captured")
