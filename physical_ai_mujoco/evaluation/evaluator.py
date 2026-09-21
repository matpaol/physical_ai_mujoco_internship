"""Valutazione su seed espliciti, con una nuova scena per episodio."""

from dataclasses import dataclass
import numpy as np
from physical_ai_mujoco.decide import Decider, PPODecider


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    actions: tuple[str, ...]
    reward: float
    disturbance: float
    success: bool
    truncated: bool


class PolicyEvaluator:
    def evaluate(self, env, decider, seeds):
        results = []
        for seed in seeds:
            observation, info = env.reset(seed=int(seed))
            rng = np.random.default_rng(seed)
            actions = []
            total = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                if isinstance(decider, PPODecider):
                    action = decider.predict_index(env.unwrapped._state_observation())
                elif isinstance(decider, Decider):
                    decision = decider.decide(env.unwrapped.decision_observation())
                    action = env.unwrapped.action_index(decision)
                else:
                    action = int(decider(env, info, rng, observation))
                if env.unwrapped.obs_mode == "sensor":
                    slots = env.unwrapped._latest_encoded.slot_ids
                    selected = slots[action]
                else:
                    selected = env.unwrapped.object_ids[action]
                observation, reward, terminated, truncated, info = env.step(action)
                actions.append(info.get("selected_object_id") or selected or f"slot_{action}")
                total += reward
            results.append(
                EpisodeResult(
                    int(seed),
                    tuple(actions),
                    total,
                    float(info["disturbance_total"]),
                    bool(info.get("is_success", False)),
                    bool(truncated),
                )
            )
        return results
