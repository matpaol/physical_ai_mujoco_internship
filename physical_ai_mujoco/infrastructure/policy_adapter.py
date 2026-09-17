"""Ponte tra i callback legacy degli script e DECIDE a oggetti."""

from pathlib import Path
from physical_ai_mujoco.decide import (
    RandomDecider,
    HighestObjectDecider,
    ImmediateTargetDecider,
    PPODecider,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CARTELLA_MODELLI = PROJECT_ROOT / "outputs/modelli"


class PolicyAdapter:
    def __init__(self, decider):
        self.decider = decider

    def __call__(self, env, info, rng, osservazione=None):
        if isinstance(self.decider, PPODecider) and osservazione is not None:
            vector = (
                osservazione["state"]
                if isinstance(osservazione, dict)
                else osservazione
            )
            return self.decider.predict_index(vector)
        observation = env.unwrapped.decision_observation()
        return env.unwrapped.action_index(self.decider.decide(observation))


def casuale(env, info, rng, osservazione=None):
    return PolicyAdapter(RandomDecider(rng))(env, info, rng, osservazione)


def alto(env, info, rng, osservazione=None):
    return PolicyAdapter(HighestObjectDecider())(env, info, rng, osservazione)


def target(env, info, rng, osservazione=None):
    return PolicyAdapter(ImmediateTargetDecider(rng))(env, info, rng, osservazione)


def carica(percorso):
    return PolicyAdapter(PPODecider.load(percorso))


def modelli_disponibili():
    return sorted(
        CARTELLA_MODELLI.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True
    )


RIFERIMENTI = {"casuale": casuale, "alto": alto}
