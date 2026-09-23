"""Decisioni su dati pubblici, senza accesso all'ambiente o al simulatore."""

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from physical_ai_mujoco.contracts import Observation, ObjectDecision, PrivilegedState


class Decider(ABC):
    """Deployable (student) decider: it sees the Observation only."""

    @abstractmethod
    def decide(self, observation: Observation) -> ObjectDecision:
        raise NotImplementedError


class TeacherDecider(ABC):
    """Simulation-only decider: it also sees the privileged branch of OBSERVE.

    Kept apart from Decider on purpose: a student decider cannot receive
    PrivilegedState, because its interface does not accept it.
    """

    @abstractmethod
    def decide(self, observation: Observation, privileged: PrivilegedState) -> ObjectDecision:
        raise NotImplementedError


class RandomDecider(Decider):
    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def decide(self, observation):
        valid = [i for i, o in enumerate(observation.objects) if o.present]
        if not valid:
            raise ValueError("Nessun oggetto presente")
        return ObjectDecision(
            observation.objects[int(self.rng.choice(valid))].object_id
        )


class HighestObjectDecider(Decider):
    def decide(self, observation):
        valid = [o for o in observation.objects if o.present and o.position is not None]
        if not valid:
            raise ValueError("Nessun oggetto presente")
        return ObjectDecision(max(valid, key=lambda o: o.position[2]).object_id)


class ImmediateTargetDecider(RandomDecider):
    def decide(self, observation):
        for obj in observation.objects:
            if obj.is_target and obj.present:
                return ObjectDecision(obj.object_id)
        return super().decide(observation)


class PPODecider:
    """Teacher PPO storico su stato privilegiato, separato dal Decider operativo."""

    def __init__(self, model, normalizer):
        self.model = model
        self.normalizer = normalizer

    @classmethod
    def load(cls, path):
        path = Path(path)
        stats = path.with_name(f"{path.stem}_normalizzazione.pkl")
        if not stats.is_file():
            raise FileNotFoundError(
                f"Manca {stats.name}: statistiche di normalizzazione richieste."
            )
        from stable_baselines3 import PPO
        import pickle

        with stats.open("rb") as stream:
            normalizer = pickle.load(stream)
        return cls(PPO.load(path), normalizer)

    def predict_index(self, vector):
        expected = tuple(self.model.observation_space.shape)
        if np.asarray(vector).shape != expected:
            raise ValueError(
                f"Il modello richiede osservazioni {expected}, ricevute {np.asarray(vector).shape}. Controllare il numero di oggetti."
            )
        action, _ = self.model.predict(
            self.normalizer.normalize_obs(vector), deterministic=True
        )
        return int(action)

    def decide(self, observation):
        raise TypeError(
            "Il PPO storico e' un teacher su stato esatto; usare predict_index "
            "con il vettore privilegiato dell'environment."
        )
