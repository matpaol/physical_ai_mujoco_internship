"""Esecuzione della richiesta: prima implementazione ideale, senza robot."""

from abc import ABC, abstractmethod
from physical_ai_mujoco.contracts import ObjectDecision, ExecutionOutcome


class Executor(ABC):
    @abstractmethod
    def execute(self, decision: ObjectDecision, simulator) -> ExecutionOutcome:
        raise NotImplementedError


class IdealRemovalExecutor(Executor):
    def execute(self, decision, simulator):
        if not simulator.is_present(decision.object_id):
            return ExecutionOutcome(decision.object_id, False, "already_removed")
        simulator.remove_object(decision.object_id)
        return ExecutionOutcome(decision.object_id, True)
