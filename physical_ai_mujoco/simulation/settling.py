"""Assestamento di Fase 0A.

Il criterio di quiete vive in `Simulator.step_until_settled`, che e' anche
quello che usa l'environment di Fase 0B: una sola definizione di "la scena e'
ferma", condivisa da entrambe le fasi. Qui resta solo la forma del risultato
attesa da Fase 0A.
"""

from __future__ import annotations

from dataclasses import dataclass

from physical_ai_mujoco.simulation.simulator import ObjectState, Simulator


@dataclass(frozen=True)
class SettlingResult:
    settled: bool
    simulation_time: float
    steps: int
    object_states: dict[str, ObjectState]


def run_until_settled(simulator: Simulator) -> SettlingResult:
    outcome = simulator.step_until_settled()
    return SettlingResult(
        settled=outcome.settled,
        simulation_time=simulator.time,
        steps=outcome.steps,
        object_states=simulator.get_object_states(),
    )
