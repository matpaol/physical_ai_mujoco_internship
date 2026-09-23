"""One episode of the full loop: scene -> OBSERVE -> DECIDE -> EXECUTE -> TASK.

The environment owns the world (scene, simulation, execution, task rules);
this module only connects it to a decider, step after step, and records what
happened. It is the single loop used by the pipeline main, and meant to be
reused by training and evaluation so that there is no second loop to keep in
sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from physical_ai_mujoco.contracts import (
    ExecutionOutcome,
    ObjectDecision,
    Observation,
    PrivilegedState,
)
from physical_ai_mujoco.decide import Decider, TeacherDecider


@dataclass(frozen=True)
class StepRecord:
    """Everything that happened in one step of the loop."""

    index: int
    observation: Observation            # what DECIDE saw
    privileged: PrivilegedState         # simulator truth at the same moment
    privileged_given_to_decider: bool   # True only for teacher deciders
    decision: ObjectDecision
    execution: ExecutionOutcome | None  # None when the action was not valid
    reward: float
    disturbance: float
    collapsed: bool
    target_removed: bool
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class EpisodeRecord:
    scene_seed: int
    target_id: str
    steps: tuple[StepRecord, ...]
    success: bool
    collapsed_ever: bool
    stopped_early: bool

    @property
    def removal_order(self) -> tuple[str, ...]:
        return tuple(
            step.decision.object_id
            for step in self.steps
            if step.execution is not None and step.execution.removed
        )

    @property
    def total_disturbance(self) -> float:
        return sum(step.disturbance for step in self.steps)


def decide(decider, observation: Observation, privileged: PrivilegedState) -> tuple[ObjectDecision, bool]:
    """Calls the decider with the inputs its interface allows."""
    if isinstance(decider, TeacherDecider):
        return decider.decide(observation, privileged), True
    if isinstance(decider, Decider):
        return decider.decide(observation), False
    raise TypeError(f"Not a Decider or TeacherDecider: {type(decider).__name__}")


def run_episode(
    env,
    decider,
    *,
    seed: int | None = None,
    on_step: Callable[[StepRecord], bool] | None = None,
) -> EpisodeRecord:
    """Runs one episode.

    ``on_step`` is called after every step (the last one included); returning
    False stops the episode early.
    """
    _, info = env.reset(seed=seed)
    base = env.unwrapped
    scene_seed = int(info["scene_seed"])
    steps: list[StepRecord] = []
    stopped_early = False
    while True:
        observation = base.decision_observation(refresh=True)
        privileged = base.privileged_state()
        decision, given = decide(decider, observation, privileged)
        _, reward, terminated, truncated, info = env.step(base.action_index(decision))
        step = StepRecord(
            index=len(steps) + 1,
            observation=observation,
            privileged=privileged,
            privileged_given_to_decider=given,
            decision=decision,
            execution=info.get("execution"),
            reward=float(reward),
            disturbance=float(info.get("disturbance_step", 0.0)),
            collapsed=bool(info.get("collapsed", False)),
            target_removed=bool(info.get("target_removed", False)),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        steps.append(step)
        keep_going = True if on_step is None else on_step(step) is not False
        if terminated or truncated:
            break
        if not keep_going:
            stopped_early = True
            break
    return EpisodeRecord(
        scene_seed=scene_seed,
        target_id=base.target_id,
        steps=tuple(steps),
        success=bool(info.get("is_success", False)),
        collapsed_ever=bool(info.get("collapsed_ever", False)),
        stopped_early=stopped_early,
    )
