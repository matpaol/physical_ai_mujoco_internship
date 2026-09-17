"""Regole scientifiche del task, indipendenti dal backend fisico."""

from copy import deepcopy
import numpy as np
from physical_ai_mujoco.contracts import TaskOutcome


class TargetExtractionTask:
    def __init__(self, config, terminate_on_target=None, terminate_on_collapse=None):
        self.config = dict(config)
        self.terminate_on_target = bool(
            config.get("terminate_on_target", True)
            if terminate_on_target is None
            else terminate_on_target
        )
        self.terminate_on_collapse = bool(
            config.get("terminate_on_collapse", True)
            if terminate_on_collapse is None
            else terminate_on_collapse
        )
        self.reset({}, None)

    def select_target(self, object_ids, rng):
        return object_ids[int(rng.integers(0, max(len(object_ids) - 1, 1)))]

    def reset(self, positions, target_id):
        self.target_id = target_id
        self.reference_positions = {
            k: np.asarray(v).copy() for k, v in positions.items()
        }
        self.total_disturbance = 0.0
        self.ever_collapsed = False

    def disturbance(self, positions):
        maximum = 0.0
        for name, position in positions.items():
            if name == self.target_id:
                continue
            reference = self.reference_positions.get(name)
            if reference is not None:
                maximum = max(
                    maximum, float(np.linalg.norm(np.asarray(position) - reference))
                )
        return maximum

    def invalid_reward(self):
        return -float(self.config["removal_cost"])

    def evaluate(self, execution, positions):
        disturbance = self.disturbance(positions)
        self.total_disturbance += disturbance
        self.reference_positions = {
            k: np.asarray(v).copy() for k, v in positions.items()
        }
        target_removed = self.target_id not in positions
        collapsed = disturbance > float(self.config["disturbance_threshold"])
        self.ever_collapsed = self.ever_collapsed or collapsed
        empty = not positions
        reward = -float(self.config["removal_cost"])
        reward -= float(self.config["disturbance_penalty"]) * disturbance
        just_removed = execution.object_id == self.target_id and execution.removed
        if just_removed:
            reward += float(self.config["target_reward"])
        if collapsed:
            reward -= float(self.config["failure_penalty"])
        terminated = empty
        if target_removed and self.terminate_on_target:
            terminated = True
        if collapsed and self.terminate_on_collapse:
            terminated = True
        return TaskOutcome(
            reward,
            terminated,
            disturbance,
            target_removed,
            just_removed,
            collapsed,
            self.ever_collapsed,
            empty,
            bool(target_removed and not self.ever_collapsed),
        )

    def snapshot(self):
        return deepcopy(
            (
                self.target_id,
                self.reference_positions,
                self.total_disturbance,
                self.ever_collapsed,
            )
        )

    def restore(self, snapshot):
        (
            self.target_id,
            self.reference_positions,
            self.total_disturbance,
            self.ever_collapsed,
        ) = deepcopy(snapshot)
