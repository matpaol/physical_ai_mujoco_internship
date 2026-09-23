"""Interactive view of the full loop: scene -> OBSERVE -> DECIDE -> EXECUTE -> TASK.

The profile chooses every component (observer, decider, executor, task rules);
this module only builds the environment, runs ``run_episode`` and shows each
step. Scenes are random (seed and object count); nothing else is asked.

Entry point: ``main_pipeline.py`` in the project root.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import select
import sys

import numpy as np

from physical_ai_mujoco.decide import TeacherDecider
from physical_ai_mujoco.evaluation.oracle_preview import (
    SceneSpec,
    describe,
    describe_privileged,
    random_scene,
)
from physical_ai_mujoco.experiments.episode import EpisodeRecord, StepRecord, run_episode
from physical_ai_mujoco.infrastructure.builder import ComponentBuilder
from physical_ai_mujoco.infrastructure.experiment import ExperimentProfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = PROJECT_ROOT / "configs/experiments/oracolo.json"
VIEWER_REDRAW_SECONDS = 0.03


def make_env(spec: SceneSpec, *, viewer: bool):
    import gymnasium as gym
    import physical_ai_mujoco.envs  # noqa: F401  (registers the environment)

    return gym.make(
        "TargetExtraction-v0",
        object_count=spec.object_count,
        obs_mode="state",
        render_mode="human" if viewer else None,
        highlight_target=viewer,
        realtime_factor=1.0 if viewer else 0.0,
        disable_env_checker=True,
    )


def _read_line(prompt: str, env=None) -> str:
    """Reads a line; with the viewer open it keeps redrawing while waiting."""
    print(prompt, end="", flush=True)
    rendering = env is not None and env.unwrapped.render_mode == "human"
    while rendering:
        try:
            ready, _, _ = select.select([sys.stdin], [], [], VIEWER_REDRAW_SECONDS)
        except (OSError, ValueError):
            break
        if ready:
            break
        env.render()
    line = sys.stdin.readline()
    if not line:
        raise EOFError
    return line.strip().lower()


def _ask(prompt: str, default: bool, env=None) -> bool:
    answer = _read_line(prompt + (" [Y/n] " if default else " [y/N] "), env)
    return default if not answer else answer.startswith("y")


def describe_step(step: StepRecord, decider_label: str) -> str:
    observation = step.observation
    relations = observation.relations
    graph = relations.dependency_graph
    target = observation.scene.target_id
    free = [object_id for object_id, uppers in graph.items() if not uppers]
    on_target = list(graph.get(target, ())) if target else []
    role = next(
        (obj.role for obj in observation.scene.objects if obj.object_id == step.decision.object_id),
        "?",
    )
    execution = step.execution
    if execution is None:
        executed = "not executed (invalid action)"
    elif execution.removed:
        executed = "removed"
    else:
        executed = f"failed ({execution.failure_reason})"
    lines = [
        f"Step {step.index}",
        f"  OBSERVE  {len(observation.scene.objects)} objects, "
        f"{len(relations.relations)} supports; nothing on top: {', '.join(free) or '-'}; "
        f"directly on the target: {', '.join(on_target) or 'nothing'}",
        f"  DECIDE   {decider_label} -> {step.decision.object_id} ({role})"
        + ("   [privileged state given]" if step.privileged_given_to_decider else ""),
        f"  EXECUTE  {executed}",
        f"  TASK     reward {step.reward:+.3f}, disturbance {1000 * step.disturbance:.1f} mm, "
        f"collapse {'YES' if step.collapsed else 'no'}, "
        f"target removed {'yes' if step.target_removed else 'no'}",
    ]
    return "\n".join(lines)


def describe_episode(episode: EpisodeRecord) -> str:
    outcome = "SUCCESS" if episode.success else "no success"
    if episode.stopped_early:
        outcome = "stopped by the user"
    return "\n".join([
        f"Episode: {outcome} in {len(episode.steps)} steps, "
        f"collapse during the episode: {'yes' if episode.collapsed_ever else 'no'}, "
        f"total disturbance {1000 * episode.total_disturbance:.1f} mm",
        f"  removal order: {' -> '.join(episode.removal_order) or '-'} (target {episode.target_id})",
    ])


class _StepView:
    """on_step callback: prints each step and lets the user drive the episode."""

    HELP = "[Enter] next step · a = run to the end · o = full observation · q = stop episode: "

    def __init__(self, env, decider_label: str, spec: SceneSpec, interactive: bool):
        self.env = env
        self.decider_label = decider_label
        self.spec = spec
        self.auto = not interactive

    def __call__(self, step: StepRecord) -> bool:
        print(describe_step(step, self.decider_label))
        if self.auto or step.terminated or step.truncated:
            return True
        while True:
            answer = _read_line("  " + self.HELP, self.env)
            if answer == "o":
                print()
                print(describe(step.observation, self.spec, -1, step.index))
                print()
                print(describe_privileged(step.privileged, step.observation))
                print()
                continue
            if answer == "a":
                self.auto = True
            return answer != "q"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE,
                        help="experiment profile (default: configs/experiments/oracolo.json)")
    parser.add_argument("--seed", type=int, help="seed of the random sequence, to replay it")
    args = parser.parse_args(argv)

    profile = ExperimentProfile.load(args.profile)
    profile.activate()
    builder = ComponentBuilder()
    components = builder.load().env.get("components", {})
    sequence_seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)
    rng = random.Random(sequence_seed)
    interactive = sys.stdin.isatty()

    print(f"\nPipeline: {profile.name}")
    print(f"  observer {components.get('observer', 'exact')} · decider {profile.default_decider} · "
          f"executor {components.get('executor', 'ideal_removal')}")
    print(f"  sequence seed {sequence_seed} (--seed to replay)\n")

    scene_index = 0
    while True:
        scene_index += 1
        spec = random_scene(rng)
        decider = builder.decider(profile.default_decider, np.random.default_rng(spec.seed))
        kind = "teacher" if isinstance(decider, TeacherDecider) else "student"
        label = f"{profile.default_decider} ({kind})"
        viewer = interactive and _ask(
            f"Scene {scene_index} ({spec.object_count} objects): watch it in the MuJoCo viewer?",
            default=False,
        )
        env = make_env(spec, viewer=viewer)
        try:
            view = _StepView(env, label, spec, interactive)
            episode = run_episode(env, decider, seed=spec.seed, on_step=view)
            print(describe_episode(episode))
            if not interactive:
                return 0
            if not _ask("\nNext random scene?", default=True, env=env):
                return 0
        finally:
            env.close()
        print()


def run_cli(argv: list[str] | None = None) -> int:
    """Command-line boundary: Ctrl+C or Ctrl+D end the session without a traceback."""
    try:
        return main(argv)
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
        return 130


if __name__ == "__main__":
    raise SystemExit(run_cli())
