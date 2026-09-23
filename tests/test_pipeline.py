"""Full loop scene -> OBSERVE -> DECIDE -> EXECUTE -> TASK (oracle profile)."""

import io

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")

import physical_ai_mujoco.envs  # noqa: E402,F401  (registers the environment)
from physical_ai_mujoco.contracts import ExecutionOutcome, ObjectDecision, PrivilegedState  # noqa: E402
from physical_ai_mujoco.decide import HighestObjectDecider, RandomDecider, TeacherDecider  # noqa: E402
from physical_ai_mujoco.experiments import pipeline_main  # noqa: E402
from physical_ai_mujoco.experiments.episode import decide, run_episode  # noqa: E402
from physical_ai_mujoco.infrastructure.builder import ComponentBuilder  # noqa: E402
from physical_ai_mujoco.observe import OracleObserver  # noqa: E402


def _oracle_env(object_count=5):
    return gym.make("TargetExtraction-v0", disable_env_checker=True, obs_mode="state",
                    object_count=object_count, observer=OracleObserver())


class _RecordingTeacher(TeacherDecider):
    """Removes the heaviest present object; records what it was given."""

    def __init__(self):
        self.inputs = []

    def decide(self, observation, privileged):
        self.inputs.append((observation, privileged))
        present = {obj.object_id for obj in observation.scene.objects}
        heaviest = max((o for o in privileged.objects if o.object_id in present), key=lambda o: o.mass)
        return ObjectDecision(heaviest.object_id)


def test_teacher_receives_privileged_state_student_does_not():
    env = _oracle_env()
    try:
        teacher = _RecordingTeacher()
        episode = run_episode(env, teacher, seed=3)
        assert teacher.inputs and all(isinstance(p, PrivilegedState) for _, p in teacher.inputs)
        assert all(step.privileged_given_to_decider for step in episode.steps)

        student = run_episode(env, RandomDecider(np.random.default_rng(0)), seed=3)
        assert not any(step.privileged_given_to_decider for step in student.steps)
        # The privileged branch is still recorded, for evaluation and teacher-student data.
        assert all(isinstance(step.privileged, PrivilegedState) for step in student.steps)
    finally:
        env.close()


def test_episode_records_every_stage_and_ends_with_the_task():
    env = _oracle_env()
    try:
        episode = run_episode(env, HighestObjectDecider(), seed=3)
        last = episode.steps[-1]
        assert last.terminated or last.truncated
        assert all(isinstance(step.execution, ExecutionOutcome) for step in episode.steps)
        assert episode.removal_order == tuple(step.decision.object_id for step in episode.steps)
        assert all(step.observation.relations.estimator == "mujoco_contacts_oracle"
                   for step in episode.steps)
        # The object removed at step k is gone from the observation of step k + 1.
        for before, after in zip(episode.steps, episode.steps[1:]):
            assert before.decision.object_id not in {o.object_id for o in after.observation.scene.objects}
        assert episode.success == (last.target_removed and not episode.collapsed_ever)
    finally:
        env.close()


def test_on_step_can_stop_the_episode_early():
    env = _oracle_env()
    try:
        episode = run_episode(env, RandomDecider(np.random.default_rng(1)), seed=3,
                              on_step=lambda step: False)
        assert len(episode.steps) == 1
        assert episode.stopped_early != (episode.steps[0].terminated or episode.steps[0].truncated)
    finally:
        env.close()


def test_decide_refuses_objects_that_are_not_deciders():
    with pytest.raises(TypeError, match="Decider"):
        decide(object(), None, None)


def test_builder_builds_deciders_named_by_the_profile():
    builder = ComponentBuilder()
    assert isinstance(builder.decider("random", np.random.default_rng(0)), RandomDecider)
    assert isinstance(builder.decider("highest"), HighestObjectDecider)
    with pytest.raises(ValueError, match="not available"):
        builder.decider("ppo")


def test_pipeline_main_runs_one_episode_without_a_terminal(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    # The main activates the profile process-wide: restore it afterwards.
    monkeypatch.setenv("PHYSICAL_AI_EXPERIMENT", "")
    assert pipeline_main.main(["--seed", "7"]) == 0
    output = capsys.readouterr().out
    assert "observer oracle · decider random · executor ideal_removal" in output
    assert "Step 1" in output and "DECIDE   random (student)" in output
    assert "Episode:" in output and "removal order:" in output
