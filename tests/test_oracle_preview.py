"""Preview tool of the synthetic OBSERVE pipeline (oracle observer)."""

import io
import random
from types import SimpleNamespace

import pytest

from physical_ai_mujoco.contracts import TaskContext
from physical_ai_mujoco.evaluation import oracle_preview
from physical_ai_mujoco.evaluation.oracle_preview import (
    OracleOutput, SceneSpec, describe, describe_privileged, random_scene, save_outputs,
)
from physical_ai_mujoco.observe import OracleObserver


def _stack_simulator():
    """Three boxes: target on the ground, 'middle' on it, 'top' on 'middle'."""
    names = ("target", "middle", "top")
    objects = tuple(
        SimpleNamespace(
            instance_id=name, type_id="box", shape="box",
            size={"x": 0.1, "y": 0.1, "z": 0.1}, mass=1.0, friction=(0.5, 0.01, 0.01),
            center_of_mass=(0.0, 0.0, 0.0),
        )
        for name in names
    )
    states = {
        name: SimpleNamespace(
            position=(0.0, 0.0, 0.05 + 0.1 * level), quaternion=(1.0, 0.0, 0.0, 0.0),
            linear_velocity=(0, 0, 0), angular_velocity=(0, 0, 0),
        )
        for level, name in enumerate(names)
    }
    return SimpleNamespace(
        scene=SimpleNamespace(objects=objects),
        time=1.0,
        is_present=lambda name: True,
        get_object_state=lambda name: states[name],
        support_graph=lambda: {"target": ["terreno"], "middle": ["target"], "top": ["middle"]},
    )


def test_random_scenes_are_reproducible_and_in_range():
    first = [random_scene(random.Random(5)) for _ in range(3)]
    again = [random_scene(random.Random(5)) for _ in range(3)]
    assert first == again
    low, high = oracle_preview.OBJECT_COUNT_RANGE
    specs = [random_scene(random.Random(seed)) for seed in range(50)]
    assert all(low <= spec.object_count <= high for spec in specs)
    assert len({spec.seed for spec in specs}) == len(specs)


def test_description_lists_objects_graph_and_what_rests_on_the_target():
    observation = OracleObserver().observe(_stack_simulator(), TaskContext("target", 0.0))
    text = describe(observation, SceneSpec(seed=1, object_count=3), scene_seed=2, index=1)
    assert "3 objects, target target" in text
    assert "mujoco_contacts_oracle, 2 contacts" in text
    assert "target       supports middle" in text
    assert "middle       supports top" in text
    assert "on the ground only: target" in text
    assert "nothing on top:     top" in text
    assert "on the target:      middle, top (through middle)" in text


def _stack_output():
    observer = OracleObserver()
    simulator = _stack_simulator()
    observation = observer.observe(simulator, TaskContext("target", 0.0))
    return OracleOutput(observation, observer.privileged_state(simulator, "target"), scene_seed=2)


def test_privileged_branch_is_shown_separately_and_matches_the_graph():
    output = _stack_output()
    text = describe_privileged(output.privileged, output.observation)
    assert text.startswith("PRIVILEGED STATE -> teacher / oracle / evaluation (simulation only)")
    assert "Contact supports: 2 pairs (same as the Observation graph)" in text
    assert "0.500/0.010/0.010" in text and "(+0.0, +0.0, +0.0)" in text
    assert "OBSERVATION -> DECIDE" in describe(output.observation, SceneSpec(1, 3), 2, 1)


def test_outputs_are_saved_as_json_and_support_graph(tmp_path):
    import json

    output = _stack_output()
    files = save_outputs(output, SceneSpec(seed=1, object_count=3), tmp_path / "scene_001")
    observation = json.loads(files["observation"].read_text(encoding="utf-8"))
    privileged = json.loads(files["privileged"].read_text(encoding="utf-8"))
    assert observation["scene_seed"] == 2
    assert observation["observation"]["scene"]["target_id"] == "target"
    assert len(observation["observation"]["relations"]["relations"]) == 2
    assert [obj["mass"] for obj in privileged["privileged_state"]["objects"]] == [1.0, 1.0, 1.0]
    assert privileged["privileged_state"]["contact_supports"] == [["middle", "top"], ["target", "middle"]]
    dot = files["graph_dot"].read_text(encoding="utf-8")
    assert '"target" -> "middle"' in dot and "TARGET" in dot


def test_observe_main_launches_the_preview():
    from physical_ai_mujoco.observe import main_test_oracle

    assert main_test_oracle.run_cli is oracle_preview.run_cli


def test_non_interactive_run_prints_one_real_oracle_scene(monkeypatch, capsys, tmp_path):
    pytest.importorskip("gymnasium")
    pytest.importorskip("mujoco")
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    # The tool activates the oracle profile process-wide: restore it afterwards.
    monkeypatch.setenv("PHYSICAL_AI_EXPERIMENT", "")
    assert oracle_preview.main(["--seed", "7", "--output", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "sequence seed 7" in output
    assert "mujoco_contacts_oracle" in output
    assert "Observation valid: yes" in output
    assert "(same as the Observation graph)" in output
    assert (tmp_path / "scene_001" / "observation.json").is_file()
    assert (tmp_path / "scene_001" / "privileged_state.json").is_file()
