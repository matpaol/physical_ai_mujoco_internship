"""Preview of the synthetic OBSERVE pipeline on random scenes.

Every scene is random (seed and object count). For each one the tool produces
the two outputs of OBSERVE in simulation:

- the ``Observation`` given to DECIDE (scene, support graph, uncertainty);
- the ``PrivilegedState`` (simulation only: teacher, oracle, evaluation):
  poses, velocities, mass, the three friction coefficients, centre of mass,
  true shape and size, contact supports.

It prints both, saves them as JSON together with the support graph (DOT, and
SVG when Graphviz is installed), then asks whether to open the graph image and
the MuJoCo viewer, and whether to continue. There is nothing to configure: the
observer comes from the ``configs/experiments/oracolo.json`` profile.

Entry point: ``physical_ai_mujoco/observe/main_test_oracle.py`` (the main of
the synthetic OBSERVE module); running this file directly works too. This file
holds the logic because it builds the Gymnasium environment, which
``observe/`` must not import.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import random
import select
import shutil
import subprocess
import sys
import webbrowser

# Launching by file path does not put the project root on sys.path.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from physical_ai_mujoco.contracts import Observation, PrivilegedState, validate_observation
from physical_ai_mujoco.infrastructure.experiment import ExperimentProfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORACLE_PROFILE = PROJECT_ROOT / "configs/experiments/oracolo.json"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/observe_tests"
OBJECT_COUNT_RANGE = (4, 10)
VIEWER_REDRAW_SECONDS = 0.03


@dataclass(frozen=True)
class SceneSpec:
    """What is needed to rebuild the same random scene."""

    seed: int
    object_count: int


@dataclass(frozen=True)
class OracleOutput:
    """The two OBSERVE outputs for one scene."""

    observation: Observation
    privileged: PrivilegedState
    scene_seed: int


def random_scene(rng: random.Random) -> SceneSpec:
    low, high = OBJECT_COUNT_RANGE
    return SceneSpec(seed=rng.randrange(2**31), object_count=rng.randint(low, high))


# ------------------------------------------------------------------ scene

def _make_env(spec: SceneSpec, *, viewer: bool):
    import gymnasium as gym
    import physical_ai_mujoco.envs  # noqa: F401  (registers the environment)

    ExperimentProfile.load(ORACLE_PROFILE).activate()
    return gym.make(
        "TargetExtraction-v0",
        object_count=spec.object_count,
        obs_mode="state",
        render_mode="human" if viewer else None,
        highlight_target=viewer,
        realtime_factor=0.0,
        disable_env_checker=True,
    )


def observe_scene(spec: SceneSpec) -> OracleOutput:
    """Builds the settled scene and returns Observation and PrivilegedState."""
    from physical_ai_mujoco.observe import OracleObserver

    env = _make_env(spec, viewer=False)
    try:
        _, info = env.reset(seed=spec.seed)
        base = env.unwrapped
        if not isinstance(base.observer, OracleObserver):
            raise RuntimeError(
                f"{ORACLE_PROFILE.name} did not build an OracleObserver "
                f"(got {type(base.observer).__name__})"
            )
        observation = base.decision_observation(refresh=True)
        validate_observation(observation)
        privileged = base.observer.privileged_state(base.simulator, base.target_id)
        return OracleOutput(observation, privileged, int(info["scene_seed"]))
    finally:
        env.close()


# ------------------------------------------------------------ description

def _resting_on(observation: Observation, object_id: str) -> list[tuple[str, str]]:
    """(object, through) pairs resting directly or indirectly on ``object_id``."""
    graph = observation.relations.dependency_graph
    found: list[tuple[str, str]] = []
    frontier = [(upper, object_id) for upper in graph.get(object_id, ())]
    seen = set()
    while frontier:
        current, through = frontier.pop(0)
        if current in seen:
            continue
        seen.add(current)
        found.append((current, through))
        frontier.extend((upper, current) for upper in graph.get(current, ()))
    return found


def _vector(values, digits=3) -> str:
    if values is None:
        return "-"
    return "(" + ", ".join(f"{value:+.{digits}f}" for value in values) + ")"


def _size(values) -> str:
    return "-" if values is None else " x ".join(f"{value:.3f}" for value in values)


def describe(observation: Observation, spec: SceneSpec, scene_seed: int, index: int) -> str:
    """The Observation given to DECIDE: objects, support graph, uncertainty."""
    scene = observation.scene
    relations = observation.relations
    lines = [
        f"Scene {index}: seed {spec.seed} (scene seed {scene_seed}), "
        f"{len(scene.objects)} objects, target {scene.target_id}",
        "",
        "OBSERVATION -> DECIDE",
        f"  {'id':<12} {'type':<22} {'role':<9} {'position [m]':<28} size [m]",
    ]
    for obj in scene.objects:
        lines.append(
            f"  {obj.object_id:<12} {str(obj.type_id):<22} {obj.role:<9} "
            f"{_vector(obj.position):<28} {_size(obj.size)}"
        )

    graph = relations.dependency_graph
    supported = {relation.target_id for relation in relations.relations}
    lines += [
        "",
        f"  Support graph ({relations.estimator}, {len(relations.relations)} contacts, "
        "lower -> upper):",
    ]
    for object_id, uppers in graph.items():
        if uppers:
            lines.append(f"    {object_id:<12} supports {', '.join(uppers)}")
    ground_only = [obj.object_id for obj in scene.objects if obj.object_id not in supported]
    lines.append(f"    on the ground only: {', '.join(ground_only) or '-'}")
    free = [object_id for object_id, uppers in graph.items() if not uppers]
    lines.append(f"    nothing on top:     {', '.join(free) or '-'}")

    if scene.target_id is not None:
        above = _resting_on(observation, scene.target_id)
        if above:
            text = ", ".join(
                item if through == scene.target_id else f"{item} (through {through})"
                for item, through in above
            )
            lines.append(f"    on the target:      {text}")
        else:
            lines.append("    on the target:      nothing, it is free")

    uncertainty = observation.uncertainty
    visible = sum(1 for item in uncertainty.objects if item.visible)
    lines += [
        "",
        f"  Uncertainty ({uncertainty.provider}): {visible}/{len(uncertainty.objects)} "
        f"objects visible, unknown space: {'yes' if uncertainty.unknown_space else 'no'}",
        "  Observation valid: yes (validate_observation)",
    ]
    return "\n".join(lines)


def describe_privileged(privileged: PrivilegedState, observation: Observation) -> str:
    """The simulation-only branch: never given to the deployable DECIDE."""
    lines = [
        "PRIVILEGED STATE -> teacher / oracle / evaluation (simulation only)",
        f"  {'id':<12} {'shape':<8} {'mass [kg]':>9}  {'friction slide/tors/roll':<25} "
        f"{'CoM offset [mm]':<22} {'speed [mm/s]':>12}  present  target",
    ]
    for obj in privileged.objects:
        speed = 1000 * sum(value * value for value in obj.linear_velocity) ** 0.5
        friction = "/".join(
            "-" if value is None else f"{value:.3f}"
            for value in (obj.friction, obj.torsional_friction, obj.rolling_friction)
        )
        com = "-" if obj.center_of_mass is None else "(" + ", ".join(
            f"{1000 * value:+.1f}" for value in obj.center_of_mass
        ) + ")"
        lines.append(
            f"  {obj.object_id:<12} {str(obj.shape):<8} {obj.mass:>9.3f}  {friction:<25} "
            f"{com:<22} {speed:>12.2f}  {'yes' if obj.present else 'no ':<7}  "
            f"{'yes' if obj.is_target else ''}"
        )
    graph_edges = {(r.source_id, r.target_id) for r in observation.relations.relations}
    same = set(privileged.contact_supports) == graph_edges
    lines.append(
        f"  Contact supports: {len(privileged.contact_supports)} pairs "
        f"({'same as' if same else 'DIFFERENT from'} the Observation graph)"
    )
    return "\n".join(lines)


# ------------------------------------------------------------------ files

def _json_default(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def support_graph_dot(observation: Observation, title: str) -> str:
    """Graphviz description of the support graph; target and free objects marked."""
    scene = observation.scene
    graph = observation.relations.dependency_graph

    def quote(text: str) -> str:
        return json.dumps(text, ensure_ascii=False)

    lines = [
        "digraph support_graph {",
        '  graph [rankdir=BT, fontname="Helvetica", labelloc=t];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fillcolor="#ffffff"];',
        f"  label={quote(title)};",
        '  ground [label="ground", shape=plaintext, style=""];',
    ]
    supported = {relation.target_id for relation in observation.relations.relations}
    for obj in scene.objects:
        fill = "#fff2b3" if obj.is_target else "#ffffff"
        border = ', color="#208050", penwidth=2' if not graph.get(obj.object_id) else ""
        label = f"{obj.object_id}\n{obj.type_id}" + ("\nTARGET" if obj.is_target else "")
        lines.append(f"  {quote(obj.object_id)} [label={quote(label)}, fillcolor=\"{fill}\"{border}];")
        if obj.object_id not in supported:
            lines.append(f'  ground -> {quote(obj.object_id)} [color="#aaaaaa"];')
    for relation in observation.relations.relations:
        lines.append(f"  {quote(relation.source_id)} -> {quote(relation.target_id)};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def save_outputs(output: OracleOutput, spec: SceneSpec, directory: Path) -> dict[str, Path]:
    """Writes observation.json, privileged_state.json and the support graph."""
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "observation": directory / "observation.json",
        "privileged": directory / "privileged_state.json",
        "graph_dot": directory / "support_graph.dot",
    }
    header = {"seed": spec.seed, "object_count": spec.object_count, "scene_seed": output.scene_seed}
    files["observation"].write_text(
        json.dumps({**header, "observation": asdict(output.observation)},
                   indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    files["privileged"].write_text(
        json.dumps({**header, "privileged_state": asdict(output.privileged)},
                   indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    title = f"Scene seed {output.scene_seed} - support graph (lower -> upper)"
    files["graph_dot"].write_text(support_graph_dot(output.observation, title), encoding="utf-8")
    dot = shutil.which("dot")
    if dot:
        svg = directory / "support_graph.svg"
        subprocess.run([dot, "-Tsvg", str(files["graph_dot"]), "-o", str(svg)], check=True)
        files["graph_svg"] = svg
    return files


# ------------------------------------------------------------ interaction

def _ask(prompt: str, default: bool) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    answer = input(prompt + suffix).strip().lower()
    return default if not answer else answer.startswith("y")


def _wait_while_rendering(env) -> None:
    """Keeps the viewer responsive until Enter is pressed in the terminal.

    A plain ``input()`` would block the window event loop and macOS would mark
    the viewer as not responding. Where ``select`` on stdin is unavailable
    (Windows) the window freezes until Enter, but the tool keeps working.
    """
    print("  Viewer open (target highlighted). Press Enter here to close it.", flush=True)
    while True:
        try:
            ready, _, _ = select.select([sys.stdin], [], [], VIEWER_REDRAW_SECONDS)
        except (OSError, ValueError):
            sys.stdin.readline()
            return
        if ready:
            sys.stdin.readline()
            return
        env.render()


def show_scene(spec: SceneSpec) -> None:
    env = _make_env(spec, viewer=True)
    try:
        env.reset(seed=spec.seed)
        env.render()
        _wait_while_rendering(env)
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seed", type=int,
        help="seed of the random sequence, to replay the same scenes",
    )
    parser.add_argument(
        "--output", type=Path,
        help=f"output folder (default: {OUTPUT_ROOT.relative_to(PROJECT_ROOT)}/oracle_<time>)",
    )
    args = parser.parse_args(argv)
    sequence_seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)
    rng = random.Random(sequence_seed)
    interactive = sys.stdin.isatty()
    run_directory = args.output or OUTPUT_ROOT / f"oracle_{datetime.now():%Y%m%d_%H%M%S}"

    print("\nSynthetic OBSERVE: oracle observation on random scenes")
    print(f"Profile {ORACLE_PROFILE.name}, {OBJECT_COUNT_RANGE[0]}-{OBJECT_COUNT_RANGE[1]} "
          f"objects per scene, sequence seed {sequence_seed} (--seed to replay)")
    print(f"Files: {run_directory}\n")
    index = 0
    while True:
        index += 1
        spec = random_scene(rng)
        output = observe_scene(spec)
        files = save_outputs(output, spec, run_directory / f"scene_{index:03d}")
        print(describe(output.observation, spec, output.scene_seed, index))
        print()
        print(describe_privileged(output.privileged, output.observation))
        print()
        for label, path in files.items():
            print(f"  {label:<12} {path}")
        if not interactive:
            return 0
        graph_image = files.get("graph_svg")
        if graph_image and _ask("\nOpen the support graph image?", default=True):
            webbrowser.open(graph_image.resolve().as_uri())
        if _ask("Open this scene in the MuJoCo viewer?", default=False):
            show_scene(spec)
        if not _ask("Next random scene?", default=True):
            return 0
        print()


def run_cli(argv: list[str] | None = None) -> int:
    """Command-line boundary: Ctrl+C or Ctrl+D end the session without a traceback."""
    try:
        return main(argv)
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
        return 130


if __name__ == "__main__":
    # Also runnable directly (e.g. the editor's Run button on this file).
    raise SystemExit(run_cli())
