from __future__ import annotations

import argparse
from pathlib import Path

from physical_ai_mujoco.scene.dataset_loader import (
    load_ground_dataset,
    load_object_dataset,
    load_scene_rules,
    load_simulation_config,
    validate_references,
)
from physical_ai_mujoco.scene.scene_generator import generate_scene
from physical_ai_mujoco.simulation.settling import (
    SettlingResult,
    run_until_settled,
)
from physical_ai_mujoco.simulation.simulator import Simulator


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    arguments = parse_arguments()
    scene = create_scene(arguments.seed)
    output_path = PROJECT_ROOT / f"outputs/scenes/{scene.scene_id}.json"
    scene.save(output_path)

    print(f"Scene: {scene.scene_id}")
    print(f"Description: {output_path}")

    simulator = Simulator(scene)
    try:
        result = run_until_settled(simulator)
        print_result(result)
        return 0 if result.settled else 1
    finally:
        simulator.close()


def create_scene(seed: int):
    object_dataset = load_object_dataset(
        PROJECT_ROOT / "datasets/object_dataset/geometric_objects.json"
    )
    ground_dataset = load_ground_dataset(
        PROJECT_ROOT / "datasets/ground_dataset/basic_grounds.json"
    )
    scene_rules = load_scene_rules(
        PROJECT_ROOT / "configs/phase_0a/scene_rules.json"
    )
    simulation_config = load_simulation_config(
        PROJECT_ROOT / "configs/phase_0a/simulation.json"
    )
    validate_references(object_dataset, ground_dataset, scene_rules)

    return generate_scene(
        object_dataset,
        ground_dataset,
        scene_rules,
        simulation_config,
        seed,
    )


def print_result(result: SettlingResult) -> None:
    print(f"Settled: {result.settled}")
    print(f"Simulation time: {result.simulation_time:.3f} s")

    for instance_id, state in result.object_states.items():
        position = ", ".join(f"{value:.4f}" for value in state.position)
        print(f"{instance_id} position: [{position}]")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
