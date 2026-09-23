"""Generazione del dataset sintetico del visore a partire da una ricetta.

MuJoCo fornisce scene fisicamente plausibili ed etichette perfette; la ricetta
decide quanta varieta' visiva ci mettiamo sopra. Ogni scena appartiene a un
solo split (train, val o test), cosi' nessuna vista di una scena di test e'
mai stata vista in addestramento.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from physical_ai_mujoco.evaluation.target_exposure import (
    measure_target_visibility,
    place_target_at_immersion,
)
from physical_ai_mujoco.sensors import SimulatedStereoCamera

from .labels import prepare_directory, write_data_yaml, write_sample
from .randomization import (
    STREAM_ABSENT,
    STREAM_IMAGE,
    STREAM_IMMERSION,
    STREAM_OBJECT_COUNT,
    STREAM_SPLIT,
    STREAM_VIEW,
    apply_image_effects,
    sample_visual_conditions,
    stream,
)
from .recipe import PROJECT_ROOT, DatasetRecipe, load_recipe


DATASETS_DIR = PROJECT_ROOT / "datasets/generated"
CAMERAS = (("left", "cam_left"), ("right", "cam_right"))
OBSTACLE_CLASS = "obstacle"


def split_by_scene(recipe: DatasetRecipe) -> list[str]:
    """Split di ogni scena, riproducibile e indipendente dal resto."""
    count = recipe.scene_count
    test = round(count * recipe.test_fraction)
    validation = round(count * recipe.validation_fraction)
    if count >= 3:
        test = max(test, 1) if recipe.test_fraction > 0 else 0
        validation = max(validation, 1) if recipe.validation_fraction > 0 else 0
    test = min(test, max(count - 1, 0))
    validation = min(validation, max(count - 1 - test, 0))
    order = stream(recipe.seed, STREAM_SPLIT).permutation(count)
    splits = ["train"] * count
    for position, scene_index in enumerate(order):
        if position < test:
            splits[scene_index] = "test"
        elif position < test + validation:
            splits[scene_index] = "val"
    return splits


def visibility_band(fraction: float | None, width: float = 0.1) -> str:
    """Etichetta della fascia di % visibile; "nascosto" se non si vede nulla."""
    if fraction is None:
        return "fuori_inquadratura"
    if fraction <= 0.0:
        return "nascosto"
    index = min(int(fraction / width), int(round(1 / width)) - 1)
    return f"{index * width:.2f}-{(index + 1) * width:.2f}"


def generate_dataset(
    recipe: DatasetRecipe,
    destination: str | Path | None = None,
    *,
    progress=print,
) -> dict:
    """Genera il dataset descritto dalla ricetta e ne restituisce il riassunto."""
    import gymnasium as gym
    import physical_ai_mujoco.envs  # noqa: F401  (registra l'environment)

    output = Path(destination) if destination is not None else DATASETS_DIR / recipe.name
    output = output.resolve()
    prepare_directory(output)
    splits = split_by_scene(recipe)
    camera = SimulatedStereoCamera(with_depth=False)
    started = time.monotonic()
    manifest: list[dict] = []
    class_names = None
    env = None
    active_object_count = None
    try:
        for scene_index in range(recipe.scene_count):
            low, high = recipe.object_count
            object_count = int(
                stream(recipe.seed, STREAM_OBJECT_COUNT, scene_index).integers(low, high + 1)
            )
            if env is None or object_count != active_object_count:
                if env is not None:
                    env.close()
                env = gym.make(
                    "TargetExtraction-v0",
                    object_count=object_count,
                    scene_rules_path=str(recipe.scene_rules),
                    obs_mode="state",
                    disable_env_checker=True,
                    highlight_target=False,
                )
                active_object_count = object_count
            _, info = env.reset(seed=recipe.seed + scene_index)
            simulator = env.unwrapped.simulator
            target_id = env.unwrapped.target_id
            target_type_id = env.unwrapped.session.scene_rules["object_selection"]["target_type_id"]
            if class_names is None:
                class_names = (OBSTACLE_CLASS, target_type_id)
            elif target_type_id not in class_names:
                raise ValueError("Le scene di un dataset devono avere lo stesso tipo di target")
            types = {item.instance_id: item.type_id for item in simulator.scene.objects}
            nominal_rgb = {
                item.instance_id: tuple(float(value) for value in item.rgba[:3])
                for item in simulator.scene.objects
            }
            immersion = None
            if recipe.target_immersion is not None:
                immersion = float(
                    stream(recipe.seed, STREAM_IMMERSION, scene_index).uniform(*recipe.target_immersion)
                )
                place_target_at_immersion(simulator, target_id, immersion, settle_others=True)

            split = splits[scene_index]
            view_count = recipe.views_per_scene
            view_index = 0
            while view_index < view_count:
                view_rng = stream(recipe.seed, STREAM_VIEW, scene_index, view_index)
                conditions = None
                if recipe.randomization is not None:
                    conditions = sample_visual_conditions(
                        recipe.randomization, view_rng,
                        object_rgb=nominal_rgb, target_id=target_id,
                    )
                    simulator.apply_visual_conditions(conditions)
                # Flusso proprio: la vista senza target deve essere la stessa
                # con o senza randomizzazione, altrimenti i due dataset non si
                # confrontano scena per scena.
                target_present = not bool(
                    stream(recipe.seed, STREAM_ABSENT, scene_index, view_index).random()
                    < recipe.target_absent_fraction
                )
                snapshot = None
                if not target_present:
                    snapshot = simulator.snapshot()
                    simulator.remove_object(target_id)
                visibility = {
                    view: (measure_target_visibility(simulator, target_id, name) if target_present else None)
                    for view, name in CAMERAS
                }
                frame = camera.capture(simulator)
                masks = {view: simulator.render_instance_masks(name) for view, name in CAMERAS}
                if snapshot is not None:
                    simulator.restore(snapshot)

                for camera_index, (view, _name) in enumerate(CAMERAS):
                    image = frame.gray_left if view == "left" else frame.gray_right
                    if recipe.randomization is not None and recipe.randomization.image is not None:
                        image = apply_image_effects(
                            image,
                            recipe.randomization.image,
                            stream(recipe.seed, STREAM_IMAGE, scene_index, view_index, camera_index),
                        )
                    sample_id = f"s{scene_index:06d}_v{view_index:02d}_{view}"
                    seen = visibility[view]
                    labelled = any(
                        type_id == target_type_id
                        and np.count_nonzero(masks[view].get(instance_id, False))
                        >= recipe.minimum_visible_pixels
                        for instance_id, type_id in types.items()
                    )
                    annotation = write_sample(
                        output, split, sample_id, image, masks[view], types,
                        class_names, recipe.minimum_visible_pixels,
                        extra={
                            "scene_index": scene_index,
                            "scene_seed": int(info["scene_seed"]),
                            "view_index": view_index,
                            "camera": view,
                            "object_count": object_count,
                            "target_type_id": target_type_id,
                            "target_present": target_present,
                            "target_labelled": labelled,
                            "target_visible_fraction": None if seen is None else seen.visible_fraction,
                            "target_visible_pixels": None if seen is None else seen.visible_pixels,
                            "target_unoccluded_pixels": None if seen is None else seen.unoccluded_pixels,
                            "target_immersion_fraction": immersion,
                            "hard_visibility_view": view_index >= recipe.views_per_scene,
                            "conditions": None if conditions is None else asdict(conditions),
                        },
                    )
                    manifest.append(annotation)

                # Viste extra solo per scene di training in cui il target e'
                # poco visibile: sono i casi su cui il detector sbaglia di piu'.
                left = visibility["left"]
                if (
                    view_index == 0
                    and split == "train"
                    and left is not None
                    and left.visible_fraction is not None
                    and recipe.hard_visibility_range[0]
                    <= left.visible_fraction
                    <= recipe.hard_visibility_range[1]
                ):
                    view_count += recipe.hard_visibility_extra_views
                view_index += 1
            if progress is not None and (scene_index + 1) % max(1, recipe.scene_count // 20) == 0:
                progress(f"  scene {scene_index + 1}/{recipe.scene_count}, immagini {len(manifest)}")
    finally:
        if env is not None:
            env.close()

    (output / "manifest.jsonl").write_text("".join(json.dumps(item) + "\n" for item in manifest), encoding="utf-8")
    write_data_yaml(output / "data.yaml", output, class_names)
    summary = _summary(recipe, manifest, class_names, time.monotonic() - started)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "recipe.json").write_text(json.dumps(recipe.source, indent=2) + "\n", encoding="utf-8")
    return summary


def _summary(recipe: DatasetRecipe, manifest: list[dict], class_names, seconds: float) -> dict:
    by_split: dict[str, dict] = {}
    for item in manifest:
        stats = by_split.setdefault(
            item["split"],
            {"images": 0, "target_labelled": 0, "target_absent": 0, "hard_visibility_views": 0,
             "visibility_bands": {}, "object_counts": {}},
        )
        stats["images"] += 1
        stats["target_labelled"] += int(item["target_labelled"])
        stats["target_absent"] += int(not item["target_present"])
        stats["hard_visibility_views"] += int(item["hard_visibility_view"])
        if item["target_present"]:
            band = visibility_band(item["target_visible_fraction"])
            stats["visibility_bands"][band] = stats["visibility_bands"].get(band, 0) + 1
        count = str(item["object_count"])
        stats["object_counts"][count] = stats["object_counts"].get(count, 0) + 1
    fractions = [
        item["target_visible_fraction"] for item in manifest
        if item["target_visible_fraction"] is not None
    ]
    return {
        "recipe_name": recipe.name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generation_seconds": round(seconds, 1),
        "class_names": list(class_names),
        "scene_count": recipe.scene_count,
        "image_count": len(manifest),
        "randomization": recipe.randomization is not None,
        "mean_target_visible_fraction": float(np.mean(fractions)) if fractions else None,
        "splits": by_split,
        "mujoco_version": _package_version("mujoco"),
        "format": "PNG in grigio + maschere lossless + JSON + poligoni YOLO-seg",
    }


def _package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Genera un dataset del visore da una ricetta")
    parser.add_argument("recipe", type=Path, help="Ricetta JSON con kind='dataset'")
    parser.add_argument("--output", type=Path, help="Cartella di destinazione (default: datasets/generated/<nome>)")
    args = parser.parse_args(argv)
    recipe = load_recipe(args.recipe)
    if not isinstance(recipe, DatasetRecipe):
        parser.error("La ricetta indicata non e' di tipo 'dataset'")
    summary = generate_dataset(recipe, args.output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
