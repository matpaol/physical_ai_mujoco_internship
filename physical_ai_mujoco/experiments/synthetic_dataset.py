"""Generazione riproducibile di immagini B/N e maschere dalla simulazione."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from physical_ai_mujoco.evaluation.target_exposure import (
    measure_target_exposure,
    place_target_at_immersion,
)
from physical_ai_mujoco.sensors import (
    ImageDisturbance,
    SimulatedStereoCamera,
    apply_image_disturbance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = ("obstacle", "pfm_1_target")


@dataclass(frozen=True)
class DatasetGenerationConfig:
    scene_count: int = 100
    object_count: int | tuple[int, int] = 6
    seed: int = 0
    validation_fraction: float = 0.2
    minimum_visible_pixels: int = 12
    augmented_copies_per_train_view: int = 1
    target_immersion_range: tuple[float, float] | None = None
    hard_visibility_copies_per_train_view: int = 0
    hard_visibility_range: tuple[float, float] = (0.05, 0.35)

    def __post_init__(self):
        valid_count = (
            isinstance(self.object_count, int)
            and not isinstance(self.object_count, bool)
            and self.object_count >= 1
        ) or (
            isinstance(self.object_count, tuple)
            and len(self.object_count) == 2
            and all(isinstance(value, int) and value >= 1 for value in self.object_count)
            and self.object_count[0] <= self.object_count[1]
        )
        if self.scene_count < 1 or not valid_count or self.seed < 0:
            raise ValueError("Scene, oggetti e seed non validi")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction deve essere in [0, 1)")
        if self.minimum_visible_pixels < 1:
            raise ValueError("minimum_visible_pixels deve essere positivo")
        if self.augmented_copies_per_train_view < 0:
            raise ValueError("augmented_copies_per_train_view non puo' essere negativo")
        if self.hard_visibility_copies_per_train_view < 0:
            raise ValueError("hard_visibility_copies_per_train_view non puo' essere negativo")
        if self.target_immersion_range is not None and (
            not isinstance(self.target_immersion_range, tuple)
            or len(self.target_immersion_range) != 2
            or not 0 <= self.target_immersion_range[0] <= self.target_immersion_range[1] <= 1
        ):
            raise ValueError("target_immersion_range deve essere (min, max) in [0, 1]")
        if (
            not isinstance(self.hard_visibility_range, tuple)
            or len(self.hard_visibility_range) != 2
            or not 0 <= self.hard_visibility_range[0]
            <= self.hard_visibility_range[1] <= 1
        ):
            raise ValueError("hard_visibility_range deve essere (min, max) in [0, 1]")


def _validation_indices(config: DatasetGenerationConfig) -> set[int]:
    if config.scene_count < 2 or config.validation_fraction == 0:
        return set()
    count = max(1, round(config.scene_count * config.validation_fraction))
    count = min(count, config.scene_count - 1)
    rng = np.random.default_rng(config.seed + 91_000_021)
    return set(int(value) for value in rng.choice(config.scene_count, count, replace=False))


def _object_count_for_scene(config: DatasetGenerationConfig, scene_index: int) -> int:
    if isinstance(config.object_count, int):
        return config.object_count
    low, high = config.object_count
    return int(
        np.random.default_rng(config.seed + scene_index + 92_000_033).integers(
            low, high + 1
        )
    )


def mask_to_yolo_segments(mask: np.ndarray) -> tuple[tuple[float, ...], ...]:
    """Converte le componenti visibili in poligoni YOLO-seg normalizzati."""
    region = np.asarray(mask, dtype=np.uint8)
    if region.ndim != 2:
        raise ValueError("La maschera deve essere bidimensionale")
    height, width = region.shape
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if len(contour) < 3 or cv2.contourArea(contour) <= 0:
            continue
        points = contour[:, 0, :].astype(float)
        normalized = np.column_stack((points[:, 0] / width, points[:, 1] / height))
        segments.append(tuple(float(value) for value in normalized.ravel()))
    return tuple(segments)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Impossibile salvare {path}")


def _prepare_directory(destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"La destinazione non e' vuota: {destination}. Scegli una nuova cartella."
        )
    destination.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        for folder in ("images", "labels", "masks", "annotations"):
            (destination / folder / split).mkdir(parents=True, exist_ok=True)


def _write_sample(
    destination: Path,
    split: str,
    sample_id: str,
    image: np.ndarray,
    masks: dict[str, np.ndarray],
    type_ids: dict[str, str],
    target_type_id: str,
    minimum_visible_pixels: int,
) -> dict:
    height, width = image.shape
    image_path = destination / "images" / split / f"{sample_id}.png"
    label_path = destination / "labels" / split / f"{sample_id}.txt"
    annotation_path = destination / "annotations" / split / f"{sample_id}.json"
    _write_image(image_path, image)

    instances = []
    labels = []
    for instance_id, raw_mask in sorted(masks.items()):
        mask = np.asarray(raw_mask, dtype=bool)
        visible_pixels = int(mask.sum())
        if visible_pixels < minimum_visible_pixels:
            continue
        class_name = target_type_id if type_ids.get(instance_id) == target_type_id else "obstacle"
        class_index = CLASS_NAMES.index(class_name)
        rows, columns = np.nonzero(mask)
        bbox = [int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())]
        mask_path = destination / "masks" / split / f"{sample_id}_{instance_id}.png"
        _write_image(mask_path, mask.astype(np.uint8) * 255)
        segments = mask_to_yolo_segments(mask)
        # YOLO-seg rappresenta una istanza con un solo poligono. Conserviamo
        # comunque la maschera lossless; per il file YOLO usiamo la componente
        # visibile maggiore, evitando di trasformare frammenti occlusi in
        # oggetti distinti.
        for segment in segments[:1]:
            labels.append(" ".join((str(class_index), *(f"{value:.8f}" for value in segment))))
        instances.append(
            {
                "instance_id": instance_id,
                "class_id": class_name,
                "class_index": class_index,
                "visible_pixels": visible_pixels,
                "bbox_xyxy": bbox,
                "mask": str(mask_path.relative_to(destination)),
                "visible_components": len(segments),
                "yolo_polygon_exported": bool(segments),
            }
        )

    label_path.write_text("\n".join(labels) + ("\n" if labels else ""))
    annotation = {
        "sample_id": sample_id,
        "split": split,
        "image": str(image_path.relative_to(destination)),
        "width": width,
        "height": height,
        "instances": instances,
    }
    annotation_path.write_text(json.dumps(annotation, indent=2) + "\n")
    return annotation


def generate_synthetic_dataset(
    config: DatasetGenerationConfig,
    destination: str | Path,
) -> dict:
    """Genera scene separate per split; MuJoCo e' usato solo per le etichette."""
    import gymnasium as gym
    import physical_ai_mujoco.envs  # noqa: F401

    output = Path(destination).resolve()
    _prepare_directory(output)
    validation = _validation_indices(config)
    camera = SimulatedStereoCamera(with_depth=False)
    env = None
    active_object_count = None
    manifest = []
    target_visible_samples = 0
    disturbance = ImageDisturbance()
    try:
        for scene_index in range(config.scene_count):
            object_count = _object_count_for_scene(config, scene_index)
            if env is None or object_count != active_object_count:
                if env is not None:
                    env.close()
                env = gym.make(
                    "TargetExtraction-v0",
                    object_count=object_count,
                    scene_rules_path=(
                        str(PROJECT_ROOT / "configs/observe_tests/immersed_scene_rules.json")
                        if config.target_immersion_range is not None
                        else None
                    ),
                    obs_mode="state",
                    disable_env_checker=True,
                    highlight_target=False,
                )
                active_object_count = object_count
            _, info = env.reset(seed=config.seed + scene_index)
            simulator = env.unwrapped.simulator
            target_id = env.unwrapped.target_id
            target_type_id = env.unwrapped.session.scene_rules[
                "object_selection"
            ]["target_type_id"]
            exposure_by_view = None
            if config.target_immersion_range is not None:
                immersion_rng = np.random.default_rng(
                    config.seed + scene_index + 93_000_041
                )
                immersion_fraction = float(
                    immersion_rng.uniform(*config.target_immersion_range)
                )
                placement = place_target_at_immersion(
                    simulator, target_id, immersion_fraction
                )
                exposure_by_view = {
                    view: measure_target_exposure(
                        simulator,
                        target_id,
                        immersion_fraction,
                        placement,
                        camera_name=f"cam_{view}",
                    )
                    for view in ("left", "right")
                }
            frame = camera.capture(simulator)
            masks_by_view = {
                "left": simulator.render_instance_masks("cam_left"),
                "right": simulator.render_instance_masks("cam_right"),
            }
            types = {
                item.instance_id: item.type_id for item in simulator.scene.objects
            }
            split = "val" if scene_index in validation else "train"
            for view, image in (
                ("left", frame.gray_left),
                ("right", frame.gray_right),
            ):
                exposure = None if exposure_by_view is None else exposure_by_view[view]
                sample_id = f"scene_{scene_index:06d}_seed_{int(info['scene_seed'])}_{view}"
                annotation = _write_sample(
                    output,
                    split,
                    sample_id,
                    image,
                    masks_by_view[view],
                    types,
                    target_type_id,
                    config.minimum_visible_pixels,
                )
                annotation.update(
                    scene_index=scene_index,
                    input_seed=config.seed + scene_index,
                    scene_seed=int(info["scene_seed"]),
                    view=view,
                    target_type_id=target_type_id,
                    object_count=object_count,
                )
                if exposure is not None:
                    annotation.update(
                        target_immersion_fraction=exposure.immersion_fraction,
                        target_geometric_exposure=exposure.geometric_exposure_fraction,
                        target_ground_visible_fraction=exposure.ground_visible_fraction,
                        target_camera_visible_fraction=exposure.camera_visible_fraction,
                        target_obstacle_visibility_fraction=exposure.obstacle_visibility_fraction,
                        target_visible_pixels=exposure.visible_pixels,
                        target_fully_exposed_pixels=exposure.fully_exposed_pixels,
                    )
                target_visible_samples += int(
                    any(item["class_id"] == target_type_id for item in annotation["instances"])
                )
                manifest.append(annotation)
                if split == "train":
                    hard_visibility = bool(
                        exposure is not None
                        and exposure.visible_pixels >= config.minimum_visible_pixels
                        and config.hard_visibility_range[0]
                        <= exposure.camera_visible_fraction
                        <= config.hard_visibility_range[1]
                    )
                    copy_count = config.augmented_copies_per_train_view + (
                        config.hard_visibility_copies_per_train_view
                        if hard_visibility else 0
                    )
                    for copy_index in range(copy_count):
                        augmented_id = f"{sample_id}_aug_{copy_index + 1:02d}"
                        augmentation_rng = np.random.default_rng(
                            int(info["scene_seed"])
                            + (1 if view == "left" else 2) * 1_000_003
                            + copy_index * 10_000_019
                        )
                        augmented = apply_image_disturbance(
                            image, disturbance, augmentation_rng
                        )
                        augmented_annotation = _write_sample(
                            output,
                            split,
                            augmented_id,
                            augmented,
                            masks_by_view[view],
                            types,
                            target_type_id,
                            config.minimum_visible_pixels,
                        )
                        augmented_annotation.update(
                            scene_index=scene_index,
                            input_seed=config.seed + scene_index,
                            scene_seed=int(info["scene_seed"]),
                            view=view,
                            target_type_id=target_type_id,
                            object_count=object_count,
                            augmented=True,
                            hard_visibility_augmentation=(
                                hard_visibility
                                and copy_index >= config.augmented_copies_per_train_view
                            ),
                            source_sample_id=sample_id,
                        )
                        if exposure is not None:
                            augmented_annotation.update(
                                target_immersion_fraction=exposure.immersion_fraction,
                                target_geometric_exposure=exposure.geometric_exposure_fraction,
                                target_ground_visible_fraction=exposure.ground_visible_fraction,
                                target_camera_visible_fraction=exposure.camera_visible_fraction,
                                target_obstacle_visibility_fraction=exposure.obstacle_visibility_fraction,
                                target_visible_pixels=exposure.visible_pixels,
                                target_fully_exposed_pixels=exposure.fully_exposed_pixels,
                            )
                        target_visible_samples += int(
                            any(
                                item["class_id"] == target_type_id
                                for item in augmented_annotation["instances"]
                            )
                        )
                        manifest.append(augmented_annotation)
    finally:
        if env is not None:
            env.close()

    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in manifest)
    )
    (output / "data.yaml").write_text(
        f"path: {output}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: obstacle\n"
        "  1: pfm_1_target\n"
    )
    summary = {
        "config": asdict(config),
        "class_names": list(CLASS_NAMES),
        "scene_count": config.scene_count,
        "image_count": len(manifest),
        "train_images": sum(item["split"] == "train" for item in manifest),
        "validation_images": sum(item["split"] == "val" for item in manifest),
        "target_visible_images": target_visible_samples,
        "target_hidden_images": len(manifest) - target_visible_samples,
        "hard_visibility_augmented_images": sum(
            bool(item.get("hard_visibility_augmentation")) for item in manifest
        ),
        "format": "lossless masks + JSON + YOLO segmentation polygons",
    }
    exposure_values = [
        item["target_camera_visible_fraction"]
        for item in manifest
        if "target_camera_visible_fraction" in item
    ]
    if exposure_values:
        summary["mean_target_camera_visible_fraction"] = float(
            np.mean(exposure_values)
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Genera il dataset sintetico PFM-1")
    parser.add_argument("--scenes", type=int, default=100)
    parser.add_argument("--objects", type=int, default=6)
    parser.add_argument("--min-objects", type=int)
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--minimum-visible-pixels", type=int, default=12)
    parser.add_argument("--augmented-copies", type=int, default=1)
    parser.add_argument("--hard-visibility-copies", type=int, default=0)
    parser.add_argument("--hard-visibility-min", type=float, default=0.05)
    parser.add_argument("--hard-visibility-max", type=float, default=0.35)
    parser.add_argument("--min-target-immersion", type=float)
    parser.add_argument("--max-target-immersion", type=float)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "datasets/generated/pfm_1",
    )
    args = parser.parse_args(argv)
    if (args.min_objects is None) != (args.max_objects is None):
        parser.error("--min-objects e --max-objects vanno specificati insieme")
    if (args.min_target_immersion is None) != (args.max_target_immersion is None):
        parser.error(
            "--min-target-immersion e --max-target-immersion vanno specificati insieme"
        )
    object_count = (
        (args.min_objects, args.max_objects)
        if args.min_objects is not None
        else args.objects
    )
    summary = generate_synthetic_dataset(
        DatasetGenerationConfig(
            scene_count=args.scenes,
            object_count=object_count,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            minimum_visible_pixels=args.minimum_visible_pixels,
            augmented_copies_per_train_view=args.augmented_copies,
            hard_visibility_copies_per_train_view=args.hard_visibility_copies,
            hard_visibility_range=(
                args.hard_visibility_min,
                args.hard_visibility_max,
            ),
            target_immersion_range=(
                None
                if args.min_target_immersion is None
                else (args.min_target_immersion, args.max_target_immersion)
            ),
        ),
        args.output,
    )
    print(json.dumps(summary, indent=2))
    print(f"Dataset salvato in: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
