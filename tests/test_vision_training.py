"""Modulo vision_training: ricette, randomizzazione, dataset, training e valutazione.

I test che non richiedono MuJoCo usano dati costruiti a mano; quelli che lo
richiedono generano scene minuscole. Il training vero (Ultralytics) non gira
qui: viene sostituito da un finto YOLO che produce un file di pesi.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from physical_ai_mujoco.sensors import SegmentationPrediction
from physical_ai_mujoco.simulation.visual_conditions import (
    BackgroundAppearance,
    CameraPerturbation,
    GroundAppearance,
    LightingConditions,
    VisualConditions,
)
from physical_ai_mujoco.vision_training import RecipeError, available_recipes, load_recipe
from physical_ai_mujoco.vision_training import evaluation as evaluate_module
from physical_ai_mujoco.vision_training import training as train_module
from physical_ai_mujoco.vision_training.dataset import (
    generate_dataset,
    split_by_scene,
    visibility_band,
)
from physical_ai_mujoco.vision_training.evaluation import (
    evaluate,
    minimum_recognizable_fraction,
)
from physical_ai_mujoco.vision_training.labels import (
    mask_to_yolo_segments,
    prepare_directory,
    write_data_yaml,
    write_sample,
)
from physical_ai_mujoco.vision_training.randomization import (
    STREAM_IMAGE,
    STREAM_VIEW,
    apply_image_effects,
    sample_visual_conditions,
    stream,
)
from physical_ai_mujoco.vision_training.recipe import (
    RECIPE_DIR,
    dataset_recipe_from_dict,
    training_recipe_from_dict,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE = RECIPE_DIR / "dataset_smoke.json"


def _smoke_data() -> dict:
    return json.loads(SMOKE.read_text())


# ------------------------------------------------------------------ ricette

def test_every_shipped_recipe_is_valid():
    datasets = available_recipes("dataset")
    trainings = available_recipes("training")
    assert {path.stem for path, _ in datasets} >= {"dataset_smoke", "dataset_sim_dr_v1", "dataset_sim_nodr_v1"}
    assert {path.stem for path, _ in trainings} >= {"training_smoke", "training_sim_dr_v1_m3", "training_real_finetune_v1"}
    for path, _ in datasets + trainings:
        load_recipe(path)


def test_recipe_rejects_typos_instead_of_using_defaults():
    data = _smoke_data()
    data["scenes"]["object_cont"] = [1, 3]
    with pytest.raises(RecipeError, match="object_cont"):
        dataset_recipe_from_dict(data)
    data = _smoke_data()
    data["randomization"]["camera"]["fovy_deg"] = [60.0, 40.0]
    with pytest.raises(RecipeError, match="min maggiore di max"):
        dataset_recipe_from_dict(data)
    data = _smoke_data()
    data["splits"] = {"validation": 0.5, "test": 0.5}
    with pytest.raises(RecipeError, match="training"):
        dataset_recipe_from_dict(data)
    training = json.loads((RECIPE_DIR / "training_smoke.json").read_text())
    training["ultralytics"]["epochs"] = 3
    with pytest.raises(RecipeError, match="epochs"):
        training_recipe_from_dict(training)


def test_sim_and_nodr_recipes_share_the_same_scenes():
    dr = load_recipe(RECIPE_DIR / "dataset_sim_dr_v1.json")
    nodr = load_recipe(RECIPE_DIR / "dataset_sim_nodr_v1.json")
    assert nodr.randomization is None and dr.randomization is not None
    assert replace(dr, name="x", randomization=None, source={}) == replace(nodr, name="x", source={})


# --------------------------------------------------------- randomizzazione

def test_randomization_is_reproducible_and_streams_are_independent():
    recipe = dataset_recipe_from_dict(_smoke_data()).randomization
    colors = {"object_000": (0.5, 0.5, 0.5), "object_001": (0.45, 0.48, 0.24)}
    first = sample_visual_conditions(recipe, stream(0, STREAM_VIEW, 3, 1), object_rgb=colors, target_id="object_001")
    second = sample_visual_conditions(recipe, stream(0, STREAM_VIEW, 3, 1), object_rgb=colors, target_id="object_001")
    other = sample_visual_conditions(recipe, stream(0, STREAM_VIEW, 3, 2), object_rgb=colors, target_id="object_001")
    assert first == second
    assert first != other
    assert isinstance(first, VisualConditions)
    palette = recipe.objects.target_palette
    jitter = recipe.objects.target_color_jitter
    assert any(
        np.all(np.abs(np.asarray(first.object_rgb["object_001"]) - np.asarray(color)) <= jitter + 1e-9)
        for color in palette
    )
    assert not np.array_equal(
        stream(0, STREAM_VIEW, 1).random(4), stream(0, STREAM_IMAGE, 1).random(4)
    )


def test_image_effects_keep_monochrome_shape_and_are_seeded():
    recipe = dataset_recipe_from_dict(_smoke_data()).randomization.image
    image = np.full((24, 32), 128, dtype=np.uint8)
    first = apply_image_effects(image, recipe, stream(1, STREAM_IMAGE, 0))
    second = apply_image_effects(image, recipe, stream(1, STREAM_IMAGE, 0))
    np.testing.assert_array_equal(first, second)
    assert first.shape == image.shape and first.dtype == np.uint8
    no_vignette = replace(recipe, vignette=(0.5, 0.5), noise_sigma=(0.0, 0.0), blur_probability=0.0,
                          contrast=(1.0, 1.0), brightness=(0.0, 0.0), gamma=(1.0, 1.0))
    dark_corners = apply_image_effects(image, no_vignette, stream(1, STREAM_IMAGE, 0))
    assert dark_corners[0, 0] < dark_corners[12, 16]


# ------------------------------------------------------------------ etichette

def test_labels_are_normalized_and_annotation_keeps_extra_fields(tmp_path):
    mask = np.zeros((10, 20), dtype=bool)
    mask[2:8, 4:16] = True
    segments = mask_to_yolo_segments(mask)
    assert len(segments) == 1 and all(0 <= value <= 1 for value in segments[0])

    destination = tmp_path / "dataset"
    prepare_directory(destination)
    target = np.zeros((12, 16), dtype=bool)
    target[2:8, 3:10] = True
    tiny = np.zeros_like(target)
    tiny[0, 0] = True
    annotation = write_sample(
        destination, "test", "sample", np.zeros(target.shape, dtype=np.uint8),
        {"target": target, "tiny": tiny}, {"target": "pfm_1_target", "tiny": "slab"},
        ("obstacle", "pfm_1_target"), 3, extra={"target_visible_fraction": 0.4},
    )
    assert [item["class_id"] for item in annotation["instances"]] == ["pfm_1_target"]
    stored = json.loads((destination / "annotations/test/sample.json").read_text())
    assert stored["target_visible_fraction"] == 0.4
    assert (destination / "labels/test/sample.txt").read_text().startswith("1 ")
    with pytest.raises(FileExistsError, match="non e' vuota"):
        prepare_directory(destination)
    yaml = write_data_yaml(tmp_path / "run/data.yaml", destination, ("obstacle", "pfm_1_target"))
    text = yaml.read_text()
    assert f"path: {destination.resolve()}" in text and "test: images/test" in text
    assert "train:" not in text  # niente split vuoti


def test_split_is_by_scene_reproducible_and_complete():
    recipe = dataset_recipe_from_dict(_smoke_data())
    recipe = replace(recipe, scene_count=20)
    splits = split_by_scene(recipe)
    assert splits == split_by_scene(recipe)
    assert {"train", "val", "test"} <= set(splits)
    assert splits.count("test") == 3 and splits.count("val") == 3


def test_visibility_bands_separate_hidden_targets():
    assert visibility_band(0.0) == "nascosto"
    assert visibility_band(None) == "fuori_inquadratura"
    assert visibility_band(0.05) == "0.00-0.10"
    assert visibility_band(1.0) == "0.90-1.00"


# ------------------------------------------------------------ valutazione

def test_minimum_fraction_stops_at_first_band_below_required_recall():
    def band(lower, recall, samples=10):
        return {"lower": lower, "recall": recall, "samples": samples}

    bands = [band(0.1, 0.2), band(0.2, 0.7), band(0.3, 0.9), band(0.4, 0.95), band(0.5, 0.3, samples=2)]
    assert minimum_recognizable_fraction(bands, 0.8, 5) == pytest.approx(0.3)
    assert minimum_recognizable_fraction([band(0.9, 0.5)], 0.8, 5) is None


def _tiny_eval_dataset(root: Path) -> Path:
    """Tre campioni: target ben visibile, target poco visibile, nessun target."""
    prepare_directory(root)
    (root / "summary.json").write_text(json.dumps({"class_names": ["obstacle", "pfm_1_target"]}))
    target = np.zeros((20, 30), dtype=bool)
    target[5:12, 3:10] = True
    for sample_id, fraction, present in (("big", 0.85, True), ("small", 0.15, True), ("none", None, False)):
        write_sample(
            root, "test", sample_id, np.zeros(target.shape, dtype=np.uint8),
            {"t": target} if present else {}, {"t": "pfm_1_target"},
            ("obstacle", "pfm_1_target"), 3,
            extra={"target_present": present, "target_visible_fraction": fraction, "object_count": 4},
        )
    return root


class _FakeBackend:
    """Riconosce il target solo nelle immagini con valore di grigio 0."""

    def __init__(self, target):
        self.target = target
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        if self.calls == 2:  # secondo campione in ordine alfabetico: "none"
            return (SegmentationPrediction(self.target, 1, 0.9),)  # falso positivo
        if self.calls == 3:  # "small": mancato
            return ()
        return (SegmentationPrediction(self.target, 1, 0.9),)


def test_evaluation_reports_recall_per_band_and_updates_card(tmp_path, monkeypatch):
    dataset = _tiny_eval_dataset(tmp_path / "ds")
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"x")
    weights.with_suffix(".json").write_text(json.dumps({"name": "model"}))
    target = np.zeros((20, 30), dtype=bool)
    target[5:12, 3:10] = True
    settings = replace(evaluate_module.EvaluationSettings(), min_samples_per_band=1)
    report = evaluate(weights, dataset, settings, backend=_FakeBackend(target), output=tmp_path / "r.json")
    bands = {item["band"]: item for item in report["bands"]}
    assert bands["0.80-0.90"]["recall"] == 1.0
    assert bands["0.10-0.20"]["recall"] == 0.0
    assert report["false_positive_rate"] == 1.0
    assert report["minimum_recognizable_visible_fraction"] == pytest.approx(0.8)
    card = json.loads(weights.with_suffix(".json").read_text())
    assert card["observability"]["minimum_recognizable_visible_fraction"] == pytest.approx(0.8)
    assert "ds:test" in card["evaluations"]


# ---------------------------------------------------------------- training

class _FakeYolo:
    created = []

    def __init__(self, base_model):
        self.base_model = base_model
        _FakeYolo.created.append(self)

    def train(self, **kwargs):
        self.kwargs = kwargs
        save_dir = Path(kwargs["project"]) / kwargs["name"]
        (save_dir / "weights").mkdir(parents=True, exist_ok=True)
        (save_dir / "weights/best.pt").write_bytes(b"pesi")
        return SimpleNamespace(save_dir=save_dir, results_dict={"metrics/mAP50(M)": 0.5})


def test_training_writes_card_records_parent_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "WEIGHTS_DIR", tmp_path / "weights")
    monkeypatch.setattr(train_module, "RUNS_DIR", tmp_path / "runs")
    dataset = _tiny_eval_dataset(tmp_path / "ds")
    for split in ("train", "val"):
        cv2.imwrite(str(dataset / "images" / split / "a.png"), np.zeros((4, 4), np.uint8))
    data = json.loads((RECIPE_DIR / "training_smoke.json").read_text())
    data.update(dataset=str(dataset), device="cpu")
    recipe = training_recipe_from_dict(data)

    card_path = train_module.train(recipe, yolo_factory=_FakeYolo, run_evaluation=False)
    card = json.loads(card_path.read_text())
    assert card_path.with_suffix(".pt").read_bytes() == b"pesi"
    assert card["resolved"]["device"] == "cpu"
    assert card["training_recipe"]["name"] == "smoke"
    assert card["parent"] is None
    assert _FakeYolo.created[-1].kwargs["hsv_s"] == 0.0
    assert "data.yaml" in _FakeYolo.created[-1].kwargs["data"]
    with pytest.raises(FileExistsError, match="non sovrascrive"):
        train_module.train(recipe, yolo_factory=_FakeYolo, run_evaluation=False)

    child = training_recipe_from_dict(dict(data, name="figlio", base_model=str(card_path.with_suffix(".pt"))))
    child_card = json.loads(train_module.train(child, yolo_factory=_FakeYolo, run_evaluation=False).read_text())
    assert child_card["parent"]["name"] == "smoke"


def test_training_requires_validation_images(tmp_path):
    dataset = tmp_path / "ds"
    prepare_directory(dataset)
    data = json.loads((RECIPE_DIR / "training_smoke.json").read_text())
    data.update(dataset=str(dataset))
    with pytest.raises(ValueError, match="validazione"):
        train_module.train(training_recipe_from_dict(data), yolo_factory=_FakeYolo)


# ------------------------------------------------------- MuJoCo (integrazione)

@pytest.fixture(scope="module")
def pile():
    import gymnasium as gym
    import physical_ai_mujoco.envs  # noqa: F401

    env = gym.make(
        "TargetExtraction-v0", object_count=4,
        scene_rules_path=str(ROOT / "configs/observe_tests/immersed_scene_rules.json"),
        obs_mode="state", disable_env_checker=True,
    )
    env.reset(seed=11)
    yield env.unwrapped
    env.close()


def test_nominal_calibration_is_unchanged_without_randomization(pile):
    simulator = pile.simulator
    intrinsics, _, _ = simulator.stereo_calibration()
    assert intrinsics[0, 0] == pytest.approx(simulator.scene.stereo_camera.focal_length_px(), rel=1e-12)


def test_visual_conditions_change_pixels_not_labels_and_restore_ground(pile):
    simulator = pile.simulator
    before, _ = simulator.render_stereo()
    masks_before = simulator.render_instance_masks("cam_left")
    simulator.apply_visual_conditions(VisualConditions(
        lighting=LightingConditions((0.3, 0.2, -1.0), 0.8, 0.2, 0.4, False),
        ground=GroundAppearance("noise", 0.6, 0.6, 8, 3, 0.0),
        background=BackgroundAppearance(0.8, 0.2, 1),
        object_rgb={pile.target_id: (0.7, 0.6, 0.4)},
    ))
    after, _ = simulator.render_stereo()
    masks_after = simulator.render_instance_masks("cam_left")
    assert np.mean(before != after) > 0.3
    assert masks_before.keys() == masks_after.keys()
    for key in masks_before:
        # Le maschere dipendono solo dalla geometria: al piu' qualche pixel di
        # bordo cambia per il ricalcolo delle pose (z-fighting fra facce a
        # contatto).
        assert np.count_nonzero(masks_before[key] != masks_after[key]) <= 3
    simulator.apply_visual_conditions(VisualConditions(ground=GroundAppearance("flat", 0.2)))
    simulator.apply_visual_conditions(VisualConditions(ground=GroundAppearance("checker")))


def test_camera_perturbation_keeps_calibration_consistent_with_masks(pile):
    simulator = pile.simulator
    simulator.apply_visual_conditions(VisualConditions(
        camera=CameraPerturbation((0.04, -0.03, 0.03), (0.01, 0.0, 0.0), 3.0, 58.0)
    ))
    intrinsics, world_from_left, baseline = simulator.stereo_calibration()
    assert intrinsics[1, 1] == pytest.approx(
        (simulator.scene.stereo_camera.height / 2) / np.tan(np.radians(58.0) / 2)
    )
    masks = simulator.render_instance_masks("cam_left")
    largest = max(masks, key=lambda key: masks[key].sum())
    point = np.append(simulator.get_object_state(largest).position, 1.0)
    camera = np.linalg.inv(world_from_left) @ point
    u = intrinsics[0, 0] * camera[0] / camera[2] + intrinsics[0, 2]
    v = intrinsics[1, 1] * camera[1] / camera[2] + intrinsics[1, 2]
    rows, columns = np.nonzero(masks[largest])
    assert columns.min() - 3 <= u <= columns.max() + 3
    assert rows.min() - 3 <= v <= rows.max() + 3
    assert baseline == pytest.approx(simulator.scene.stereo_camera.baseline)


def test_visibility_measure_restores_scene(pile):
    from physical_ai_mujoco.evaluation.target_exposure import measure_target_visibility

    simulator = pile.simulator
    before = {key: state.position for key, state in simulator.get_object_states().items()}
    visibility = measure_target_visibility(simulator, pile.target_id)
    after = {key: state.position for key, state in simulator.get_object_states().items()}
    assert before == after
    # Togliere gli ostacoli puo' solo scoprire pixel; qualche pixel di bordo
    # oscilla per il rasterizzatore, da qui la tolleranza del 3%.
    assert visibility.unoccluded_pixels >= 0.97 * visibility.visible_pixels
    if visibility.visible_fraction is not None:
        assert 0.0 <= visibility.visible_fraction <= 1.0


def test_generated_dataset_has_scene_level_splits_and_full_annotations(tmp_path):
    data = _smoke_data()
    data["scenes"].update(count=3, object_count=[2, 3], views_per_scene=1)
    recipe = dataset_recipe_from_dict(data)
    summary = generate_dataset(recipe, tmp_path / "ds", progress=None)
    root = tmp_path / "ds"
    assert set(summary["splits"]) == {"train", "val", "test"}
    annotations = [json.loads(path.read_text()) for path in root.glob("annotations/*/*.json")]
    assert len(annotations) == summary["image_count"] >= 6
    for item in annotations:
        assert {"target_visible_fraction", "object_count", "conditions", "target_present"} <= set(item)
        assert (root / item["image"]).is_file()
    scenes_by_split = {}
    for item in annotations:
        scenes_by_split.setdefault(item["scene_index"], set()).add(item["split"])
    assert all(len(splits) == 1 for splits in scenes_by_split.values())
    assert json.loads((root / "recipe.json").read_text())["name"] == "smoke"
    assert "path: " in (root / "data.yaml").read_text()


def test_randomization_does_not_change_which_views_lack_the_target(tmp_path):
    data = _smoke_data()
    data["scenes"].update(count=3, object_count=[2, 2], views_per_scene=2, target_absent_fraction=0.5)
    with_dr = dataset_recipe_from_dict(data)
    without_dr = replace(with_dr, name="smoke_nodr", randomization=None)
    generate_dataset(with_dr, tmp_path / "dr", progress=None)
    generate_dataset(without_dr, tmp_path / "nodr", progress=None)

    def presence(root):
        return {
            path.stem: json.loads(path.read_text())["target_present"]
            for path in root.glob("annotations/*/*.json")
        }

    assert presence(tmp_path / "dr") == presence(tmp_path / "nodr")
