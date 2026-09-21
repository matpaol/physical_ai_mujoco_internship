"""Ricette JSON del visore: l'unica fonte di verita' di dataset e training.

Una ricetta descrive per intero un esperimento: rigenerare un dataset o
riaddestrare un modello significa rilanciare la stessa ricetta, non ricordarsi
quali domande del menu erano state risposte. Le chiavi sconosciute sono un
errore: un refuso non deve diventare in silenzio un valore di default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

from physical_ai_mujoco.simulation.visual_conditions import GROUND_KINDS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECIPE_DIR = PROJECT_ROOT / "configs/vision_training"


class RecipeError(ValueError):
    """La ricetta non e' valida; il messaggio indica la chiave."""


# ---------------------------------------------------------------- lettura

def _section(data, name: str, allowed: set[str], *, required: set[str] = frozenset()) -> dict:
    if not isinstance(data, dict):
        raise RecipeError(f"{name}: atteso un oggetto JSON")
    unknown = set(data) - allowed
    if unknown:
        raise RecipeError(f"{name}: chiavi sconosciute {sorted(unknown)}")
    missing = set(required) - set(data)
    if missing:
        raise RecipeError(f"{name}: chiavi mancanti {sorted(missing)}")
    return data


def _number(value, name: str, low: float = -math.inf, high: float = math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecipeError(f"{name}: atteso un numero")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise RecipeError(f"{name}: {result} fuori da [{low}, {high}]")
    return result


def _integer(value, name: str, low: int = 0, high: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecipeError(f"{name}: atteso un intero")
    if value < low or (high is not None and value > high):
        raise RecipeError(f"{name}: {value} fuori da [{low}, {high}]")
    return value


def _interval(value, name: str, low: float = -math.inf, high: float = math.inf) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise RecipeError(f"{name}: atteso [min, max]")
    first = _number(value[0], name, low, high)
    second = _number(value[1], name, low, high)
    if first > second:
        raise RecipeError(f"{name}: min maggiore di max")
    return first, second


def _int_interval(value, name: str, low: int = 0) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise RecipeError(f"{name}: atteso [min, max]")
    first = _integer(value[0], name, low)
    second = _integer(value[1], name, low)
    if first > second:
        raise RecipeError(f"{name}: min maggiore di max")
    return first, second


def _rgb(value, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise RecipeError(f"{name}: atteso [r, g, b]")
    return tuple(_number(item, name, 0.0, 1.0) for item in value)


# ------------------------------------------------------- randomizzazione

@dataclass(frozen=True)
class CameraRandomization:
    position_jitter_m: float
    target_jitter_m: float
    roll_deg: float
    fovy_deg: tuple[float, float] | None


@dataclass(frozen=True)
class LightingRandomization:
    azimuth_deg: tuple[float, float]
    elevation_deg: tuple[float, float]
    diffuse: tuple[float, float]
    ambient: tuple[float, float]
    headlight: tuple[float, float]
    shadow_probability: float


@dataclass(frozen=True)
class GroundRandomization:
    kinds: dict[str, float]
    gray: tuple[float, float]
    contrast: tuple[float, float]
    grain_px: tuple[int, int]
    reflectance: tuple[float, float] | None


@dataclass(frozen=True)
class BackgroundRandomization:
    gray: tuple[float, float]
    contrast: tuple[float, float]


@dataclass(frozen=True)
class ObjectRandomization:
    obstacle_color_jitter: float
    target_palette: tuple[tuple[float, float, float], ...]
    target_color_jitter: float


@dataclass(frozen=True)
class ImageRandomization:
    contrast: tuple[float, float]
    brightness: tuple[float, float]
    gamma: tuple[float, float]
    noise_sigma: tuple[float, float]
    blur_probability: float
    blur_kernel: int
    vignette: tuple[float, float]


@dataclass(frozen=True)
class RandomizationRecipe:
    """Ogni sezione assente lascia quell'aspetto al valore nominale."""

    camera: CameraRandomization | None = None
    lighting: LightingRandomization | None = None
    ground: GroundRandomization | None = None
    background: BackgroundRandomization | None = None
    objects: ObjectRandomization | None = None
    image: ImageRandomization | None = None


def _randomization(data) -> RandomizationRecipe | None:
    if data is None:
        return None
    _section(data, "randomization", {"camera", "lighting", "ground", "background", "objects", "image"})
    result = {}
    if "camera" in data:
        item = _section(data["camera"], "randomization.camera",
                        {"position_jitter_m", "target_jitter_m", "roll_deg", "fovy_deg"},
                        required={"position_jitter_m", "target_jitter_m", "roll_deg"})
        result["camera"] = CameraRandomization(
            _number(item["position_jitter_m"], "camera.position_jitter_m", 0.0, 0.5),
            _number(item["target_jitter_m"], "camera.target_jitter_m", 0.0, 0.5),
            _number(item["roll_deg"], "camera.roll_deg", 0.0, 45.0),
            None if item.get("fovy_deg") is None
            else _interval(item["fovy_deg"], "camera.fovy_deg", 5.0, 120.0),
        )
    if "lighting" in data:
        keys = {"azimuth_deg", "elevation_deg", "diffuse", "ambient", "headlight", "shadow_probability"}
        item = _section(data["lighting"], "randomization.lighting", keys, required=keys)
        result["lighting"] = LightingRandomization(
            _interval(item["azimuth_deg"], "lighting.azimuth_deg", -360.0, 720.0),
            _interval(item["elevation_deg"], "lighting.elevation_deg", 1.0, 90.0),
            _interval(item["diffuse"], "lighting.diffuse", 0.0, 2.0),
            _interval(item["ambient"], "lighting.ambient", 0.0, 2.0),
            _interval(item["headlight"], "lighting.headlight", 0.0, 2.0),
            _number(item["shadow_probability"], "lighting.shadow_probability", 0.0, 1.0),
        )
    if "ground" in data:
        item = _section(data["ground"], "randomization.ground",
                        {"kinds", "gray", "contrast", "grain_px", "reflectance"},
                        required={"kinds", "gray", "contrast", "grain_px"})
        kinds = _section(item["kinds"], "ground.kinds", set(GROUND_KINDS))
        weights = {kind: _number(value, f"ground.kinds.{kind}", 0.0) for kind, value in kinds.items()}
        if not weights or sum(weights.values()) <= 0:
            raise RecipeError("ground.kinds: serve almeno un peso positivo")
        result["ground"] = GroundRandomization(
            weights,
            _interval(item["gray"], "ground.gray", 0.0, 1.0),
            _interval(item["contrast"], "ground.contrast", 0.0, 1.0),
            _int_interval(item["grain_px"], "ground.grain_px", 1),
            None if item.get("reflectance") is None
            else _interval(item["reflectance"], "ground.reflectance", 0.0, 1.0),
        )
    if "background" in data:
        item = _section(data["background"], "randomization.background", {"gray", "contrast"},
                        required={"gray", "contrast"})
        result["background"] = BackgroundRandomization(
            _interval(item["gray"], "background.gray", 0.0, 1.0),
            _interval(item["contrast"], "background.contrast", 0.0, 1.0),
        )
    if "objects" in data:
        keys = {"obstacle_color_jitter", "target_palette", "target_color_jitter"}
        item = _section(data["objects"], "randomization.objects", keys, required=keys)
        palette = item["target_palette"]
        if not isinstance(palette, list):
            raise RecipeError("objects.target_palette: attesa una lista di [r, g, b]")
        result["objects"] = ObjectRandomization(
            _number(item["obstacle_color_jitter"], "objects.obstacle_color_jitter", 0.0, 1.0),
            tuple(_rgb(color, "objects.target_palette") for color in palette),
            _number(item["target_color_jitter"], "objects.target_color_jitter", 0.0, 1.0),
        )
    if "image" in data:
        keys = {"contrast", "brightness", "gamma", "noise_sigma", "blur_probability",
                "blur_kernel", "vignette"}
        item = _section(data["image"], "randomization.image", keys, required=keys)
        kernel = _integer(item["blur_kernel"], "image.blur_kernel", 1, 31)
        if kernel % 2 == 0:
            raise RecipeError("image.blur_kernel deve essere dispari")
        contrast = _interval(item["contrast"], "image.contrast", 0.01, 5.0)
        gamma = _interval(item["gamma"], "image.gamma", 0.01, 5.0)
        result["image"] = ImageRandomization(
            contrast,
            _interval(item["brightness"], "image.brightness", -255.0, 255.0),
            gamma,
            _interval(item["noise_sigma"], "image.noise_sigma", 0.0, 100.0),
            _number(item["blur_probability"], "image.blur_probability", 0.0, 1.0),
            kernel,
            _interval(item["vignette"], "image.vignette", 0.0, 1.0),
        )
    return RandomizationRecipe(**result)


# ------------------------------------------------------------- dataset

@dataclass(frozen=True)
class DatasetRecipe:
    name: str
    seed: int
    scene_rules: Path
    scene_count: int
    object_count: tuple[int, int]
    target_immersion: tuple[float, float] | None
    views_per_scene: int
    target_absent_fraction: float
    hard_visibility_range: tuple[float, float]
    hard_visibility_extra_views: int
    validation_fraction: float
    test_fraction: float
    minimum_visible_pixels: int
    randomization: RandomizationRecipe | None
    source: dict = field(default_factory=dict, compare=False, repr=False)


def dataset_recipe_from_dict(data: dict) -> DatasetRecipe:
    _section(data, "ricetta dataset",
             {"kind", "name", "description", "seed", "scene_rules", "scenes", "splits", "labels", "randomization"},
             required={"kind", "name", "seed", "scene_rules", "scenes", "splits", "labels"})
    if data["kind"] != "dataset":
        raise RecipeError("kind: attesa 'dataset'")
    name = _name(data["name"])
    scenes = _section(data["scenes"], "scenes",
                      {"count", "object_count", "target_immersion", "views_per_scene",
                       "target_absent_fraction", "hard_visibility"},
                      required={"count", "object_count", "views_per_scene"})
    hard = _section(scenes.get("hard_visibility", {"range": [0.0, 0.0], "extra_views": 0}),
                    "scenes.hard_visibility", {"range", "extra_views"}, required={"range", "extra_views"})
    splits = _section(data["splits"], "splits", {"validation", "test"}, required={"validation", "test"})
    validation = _number(splits["validation"], "splits.validation", 0.0, 0.9)
    test = _number(splits["test"], "splits.test", 0.0, 0.9)
    if validation + test >= 1.0:
        raise RecipeError("splits: validation + test deve lasciare scene per il training")
    labels = _section(data["labels"], "labels", {"minimum_visible_pixels"}, required={"minimum_visible_pixels"})
    rules = _project_path(data["scene_rules"], "scene_rules")
    if not rules.is_file():
        raise RecipeError(f"scene_rules: file non trovato {rules}")
    return DatasetRecipe(
        name=name,
        seed=_integer(data["seed"], "seed"),
        scene_rules=rules,
        scene_count=_integer(scenes["count"], "scenes.count", 1),
        object_count=_int_interval(scenes["object_count"], "scenes.object_count", 1),
        target_immersion=(
            None if scenes.get("target_immersion") is None
            else _interval(scenes["target_immersion"], "scenes.target_immersion", 0.0, 1.0)
        ),
        views_per_scene=_integer(scenes["views_per_scene"], "scenes.views_per_scene", 1, 50),
        target_absent_fraction=_number(
            scenes.get("target_absent_fraction", 0.0), "scenes.target_absent_fraction", 0.0, 1.0
        ),
        hard_visibility_range=_interval(hard["range"], "hard_visibility.range", 0.0, 1.0),
        hard_visibility_extra_views=_integer(hard["extra_views"], "hard_visibility.extra_views", 0, 50),
        validation_fraction=validation,
        test_fraction=test,
        minimum_visible_pixels=_integer(labels["minimum_visible_pixels"], "labels.minimum_visible_pixels", 1),
        randomization=_randomization(data.get("randomization")),
        source=data,
    )


# ------------------------------------------------------------ training

@dataclass(frozen=True)
class EvaluationSettings:
    split: str = "test"
    iou_threshold: float = 0.5
    confidence: float = 0.25
    band_width: float = 0.1
    required_recall: float = 0.8
    min_samples_per_band: int = 10


def evaluation_settings_from_dict(data: dict | None) -> EvaluationSettings:
    if data is None:
        return EvaluationSettings()
    item = _section(data, "evaluation", {"split", "iou_threshold", "confidence", "band_width",
                                         "required_recall", "min_samples_per_band"})
    defaults = EvaluationSettings()
    split = item.get("split", defaults.split)
    if split not in ("train", "val", "test"):
        raise RecipeError("evaluation.split: atteso train, val o test")
    band = _number(item.get("band_width", defaults.band_width), "evaluation.band_width", 0.01, 1.0)
    return EvaluationSettings(
        split,
        _number(item.get("iou_threshold", defaults.iou_threshold), "evaluation.iou_threshold", 0.01, 1.0),
        _number(item.get("confidence", defaults.confidence), "evaluation.confidence", 0.0, 1.0),
        band,
        _number(item.get("required_recall", defaults.required_recall), "evaluation.required_recall", 0.0, 1.0),
        _integer(item.get("min_samples_per_band", defaults.min_samples_per_band),
                 "evaluation.min_samples_per_band", 1),
    )


@dataclass(frozen=True)
class TrainingRecipe:
    name: str
    dataset: Path
    base_model: str
    epochs: int
    image_size: int
    batch: int
    patience: int
    seed: int
    device: str
    ultralytics: dict
    evaluation: EvaluationSettings
    source: dict = field(default_factory=dict, compare=False, repr=False)


def training_recipe_from_dict(data: dict) -> TrainingRecipe:
    _section(data, "ricetta training",
             {"kind", "name", "description", "dataset", "base_model", "epochs", "image_size",
              "batch", "patience", "seed", "device", "ultralytics", "evaluation"},
             required={"kind", "name", "dataset", "base_model", "epochs", "image_size", "batch", "seed"})
    if data["kind"] != "training":
        raise RecipeError("kind: attesa 'training'")
    image_size = _integer(data["image_size"], "image_size", 32, 4096)
    if image_size % 32:
        raise RecipeError("image_size deve essere multiplo di 32")
    device = data.get("device", "auto")
    if not isinstance(device, str) or not device:
        raise RecipeError("device: atteso 'auto', 'cpu', 'mps' o un indice CUDA")
    extra = data.get("ultralytics", {})
    if not isinstance(extra, dict):
        raise RecipeError("ultralytics: atteso un oggetto JSON")
    reserved = {"data", "epochs", "imgsz", "batch", "seed", "device", "project", "name", "exist_ok", "patience"}
    clash = reserved & set(extra)
    if clash:
        raise RecipeError(f"ultralytics: {sorted(clash)} vanno scritti fuori da questa sezione")
    base_model = data["base_model"]
    if not isinstance(base_model, str) or not base_model:
        raise RecipeError("base_model: atteso un nome di modello o un percorso .pt")
    return TrainingRecipe(
        name=_name(data["name"]),
        dataset=_project_path(data["dataset"], "dataset"),
        base_model=base_model,
        epochs=_integer(data["epochs"], "epochs", 1),
        image_size=image_size,
        batch=_integer(data["batch"], "batch", 1),
        patience=_integer(data.get("patience", 50), "patience", 0),
        seed=_integer(data["seed"], "seed"),
        device=device,
        ultralytics=dict(extra),
        evaluation=evaluation_settings_from_dict(data.get("evaluation")),
        source=data,
    )


# -------------------------------------------------------------- comuni

def _name(value) -> str:
    if not isinstance(value, str) or not value or any(
        not (character.isalnum() or character in "_-.") for character in value
    ):
        raise RecipeError("name: usare solo lettere, cifre, '_', '-' e '.'")
    return value


def _project_path(value, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RecipeError(f"{name}: atteso un percorso")
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_recipe(path: str | Path) -> DatasetRecipe | TrainingRecipe:
    """Legge una ricetta di dataset o di training dal suo campo `kind`."""
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise RecipeError(f"{path.name}: JSON non valido ({error})") from error
    if not isinstance(data, dict) or data.get("kind") not in ("dataset", "training"):
        raise RecipeError(f"{path.name}: kind deve essere 'dataset' o 'training'")
    if data["kind"] == "dataset":
        return dataset_recipe_from_dict(data)
    return training_recipe_from_dict(data)


def available_recipes(kind: str, directory: Path = RECIPE_DIR) -> list[tuple[Path, dict]]:
    """Ricette di un certo tipo presenti nella cartella, in ordine di nome."""
    result = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("kind") == kind:
            result.append((path, data))
    return result
