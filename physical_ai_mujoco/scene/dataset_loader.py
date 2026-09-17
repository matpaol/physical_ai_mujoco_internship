from __future__ import annotations

import json
from pathlib import Path


OBJECT_SIZE_FIELDS = {
    "box": {"x", "y", "z"},
    "cylinder": {"radius", "height"},
    "sphere": {"radius"},
}

# "box" e' un tavolo (ha bordi), "plane" un pavimento (infinito nelle
# collisioni). La differenza e' documentata in GroundDescription.
SUPPORTED_GROUND_SHAPES = {"box", "plane"}

# "simultaneous": tutti gli oggetti cadono insieme da quote diverse, e devono
# quindi partire gia' distanti fra loro. "sequential": cadono uno alla volta da
# poco sopra la cima della pila, quindi non si sovrappongono mai e l'impatto
# resta debole.
SUPPORTED_RELEASE_MODES = {"simultaneous", "sequential"}


def load_object_dataset(path: str | Path) -> dict:
    data = _load_json(path)
    object_types = _require_list(data, "object_types")
    ids = set()

    for item in object_types:
        _require_keys(
            item,
            "id",
            "shape",
            "size_range",
            "density_range",
            "friction",
            "rgba",
        )

        type_id = item["id"]
        if not isinstance(type_id, str) or not type_id:
            raise ValueError("Object type id must be a non-empty string")
        if type_id in ids:
            raise ValueError(f"Duplicate object type id: {type_id}")
        ids.add(type_id)

        shape = item["shape"]
        if shape not in OBJECT_SIZE_FIELDS:
            raise ValueError(f"Unsupported object shape: {shape}")

        size_range = item["size_range"]
        if set(size_range) != OBJECT_SIZE_FIELDS[shape]:
            raise ValueError(f"Invalid size fields for object type: {type_id}")
        for name, bounds in size_range.items():
            _validate_range(bounds, f"{type_id}.{name}", positive=True)

        _validate_range(
            item["density_range"],
            f"{type_id}.density_range",
            positive=True,
        )

        friction = item["friction"]
        _require_keys(friction, "sliding_range", "torsional", "rolling")
        _validate_range(
            friction["sliding_range"],
            f"{type_id}.sliding_range",
            positive=False,
        )
        _validate_non_negative(friction["torsional"], f"{type_id}.torsional")
        _validate_non_negative(friction["rolling"], f"{type_id}.rolling")
        _validate_rgba(item["rgba"], type_id)

    return data


def load_ground_dataset(path: str | Path) -> dict:
    data = _load_json(path)
    ground_types = _require_list(data, "ground_types")
    ids = set()

    for item in ground_types:
        _require_keys(
            item,
            "id",
            "shape",
            "size",
            "inclination",
            "friction",
            "roughness",
            "rgba",
        )

        type_id = item["id"]
        if not isinstance(type_id, str) or not type_id:
            raise ValueError("Ground type id must be a non-empty string")
        if type_id in ids:
            raise ValueError(f"Duplicate ground type id: {type_id}")
        ids.add(type_id)

        # "box" e' un tavolo, con bordi da cui si puo' cadere; "plane" e' un
        # pavimento, infinito nelle collisioni. Vedi GroundDescription.
        if item["shape"] not in SUPPORTED_GROUND_SHAPES:
            raise ValueError(
                f"Unsupported ground shape: {item['shape']}. "
                f"Supported: {', '.join(sorted(SUPPORTED_GROUND_SHAPES))}"
            )

        size = item["size"]
        if set(size) != {"x", "y", "z"}:
            raise ValueError(f"Invalid size fields for ground type: {type_id}")
        for name, value in size.items():
            _validate_positive(value, f"{type_id}.{name}")

        inclination = item["inclination"]
        _require_keys(inclination, "x", "y")
        _validate_number(inclination["x"], f"{type_id}.inclination.x")
        _validate_number(inclination["y"], f"{type_id}.inclination.y")

        friction = item["friction"]
        _require_keys(friction, "sliding", "torsional", "rolling")
        for name, value in friction.items():
            _validate_non_negative(value, f"{type_id}.{name}")

        roughness = item["roughness"]
        _require_keys(roughness, "type")
        if roughness["type"] != "none":
            raise ValueError("Phase 0A supports only flat ground")

        _validate_rgba(item["rgba"], type_id)

    return data


def load_scene_rules(path: str | Path) -> dict:
    data = _load_json(path)
    _require_keys(
        data,
        "ground_selection",
        "object_selection",
        "spawn",
        "release_mode",
    )

    _require_keys(data["ground_selection"], "type_id")
    object_selection = data["object_selection"]
    object_ids = _require_list(object_selection, "required_type_ids")
    if not object_ids:
        raise ValueError("At least one object type is required")
    # I tipi possono ripetersi: la lista descrive ISTANZE, non un insieme di
    # tipi. Due "box" nella lista sono due scatole distinte, con dimensioni e
    # densita' campionate indipendentemente.

    spawn = data["spawn"]
    _require_keys(
        spawn,
        "x_range",
        "y_range",
        "z_range",
        "random_orientation",
        "allow_initial_overlap",
        "edge_margin",
        "max_attempts_per_object",
    )
    for name in ("x_range", "y_range", "z_range"):
        _validate_range(spawn[name], name, positive=False)

    if not isinstance(spawn["random_orientation"], bool):
        raise ValueError("random_orientation must be a boolean")
    # Con rilascio sequenziale gli oggetti non coesistono mai in aria, quindi
    # non c'e' nulla da non sovrapporre: il controllo va disattivato o il
    # campionamento fallisce inutilmente. Con rilascio simultaneo, invece, e'
    # l'unica cosa che impedisce di partire compenetrati.
    if not isinstance(spawn["allow_initial_overlap"], bool):
        raise ValueError("allow_initial_overlap must be a boolean")
    if spawn["allow_initial_overlap"] and data["release_mode"] == "simultaneous":
        raise ValueError(
            "allow_initial_overlap richiede release_mode 'sequential': "
            "rilasciando tutto insieme gli oggetti partirebbero compenetrati"
        )
    _validate_non_negative(spawn["edge_margin"], "edge_margin")

    attempts = spawn["max_attempts_per_object"]
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
        raise ValueError("max_attempts_per_object must be a positive integer")

    if data["release_mode"] not in SUPPORTED_RELEASE_MODES:
        raise ValueError(
            f"Unsupported release mode: {data['release_mode']}. "
            f"Supported: {', '.join(sorted(SUPPORTED_RELEASE_MODES))}"
        )

    return data


def load_simulation_config(path: str | Path) -> dict:
    data = _load_json(path)
    _require_keys(data, "gravity", "timestep", "integrator", "settling")

    gravity = data["gravity"]
    if not isinstance(gravity, list) or len(gravity) != 3:
        raise ValueError("gravity must contain three values")
    for value in gravity:
        _validate_number(value, "gravity")

    _validate_positive(data["timestep"], "timestep")
    if data["integrator"] not in {"Euler", "RK4", "implicit", "implicitfast"}:
        raise ValueError(f"Unsupported integrator: {data['integrator']}")

    settling = data["settling"]
    _require_keys(
        settling,
        "linear_tolerance",
        "angular_tolerance",
        "stable_duration",
        "timeout",
    )
    for name, value in settling.items():
        _validate_positive(value, name)

    return data


def validate_references(
    object_dataset: dict,
    ground_dataset: dict,
    scene_rules: dict,
) -> None:
    object_ids = {item["id"] for item in object_dataset["object_types"]}
    required_ids = set(scene_rules["object_selection"]["required_type_ids"])
    missing_objects = required_ids - object_ids
    if missing_objects:
        names = ", ".join(sorted(missing_objects))
        raise ValueError(f"Unknown object type ids: {names}")

    ground_ids = {item["id"] for item in ground_dataset["ground_types"]}
    ground_id = scene_rules["ground_selection"]["type_id"]
    if ground_id not in ground_ids:
        raise ValueError(f"Unknown ground type id: {ground_id}")


def _load_json(path: str | Path) -> dict:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"File not found: {input_path}")

    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {input_path}: {error}") from error

    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {input_path}")
    return data


def _require_keys(data: dict, *keys: str) -> None:
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    missing = [key for key in keys if key not in data]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Missing required fields: {names}")


def _require_list(data: dict, key: str) -> list:
    if key not in data or not isinstance(data[key], list):
        raise ValueError(f"{key} must be a list")
    return data[key]


def _validate_range(values: list, name: str, positive: bool) -> None:
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError(f"{name} must contain two values")
    lower, upper = values
    _validate_number(lower, name)
    _validate_number(upper, name)
    if lower > upper:
        raise ValueError(f"Invalid range for {name}")
    if positive and lower <= 0:
        raise ValueError(f"{name} values must be positive")
    if not positive and lower < 0 and name not in {"x_range", "y_range", "z_range"}:
        raise ValueError(f"{name} values must be non-negative")


def _validate_rgba(values: list, name: str) -> None:
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError(f"{name}.rgba must contain four values")
    for value in values:
        _validate_number(value, f"{name}.rgba")
        if not 0 <= value <= 1:
            raise ValueError(f"{name}.rgba values must be between 0 and 1")


def _validate_number(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")


def _validate_positive(value: float, name: str) -> None:
    _validate_number(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_non_negative(value: float, name: str) -> None:
    _validate_number(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")

