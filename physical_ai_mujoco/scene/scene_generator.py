from __future__ import annotations

import math

import numpy as np

from physical_ai_mujoco.scene.dataset_loader import validate_references
from physical_ai_mujoco.scene.scene_description import (
    bounding_radius,
    half_extents,
    GroundDescription,
    ObjectDescription,
    Pose,
    SceneDescription,
    SimulationDescription,
    StereoCameraDescription,
)


def generate_scene(
    object_dataset: dict,
    ground_dataset: dict,
    scene_rules: dict,
    simulation_config: dict,
    seed: int,
    stereo_config: dict | None = None,
) -> SceneDescription:
    validate_references(object_dataset, ground_dataset, scene_rules)
    rng = np.random.default_rng(seed)

    object_types = {
        item["id"]: item for item in object_dataset["object_types"]
    }
    ground_types = {
        item["id"]: item for item in ground_dataset["ground_types"]
    }

    ground_type_id = scene_rules["ground_selection"]["type_id"]
    ground = _create_ground(ground_types[ground_type_id])
    objects = _create_objects(
        object_types,
        scene_rules,
        ground,
        rng,
    )
    simulation = _create_simulation(simulation_config)

    return SceneDescription(
        schema_version=1,
        scene_id=f"phase0a_seed_{seed}",
        seed=seed,
        ground=ground,
        objects=tuple(objects),
        simulation=simulation,
        stereo_camera=_create_stereo_camera(stereo_config),
    )


def _create_stereo_camera(
    config: dict | None,
) -> StereoCameraDescription | None:
    if config is None:
        return None
    return StereoCameraDescription(
        baseline=float(config["baseline"]),
        width=int(config["width"]),
        height=int(config["height"]),
        fovy=float(config["fovy"]),
        position=tuple(float(value) for value in config["position"]),
        target=tuple(float(value) for value in config["target"]),
    )


def resample_object_poses(
    scene: SceneDescription,
    scene_rules: dict,
    rng: np.random.Generator,
) -> tuple[Pose, ...]:
    """Nuove pose iniziali per gli oggetti di una scena esistente.

    Le FORME restano quelle che sono: cambiano solo posizione e orientamento.
    Serve a rimescolare una scena senza ricompilare il modello MuJoCo, cosi'
    che una finestra interattiva gia' aperta resti valida fra un episodio e
    l'altro (un viewer e' legato a un modello compilato).
    """
    spawn = scene_rules["spawn"]
    placed: list[ObjectDescription] = []
    poses = []

    for item in scene.objects:
        radius = bounding_radius(item.shape, item.size)
        position = _sample_position(spawn, scene.ground, radius, placed, rng)
        if spawn["random_orientation"]:
            quaternion = _sample_quaternion(rng)
        else:
            quaternion = (1.0, 0.0, 0.0, 0.0)

        poses.append(Pose(position=position, quaternion=quaternion))
        # Si registra la posa appena scelta perche' il controllo di
        # non sovrapposizione la veda.
        placed.append(
            ObjectDescription(
                instance_id=item.instance_id,
                type_id=item.type_id,
                shape=item.shape,
                size=item.size,
                density=item.density,
                mass=item.mass,
                friction=item.friction,
                rgba=item.rgba,
                pose=poses[-1],
                center_of_mass=item.center_of_mass,
            )
        )

    return tuple(poses)


def _create_ground(type_definition: dict) -> GroundDescription:
    shape = type_definition["shape"]
    size = {name: float(value) for name, value in type_definition["size"].items()}
    inclination = type_definition["inclination"]
    quaternion = _quaternion_from_xy_angles(
        float(inclination["x"]),
        float(inclination["y"]),
    )
    friction = type_definition["friction"]

    # In entrambi i casi la superficie di appoggio finisce a z=0. Un box ha
    # spessore, quindi il corpo va abbassato di meta'; un plane non ne ha.
    thickness = 0.0 if shape == "plane" else size["z"] / 2.0

    return GroundDescription(
        type_id=type_definition["id"],
        shape=shape,
        size=size,
        friction=(
            float(friction["sliding"]),
            float(friction["torsional"]),
            float(friction["rolling"]),
        ),
        roughness_type=type_definition["roughness"]["type"],
        rgba=tuple(float(value) for value in type_definition["rgba"]),
        pose=Pose(
            position=(0.0, 0.0, -thickness),
            quaternion=quaternion,
        ),
    )


def _create_objects(
    object_types: dict[str, dict],
    scene_rules: dict,
    ground: GroundDescription,
    rng: np.random.Generator,
) -> list[ObjectDescription]:
    objects = []
    required_ids = scene_rules["object_selection"]["required_type_ids"]

    for index, type_id in enumerate(required_ids):
        type_definition = object_types[type_id]
        size = {
            name: float(rng.uniform(bounds[0], bounds[1]))
            for name, bounds in type_definition["size_range"].items()
        }
        density = float(rng.uniform(*type_definition["density_range"]))
        mass = density * _object_volume(type_definition["shape"], size)
        friction = type_definition["friction"]
        sliding = float(rng.uniform(*friction["sliding_range"]))
        radius = bounding_radius(type_definition["shape"], size)
        position = _sample_position(
            scene_rules["spawn"],
            ground,
            radius,
            objects,
            rng,
        )

        if scene_rules["spawn"]["random_orientation"]:
            quaternion = _sample_quaternion(rng)
        else:
            quaternion = (1.0, 0.0, 0.0, 0.0)

        objects.append(
            ObjectDescription(
                center_of_mass=_sample_center_of_mass(
                    type_definition["shape"], size, scene_rules, rng
                ),
                instance_id=f"object_{index:03d}",
                type_id=type_id,
                shape=type_definition["shape"],
                size=size,
                density=density,
                mass=mass,
                friction=(
                    sliding,
                    float(friction["torsional"]),
                    float(friction["rolling"]),
                ),
                rgba=tuple(float(value) for value in type_definition["rgba"]),
                pose=Pose(position=position, quaternion=quaternion),
            )
        )

    return objects


def _sample_center_of_mass(
    shape: str,
    size: dict[str, float],
    scene_rules: dict,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Scostamento del centro di massa dal centro geometrico.

    `center_of_mass_jitter` e' una FRAZIONE della semi-dimensione su ciascun
    asse, cosi' la stessa regola vale per una lastra sottile e per una scatola
    cubica senza spostare la massa fuori dall'oggetto. A 0 (il default) tutti i
    solidi restano uniformi e il comportamento e' identico a prima.

    Perche' serve: un oggetto reale non ha la massa al centro geometrico, e su
    un compito che dipende da cosa si ribalta e cosa no il centro di massa
    conta piu' della massa stessa. Randomizzarlo e' anche una forma di domain
    randomization su un parametro che nel reale non si misura mai — molto piu'
    difendibile che randomizzare quelli che si potrebbero misurare.
    """
    jitter = float(
        scene_rules.get("randomisation", {}).get("center_of_mass_jitter", 0.0)
    )
    if jitter <= 0.0:
        return (0.0, 0.0, 0.0)
    return tuple(
        float(rng.uniform(-jitter, jitter) * extent)
        for extent in half_extents(shape, size)
    )


def _create_simulation(config: dict) -> SimulationDescription:
    settling = config["settling"]
    contact = config.get("contact", {})
    return SimulationDescription(
        gravity=tuple(float(value) for value in config["gravity"]),
        timestep=float(config["timestep"]),
        integrator=config["integrator"],
        impratio=float(contact.get("impratio", 1.0)),
        cone=str(contact.get("cone", "pyramidal")),
        noslip_iterations=int(contact.get("noslip_iterations", 0)),
        contact_timeconst=float(contact.get("timeconst", 0.02)),
        linear_settle_tolerance=float(settling["linear_tolerance"]),
        angular_settle_tolerance=float(settling["angular_tolerance"]),
        stable_duration=float(settling["stable_duration"]),
        timeout=float(settling["timeout"]),
    )


def _sample_position(
    spawn: dict,
    ground: GroundDescription,
    radius: float,
    existing_objects: list[ObjectDescription],
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    x_bounds = _safe_axis_range(spawn, ground, radius, "x")
    y_bounds = _safe_axis_range(spawn, ground, radius, "y")
    z_bounds = spawn["z_range"]

    for _ in range(spawn["max_attempts_per_object"]):
        # Si campiona nel frame del TERRENO, poi si trasforma nel frame mondo.
        # Con terreno orizzontale la rotazione e' l'identita' e il risultato e'
        # identico a prima; con terreno inclinato gli oggetti cadono comunque
        # sopra la superficie invece che accanto ad essa.
        local_position = np.array(
            [
                float(rng.uniform(*x_bounds)),
                float(rng.uniform(*y_bounds)),
                float(rng.uniform(*z_bounds)),
            ]
        )
        position = _ground_frame_to_world(local_position, ground)
        # Con rilascio sequenziale gli oggetti non sono mai in scena insieme,
        # quindi le pose iniziali possono sovrapporsi: la prima estrazione va
        # sempre bene.
        if spawn["allow_initial_overlap"]:
            return position
        if _position_is_free(position, radius, existing_objects):
            return position

    raise RuntimeError(
        "Impossibile campionare una posizione libera: la colonna di rilascio e' "
        "troppo stretta o troppo bassa per il numero di oggetti richiesto. "
        "Con release_mode 'sequential' il problema non si pone."
    )


def _ground_frame_to_world(
    local_position: np.ndarray,
    ground: GroundDescription,
) -> tuple[float, float, float]:
    """Porta una posizione dal frame del terreno al frame mondo.

    Il frame del terreno ha origine sulla SUPERFICIE di appoggio, cosi' che z
    locale sia l'altezza di rilascio sopra il piano. Per un box la superficie
    e' mezzo spessore sopra l'origine del corpo; per un plane coincide con
    essa.
    """
    rotation = _rotation_matrix(ground.pose.quaternion)
    surface_offset = rotation @ np.array([0.0, 0.0, ground.surface_thickness])
    origin = np.asarray(ground.pose.position) + surface_offset
    world = origin + rotation @ local_position
    return tuple(float(value) for value in world)


def _rotation_matrix(
    quaternion: tuple[float, float, float, float],
) -> np.ndarray:
    """Matrice di rotazione da un quaternione MuJoCo (w, x, y, z)."""
    w, x, y, z = (float(value) for value in quaternion)
    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ]
    )


def _safe_axis_range(
    spawn: dict,
    ground: GroundDescription,
    radius: float,
    axis_name: str,
) -> tuple[float, float]:
    """Intervallo di rilascio su un asse, ristretto per non sporgere dal bordo.

    Su un pavimento infinito non ci sono bordi: l'intervallo richiesto passa
    intatto, e l'area di rilascio resta una decisione delle sole regole di
    scena. E' una separazione che conta — legare l'area al terreno significa
    che allargare la scena richiede di cambiare il terreno.
    """
    requested_range = spawn[f"{axis_name}_range"]
    lower, upper = float(requested_range[0]), float(requested_range[1])

    if ground.is_bounded:
        limit = ground.size[axis_name] / 2.0 - radius - float(spawn["edge_margin"])
        lower, upper = max(lower, -limit), min(upper, limit)

    if lower > upper:
        raise ValueError(
            f"Nessun intervallo di rilascio valido sull'asse {axis_name}: "
            f"il terreno '{ground.type_id}' e' troppo piccolo per un oggetto "
            f"di raggio {radius:.3f} m"
        )
    return lower, upper


def _position_is_free(
    position: tuple[float, float, float],
    radius: float,
    existing_objects: list[ObjectDescription],
) -> bool:
    candidate = np.asarray(position)
    for other in existing_objects:
        other_position = np.asarray(other.pose.position)
        other_radius = bounding_radius(other.shape, other.size)
        if np.linalg.norm(candidate - other_position) < radius + other_radius:
            return False
    return True


def _object_volume(shape: str, size: dict[str, float]) -> float:
    if shape == "box":
        return size["x"] * size["y"] * size["z"]
    if shape == "cylinder":
        return math.pi * size["radius"] ** 2 * size["height"]
    if shape == "sphere":
        return 4.0 * math.pi * size["radius"] ** 3 / 3.0
    raise ValueError(f"Unsupported shape: {shape}")


def _sample_quaternion(
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def _quaternion_from_xy_angles(
    angle_x: float,
    angle_y: float,
) -> tuple[float, float, float, float]:
    half_x = angle_x / 2.0
    half_y = angle_y / 2.0
    cos_x = math.cos(half_x)
    sin_x = math.sin(half_x)
    cos_y = math.cos(half_y)
    sin_y = math.sin(half_y)
    return (
        cos_x * cos_y,
        sin_x * cos_y,
        cos_x * sin_y,
        -sin_x * sin_y,
    )

