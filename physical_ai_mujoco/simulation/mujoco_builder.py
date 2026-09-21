from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import mujoco
import numpy as np

from physical_ai_mujoco.scene.scene_description import (
    GroundDescription,
    principal_inertia,
    ObjectDescription,
    SceneDescription,
    StereoCameraDescription,
)


def build_mjcf(scene: SceneDescription) -> str:
    root = ElementTree.Element("mujoco", model=scene.scene_id)
    # `balanceinertia` non e' impostato di proposito: correggerebbe in silenzio
    # inerzie non fisiche invece di farle emergere come errore di compilazione.
    ElementTree.SubElement(
        root,
        "compiler",
        angle="radian",
        autolimits="true",
    )
    ElementTree.SubElement(
        root,
        "option",
        timestep=_values(scene.simulation.timestep),
        gravity=_values(scene.simulation.gravity),
        integrator=scene.simulation.integrator,
        impratio=_values(scene.simulation.impratio),
        cone=scene.simulation.cone,
        noslip_iterations=str(int(scene.simulation.noslip_iterations)),
    )

    # La rigidezza del contatto si dichiara una volta sola qui invece che su
    # ogni geom: e' una proprieta' della simulazione, non dell'oggetto. Lo
    # smorzamento resta critico (1): sotto, i contatti rimbalzano; sopra, si
    # ammorbidiscono e la compenetrazione torna a crescere.
    defaults = ElementTree.SubElement(root, "default")
    ElementTree.SubElement(
        defaults,
        "geom",
        solref=f"{scene.simulation.contact_timeconst:.10g} 1",
    )

    # Aspetto della scena. Il viewer e' sempre lo stesso: e' QUESTA roba a
    # dare l'aria "da tutorial MuJoCo" — cielo sfumato, foschia, pavimento a
    # scacchi con riflessi, ombre morbide.
    visual = ElementTree.SubElement(root, "visual")
    ElementTree.SubElement(
        visual,
        "headlight",
        diffuse="0.6 0.6 0.6",
        ambient="0.3 0.3 0.3",
        specular="0 0 0",
    )
    ElementTree.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")
    ElementTree.SubElement(visual, "quality", shadowsize="4096")
    ElementTree.SubElement(visual, "map", shadowclip="1.5", shadowscale="0.8")

    # <global> puo' comparire una sola volta: angolo di vista iniziale e
    # dimensione del framebuffer offscreen stanno nello stesso elemento.
    # (Il default offscreen e' 640x480: senza allargarlo, rendere a
    # risoluzioni maggiori fallisce.)
    global_options = {"azimuth": "130", "elevation": "-25"}
    if scene.stereo_camera is not None:
        global_options["offwidth"] = str(scene.stereo_camera.width)
        global_options["offheight"] = str(scene.stereo_camera.height)
    ElementTree.SubElement(visual, "global", **global_options)

    asset = ElementTree.SubElement(root, "asset")
    _add_mesh_assets(asset, scene)
    ElementTree.SubElement(
        asset,
        "texture",
        name="skybox",
        type="skybox",
        builtin="gradient",
        rgb1="0.3 0.5 0.7",
        rgb2="0 0 0",
        width="512",
        height="3072",
    )
    ElementTree.SubElement(
        asset,
        "texture",
        name="griglia",
        type="2d",
        builtin="checker",
        mark="edge",
        rgb1="0.2 0.3 0.4",
        rgb2="0.1 0.2 0.3",
        markrgb="0.8 0.8 0.8",
        width="300",
        height="300",
    )
    ElementTree.SubElement(
        asset,
        "material",
        name="mat_piano",
        texture="griglia",
        # Un quadretto ogni ~12 cm: con il terreno da 1 m, texrepeat=5 dava
        # scacchi enormi rispetto agli oggetti.
        texrepeat="8 8",
        texuniform="true",
        reflectance="0.2",
    )

    worldbody = ElementTree.SubElement(root, "worldbody")
    ElementTree.SubElement(
        worldbody,
        "light",
        name="sole",
        pos="0.6 -0.6 2.5",
        dir="-0.2 0.2 -1",
        directional="true",
        castshadow="true",
        diffuse="0.5 0.5 0.5",
        specular="0.1 0.1 0.1",
    )
    _add_ground(worldbody, scene)
    for item in scene.objects:
        _add_object(worldbody, item)
    if scene.stereo_camera is not None:
        _add_stereo_cameras(worldbody, scene.stereo_camera)
    _add_overview_camera(worldbody, scene)

    return ElementTree.tostring(root, encoding="unicode")


def build_model(scene: SceneDescription) -> tuple[mujoco.MjModel, mujoco.MjData]:
    assets = {
        _mesh_virtual_file(item): Path(item.mesh_file).read_bytes()
        for item in scene.objects
        if item.shape == "mesh" and item.mesh_file is not None
    }
    model = mujoco.MjModel.from_xml_string(build_mjcf(scene), assets)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _add_mesh_assets(asset: ElementTree.Element, scene: SceneDescription) -> None:
    for item in scene.objects:
        if item.shape != "mesh":
            continue
        if item.mesh_file is None:
            raise ValueError(f"Mesh file mancante per {item.instance_id}")
        ElementTree.SubElement(
            asset,
            "mesh",
            name=_mesh_asset_name(item),
            file=_mesh_virtual_file(item),
            scale=_values(item.mesh_scale),
        )


def _mesh_asset_name(item: ObjectDescription) -> str:
    return f"{item.instance_id}_mesh"


def _mesh_virtual_file(item: ObjectDescription) -> str:
    return f"{item.instance_id}.stl"


def _add_ground(worldbody: ElementTree.Element, scene: SceneDescription) -> None:
    ground = scene.ground
    body = ElementTree.SubElement(
        worldbody,
        "body",
        name="ground",
        pos=_values(ground.pose.position),
        quat=_values(ground.pose.quaternion),
    )
    ElementTree.SubElement(
        body,
        "geom",
        name="ground_geom",
        type=ground.shape,
        size=_values(_ground_size(ground)),
        condim="6",
        friction=_values(ground.friction),
        # Il materiale a scacchi sostituisce l'rgba del dataset: quel grigio
        # piatto tingerebbe la texture e ne spegnerebbe il contrasto.
        material="mat_piano",
    )


def _ground_size(ground: GroundDescription) -> tuple[float, float, float]:
    """L'attributo `size` del geom del terreno, che dipende dalla forma.

    Per un **box** sono le tre semi-dimensioni: e' l'ingombro reale, bordi
    compresi. Per un **plane** MuJoCo interpreta i tre valori come meta'
    estensione x, meta' estensione y e passo della griglia, e li usa SOLO per
    disegnare: nelle collisioni un plane e' infinito. Da qui la differenza che
    conta per noi — da un tavolo si cade, da un pavimento no.
    """
    if ground.shape == "plane":
        return (ground.size["x"] / 2.0, ground.size["y"] / 2.0, ground.size["z"])
    return (
        ground.size["x"] / 2.0,
        ground.size["y"] / 2.0,
        ground.size["z"] / 2.0,
    )


def _add_object(
    worldbody: ElementTree.Element,
    item: ObjectDescription,
) -> None:
    body = ElementTree.SubElement(
        worldbody,
        "body",
        name=item.instance_id,
        pos=_values(item.pose.position),
        quat=_values(item.pose.quaternion),
    )
    ElementTree.SubElement(
        body,
        "freejoint",
        name=f"{item.instance_id}_joint",
    )

    geom_attributes = {"mass": _values(item.mass)}
    if item.has_offset_mass:
        # Con un centro di massa spostato l'inerzia va dichiarata: il
        # compilatore la ricaverebbe dalla geometria, e la geometria dice
        # "uniforme". `<inertial>` prevale su qualunque massa dei geom, quindi
        # quella del geom si toglie per non lasciare due fonti in disaccordo.
        geom_attributes = {}
        ElementTree.SubElement(
            body,
            "inertial",
            pos=_values(item.center_of_mass),
            mass=_values(item.mass),
            diaginertia=_values(
                principal_inertia(item.shape, item.size, item.mass)
            ),
        )

    shape_attributes = (
        {"type": "mesh", "mesh": _mesh_asset_name(item)}
        if item.shape == "mesh"
        else {"type": item.shape, "size": _values(_mujoco_size(item))}
    )
    ElementTree.SubElement(
        body,
        "geom",
        name=f"{item.instance_id}_geom",
        **shape_attributes,
        **geom_attributes,
        condim="6",
        friction=_values(item.friction),
        rgba=_values(item.rgba),
    )


def _add_stereo_cameras(
    worldbody: ElementTree.Element,
    rig: StereoCameraDescription,
) -> None:
    """Aggiunge `cam_left` e `cam_right`, parallele e separate da `baseline`.

    Le due camere hanno lo STESSO orientamento (rig rettificato): l'immagine
    destra e' l'immagine sinistra traslata, quindi la disparita' e' puramente
    orizzontale ed e' legata alla profondita' da z = f * B / d.
    """
    eye = np.asarray(rig.position, dtype=float)
    target = np.asarray(rig.target, dtype=float)
    right, up = _camera_axes(eye, target)

    half = rig.baseline / 2.0
    for name, offset in (("cam_left", -half), ("cam_right", +half)):
        ElementTree.SubElement(
            worldbody,
            "camera",
            name=name,
            mode="fixed",
            fovy=_values(rig.fovy),
            pos=_values(tuple(eye + right * offset)),
            # MuJoCo: xyaxes = asse x (destra) e asse y (alto) del frame camera.
            # La camera guarda lungo -z, cioe' verso `forward`.
            xyaxes=_values(tuple(right) + tuple(up)),
        )


def _add_overview_camera(
    worldbody: ElementTree.Element,
    scene: SceneDescription,
) -> None:
    """Camera panoramica `cam_overview`, sempre presente.

    Serve alla visualizzazione (il monitor a griglia), non all'agente: le
    camere dello student restano `cam_left` / `cam_right`. E' fissa e inquadra
    tutto il terreno, cosi' scene diverse restano confrontabili a colpo
    d'occhio quando sono affiancate.
    """
    # L'inquadratura segue gli OGGETTI, non il terreno: su un pavimento
    # infinito la dimensione del terreno e' una scelta di disegno e non dice
    # piu' niente su quanto e' grande la scena.
    extent = max(
        (
            max(abs(item.pose.position[0]), abs(item.pose.position[1]))
            for item in scene.objects
        ),
        default=0.0,
    )
    distance = max(2.0 * extent, 0.9) * 1.2

    _add_lookat_camera(
        worldbody,
        name="cam_overview",
        eye=np.array([distance * 0.75, -distance * 0.75, distance * 0.65]),
        target=np.array([0.0, 0.0, 0.05]),
        fovy=45.0,
    )


def _add_lookat_camera(
    worldbody: ElementTree.Element,
    name: str,
    eye: np.ndarray,
    target: np.ndarray,
    fovy: float,
) -> None:
    right, up = _camera_axes(eye, target)
    ElementTree.SubElement(
        worldbody,
        "camera",
        name=name,
        mode="fixed",
        fovy=_values(fovy),
        pos=_values(tuple(eye)),
        xyaxes=_values(tuple(right) + tuple(up)),
    )


def _camera_axes(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assi x (destra) e y (alto) del frame camera che guarda `target`."""
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        raise ValueError("Una camera non puo' guardare la propria posizione")
    forward = forward / norm

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, world_up))) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    return right, up


def _mujoco_size(item: ObjectDescription) -> tuple[float, ...]:
    if item.shape == "box":
        return (
            item.size["x"] / 2.0,
            item.size["y"] / 2.0,
            item.size["z"] / 2.0,
        )
    if item.shape == "cylinder":
        return (item.size["radius"], item.size["height"] / 2.0)
    if item.shape == "sphere":
        return (item.size["radius"],)
    raise ValueError(f"Unsupported shape: {item.shape}")


def _values(values) -> str:
    if isinstance(values, (int, float)):
        return f"{values:.10g}"
    return " ".join(f"{value:.10g}" for value in values)
