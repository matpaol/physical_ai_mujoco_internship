"""Ground-truth di esposizione usato solo da dataset e benchmark OSSERVA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physical_ai_mujoco.scene.scene_description import Pose


@dataclass(frozen=True)
class TargetExposure:
    immersion_fraction: float
    geometric_exposure_fraction: float
    ground_visible_fraction: float
    camera_visible_fraction: float
    obstacle_visibility_fraction: float
    visible_pixels: int
    ground_only_pixels: int
    fully_exposed_pixels: int


@dataclass(frozen=True)
class ImmersionPlacement:
    immersed_pose: Pose
    fully_exposed_pose: Pose
    vertical_extent: float


def _rotation_matrix(quaternion) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def place_target_at_immersion(
    simulator, target_id: str, fraction: float, *, settle_others: bool = False
) -> ImmersionPlacement:
    """Posiziona il target nella superficie; di default senza avanzare la fisica.

    E' una trasformazione per studi percettivi: dopo averla applicata si
    acquisiscono subito i sensori. Non sostituisce ancora un modello fisico
    del terreno deformabile per gli episodi di manipolazione.

    Abbassando il target, gli oggetti che gli poggiavano sopra restano sospesi
    nel vuoto. Con `settle_others=True` il target viene tenuto fermo e il resto
    della pila ricade e si assesta: le ombre e gli oggetti a mezz'aria non
    segnalano piu' al detector dove sta il target.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("La frazione di immersione deve essere in [0, 1]")
    state = simulator.get_object_state(target_id)
    ground = simulator.scene.ground
    ground_rotation = _rotation_matrix(ground.pose.quaternion)
    normal = ground_rotation[:, 2]
    surface_origin = (
        np.asarray(ground.pose.position, dtype=float)
        + normal * ground.surface_thickness
    )
    lower, upper = simulator.object_projection_bounds(target_id, normal)
    vertical_extent = float(upper - lower)
    if vertical_extent <= 0:
        raise ValueError("Il target deve avere estensione positiva lungo la normale")

    current = np.asarray(state.position, dtype=float)
    surface_projection = float(normal @ surface_origin)
    fully_exposed = current + normal * (surface_projection - lower)
    immersed = fully_exposed - normal * (vertical_extent * float(fraction))
    full_pose = Pose(tuple(float(v) for v in fully_exposed), state.quaternion)
    immersed_pose = Pose(tuple(float(v) for v in immersed), state.quaternion)
    simulator.set_object_pose(target_id, immersed_pose.position, immersed_pose.quaternion)
    if settle_others:
        simulator.hold_object(target_id)
        try:
            simulator.step_until_settled()
        finally:
            simulator.release_object(target_id)
    return ImmersionPlacement(immersed_pose, full_pose, vertical_extent)


def measure_target_exposure(
    simulator,
    target_id: str,
    immersion_fraction: float,
    placement: ImmersionPlacement,
    *,
    camera_name: str = "cam_left",
) -> TargetExposure:
    """Separa copertura del terreno e occlusione da parte degli oggetti."""
    def target_pixels() -> int:
        mask = simulator.render_instance_masks(camera_name).get(target_id)
        return 0 if mask is None else int(np.asarray(mask, dtype=bool).sum())

    visible_pixels = target_pixels()
    snapshot = simulator.snapshot()
    try:
        for instance_id in tuple(simulator.present_objects()):
            if instance_id != target_id:
                simulator.remove_object(instance_id)
        ground_only_pixels = target_pixels()
        simulator.set_object_pose(
            target_id,
            placement.fully_exposed_pose.position,
            placement.fully_exposed_pose.quaternion,
        )
        fully_exposed_pixels = target_pixels()
    finally:
        simulator.restore(snapshot)

    ground_fraction = (
        ground_only_pixels / fully_exposed_pixels if fully_exposed_pixels else 0.0
    )
    camera_fraction = visible_pixels / fully_exposed_pixels if fully_exposed_pixels else 0.0
    obstacle_fraction = visible_pixels / ground_only_pixels if ground_only_pixels else 0.0
    return TargetExposure(
        immersion_fraction=float(immersion_fraction),
        geometric_exposure_fraction=1.0 - float(immersion_fraction),
        ground_visible_fraction=float(np.clip(ground_fraction, 0.0, 1.0)),
        camera_visible_fraction=float(np.clip(camera_fraction, 0.0, 1.0)),
        obstacle_visibility_fraction=float(np.clip(obstacle_fraction, 0.0, 1.0)),
        visible_pixels=visible_pixels,
        ground_only_pixels=ground_only_pixels,
        fully_exposed_pixels=fully_exposed_pixels,
    )


# Sotto questa profondita' il target non e' "interrato": e' la compenetrazione
# normale di un contatto appoggiato, che MuJoCo lascia di qualche decimo di mm.
BURIAL_TOLERANCE_M = 0.002


@dataclass(frozen=True)
class TargetVisibility:
    """Quanta parte della sagoma del target vede una camera, in questa scena.

    `visible_fraction` = pixel visibili ora / pixel che si vedrebbero se il
    target fosse da solo e non interrato, nella stessa posa e dalla stessa
    camera. Vale `None` se anche cosi' il target e' fuori inquadratura: in quel
    caso non ha senso parlare di percentuale visibile.
    """

    visible_pixels: int
    unoccluded_pixels: int
    visible_fraction: float | None


def measure_target_visibility(
    simulator, target_id: str, camera_name: str = "cam_left"
) -> TargetVisibility:
    """Misura la % visibile del target senza modificare la scena.

    Il denominatore si ottiene togliendo tutti gli altri oggetti e, solo se il
    target e' sotto la superficie, sollevandolo quanto basta a scoprirlo. Un
    target appoggiato su altri oggetti resta alla sua quota: spostarlo a terra
    cambierebbe la sua distanza dalla camera e quindi la sua dimensione in
    pixel. Tutto avviene dentro una fotografia dello stato, poi ripristinata.
    """

    def target_pixels() -> int:
        mask = simulator.render_instance_masks(camera_name).get(target_id)
        return 0 if mask is None else int(np.asarray(mask, dtype=bool).sum())

    visible_pixels = target_pixels()
    snapshot = simulator.snapshot()
    try:
        for instance_id in tuple(simulator.present_objects()):
            if instance_id != target_id:
                simulator.remove_object(instance_id)
        ground = simulator.scene.ground
        normal = _rotation_matrix(ground.pose.quaternion)[:, 2]
        surface = float(
            normal
            @ (
                np.asarray(ground.pose.position, dtype=float)
                + normal * ground.surface_thickness
            )
        )
        lower, _ = simulator.object_projection_bounds(target_id, normal)
        if surface - lower > BURIAL_TOLERANCE_M:
            state = simulator.get_object_state(target_id)
            lifted = np.asarray(state.position, dtype=float) + normal * (surface - lower)
            simulator.set_object_pose(target_id, lifted, state.quaternion)
        unoccluded_pixels = target_pixels()
    finally:
        simulator.restore(snapshot)

    fraction = (
        float(np.clip(visible_pixels / unoccluded_pixels, 0.0, 1.0))
        if unoccluded_pixels
        else None
    )
    return TargetVisibility(visible_pixels, unoccluded_pixels, fraction)
