"""Blocchi interni di OSSERVA, indipendenti dalla policy decisionale."""

from dataclasses import replace
import math

import numpy as np

from physical_ai_mujoco.contracts import (
    Observation,
    ObjectUncertainty,
    PerceptualObject,
    PerceptualState,
    PhysicalRelation,
    PhysicalRelationState,
    SceneObject,
    SceneState,
    SensorEvidence,
    SensorBundle,
    TaskContext,
    UncertaintyState,
    validate_observation,
)


def _size_xyz(item) -> tuple[float, float, float]:
    if item.shape in {"box", "mesh"}:
        return tuple(float(item.size[axis]) for axis in ("x", "y", "z"))
    if item.shape == "cylinder":
        return (2 * float(item.size["radius"]),) * 2 + (
            float(item.size["height"]),
        )
    if item.shape == "sphere":
        return (2 * float(item.size["radius"]),) * 3
    raise ValueError(f"Forma non supportata: {item.shape}")


class ExactStateExtractor:
    """Sorgente di riferimento 0B/1A; non produce dati fisici privilegiati."""

    def extract(self, simulator) -> tuple[PerceptualState, SensorEvidence]:
        timestamp = simulator.time
        objects = []
        for item in simulator.scene.objects:
            if not simulator.is_present(item.instance_id):
                continue
            state = simulator.get_object_state(item.instance_id)
            objects.append(
                PerceptualObject(
                    item.instance_id,
                    item.type_id,
                    tuple(state.position),
                    tuple(state.quaternion),
                    item.shape,
                    _size_xyz(item),
                    True,
                    1.0,
                )
            )
        return (
            PerceptualState(tuple(objects), "world", timestamp),
            SensorEvidence("world", timestamp, observed_fraction=1.0),
        )


class DegradedStateExtractor:
    """Sorgente MuJoCo alternativa con errori controllati e riproducibili.

    La confidence qui indica la qualita' nominale della sorgente sintetica;
    non e' una probabilita' calibrata su dati stereo.
    """

    def __init__(self, position_sigma=0.01, drop_probability=0.1):
        if position_sigma < 0 or not 0 <= drop_probability <= 1:
            raise ValueError("Parametri di degradazione non validi")
        self.position_sigma = float(position_sigma)
        self.drop_probability = float(drop_probability)
        self.rng = np.random.default_rng()
        self.exact = ExactStateExtractor()

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def extract(self, simulator) -> tuple[PerceptualState, SensorEvidence]:
        perceptual, evidence = self.exact.extract(simulator)
        objects = []
        for item in perceptual.objects:
            if self.rng.random() < self.drop_probability:
                continue
            noise = self.rng.normal(0.0, self.position_sigma, 3)
            position = tuple(float(a + b) for a, b in zip(item.position, noise))
            objects.append(
                replace(
                    item,
                    position=position,
                    confidence=max(0.0, 1.0 - self.drop_probability),
                )
            )
        observed_fraction = len(objects) / len(perceptual.objects) if perceptual.objects else 1.0
        return (
            replace(perceptual, objects=tuple(objects)),
            replace(evidence, observed_fraction=observed_fraction),
        )


class StereoBundleExtractor:
    """Triangola track ID associati nelle due viste rettificate.

    Il matching delle maschere appartiene al detector/tracker a monte. Una
    detection priva di corrispondenza o disparita' positiva resta ignota.
    """

    def extract(self, bundle: SensorBundle) -> tuple[PerceptualState, SensorEvidence]:
        left = np.asarray(bundle.rgb_left)
        right = np.asarray(bundle.rgb_right)
        K = np.asarray(bundle.intrinsics, dtype=float)
        world_from_left = np.asarray(bundle.world_from_left, dtype=float)
        if left.shape != right.shape or left.ndim != 3 or left.shape[2] != 3:
            raise ValueError("La coppia stereo RGB deve avere la stessa dimensione")
        if K.shape != (3, 3) or world_from_left.shape != (4, 4) or bundle.baseline <= 0:
            raise ValueError("Calibrazione stereo non valida")
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("Lunghezza focale non valida")
        objects = []
        for track_id in bundle.left_masks.keys() & bundle.right_masks.keys():
            left_mask = np.asarray(bundle.left_masks[track_id], dtype=bool)
            right_mask = np.asarray(bundle.right_masks[track_id], dtype=bool)
            if left_mask.shape != left.shape[:2] or right_mask.shape != right.shape[:2]:
                raise ValueError(f"Dimensione mask non valida per {track_id}")
            if not left_mask.any() or not right_mask.any():
                continue
            vl, ul = np.mean(np.argwhere(left_mask), axis=0)
            vr, ur = np.mean(np.argwhere(right_mask), axis=0)
            disparity = float(ul - ur)
            if disparity <= 0 or abs(vl - vr) > 2.0:
                continue
            depth = float(K[0, 0] * bundle.baseline / disparity)
            camera_point = np.asarray(
                [(ul - K[0, 2]) * depth / K[0, 0],
                 (vl - K[1, 2]) * depth / K[1, 1], depth, 1.0]
            )
            world_point = world_from_left @ camera_point
            objects.append(
                PerceptualObject(
                    track_id,
                    None if bundle.type_ids is None else bundle.type_ids.get(track_id),
                    tuple(float(v) for v in world_point[:3]),
                    None,
                    None,
                    None,
                    True,
                    None,
                )
            )
        objects.sort(key=lambda item: item.object_id)
        evidence = SensorEvidence(
            frame=bundle.frame,
            timestamp=bundle.timestamp,
            observed_fraction=None,
            instance_ids=tuple(item.object_id for item in objects),
            instance_masks=bundle.left_masks,
            camera_intrinsics=K,
            camera_extrinsics=world_from_left,
            rgb_left=left,
            rgb_right=right,
        )
        return PerceptualState(tuple(objects), bundle.frame, bundle.timestamp), evidence


class DepthBundleExtractor:
    """Ricostruisce geometria dalla depth renderizzata e dalle maschere."""

    def extract(self, bundle: SensorBundle) -> tuple[PerceptualState, SensorEvidence]:
        from physical_ai_mujoco.sensors.cloud import estimate_geometry, mask_to_point_cloud
        from physical_ai_mujoco.sensors.rig import StereoRig

        if bundle.depth_left is None or bundle.depth_right is None:
            raise ValueError("DepthBundleExtractor richiede depth in entrambe le viste")
        left = np.asarray(bundle.rgb_left)
        right = np.asarray(bundle.rgb_right)
        if left.shape != right.shape or left.ndim != 3 or left.shape[2] != 3:
            raise ValueError("Immagini stereo non valide")
        if bundle.depth_left.shape != left.shape[:2] or bundle.depth_right.shape != right.shape[:2]:
            raise ValueError("La depth non e' allineata alle immagini")
        rig = StereoRig(bundle.intrinsics, bundle.world_from_left, bundle.baseline,
                        left.shape[0], left.shape[1])
        objects = []
        clouds = []
        for track_id in sorted(bundle.left_masks.keys() | bundle.right_masks.keys()):
            parts = []
            for eye, masks, depth in (("left", bundle.left_masks, bundle.depth_left),
                                      ("right", bundle.right_masks, bundle.depth_right)):
                mask = masks.get(track_id)
                if mask is None:
                    continue
                points = mask_to_point_cloud(depth, rig.intrinsics, rig.world_from_eye(eye), mask)
                if len(points):
                    parts.append(points)
            if not parts:
                continue
            points = np.concatenate(parts, axis=0)
            if len(points) < 6:
                continue
            geometry = estimate_geometry(points)
            clouds.append(points)
            objects.append(PerceptualObject(
                track_id,
                None if bundle.type_ids is None else bundle.type_ids.get(track_id),
                geometry.position, geometry.quaternion, geometry.shape,
                geometry.size, True, geometry.confidence,
            ))
        evidence = SensorEvidence(
            frame=bundle.frame, timestamp=bundle.timestamp,
            observed_fraction=None,
            point_cloud=np.concatenate(clouds) if clouds else np.empty((0, 3)),
            instance_ids=tuple(obj.object_id for obj in objects),
            depth_map=bundle.depth_left, instance_masks=bundle.left_masks,
            camera_intrinsics=rig.intrinsics, camera_extrinsics=rig.world_from_left,
            rgb_left=left, rgb_right=right,
        )
        return PerceptualState(tuple(objects), bundle.frame, bundle.timestamp), evidence


class SceneUnderstanding:
    def build(
        self, perceptual: PerceptualState, evidence: SensorEvidence, task: TaskContext
    ) -> SceneState:
        if perceptual.frame != evidence.frame or perceptual.timestamp != evidence.timestamp:
            raise ValueError("PerceptualState e SensorEvidence non sono sincronizzati")
        target_id = task.target_id
        if target_id is None and task.target_type_id is not None:
            candidates = [
                item for item in perceptual.objects
                if item.type_id == task.target_type_id
            ]
            if candidates:
                target_id = max(
                    candidates,
                    key=lambda item: (
                        item.classification_confidence
                        if item.classification_confidence is not None
                        else (-1.0 if item.confidence is None else item.confidence)
                    ),
                ).object_id
        objects = tuple(
            SceneObject(
                object_id=obj.object_id,
                role="target" if obj.object_id == target_id else "obstacle",
                type_id=obj.type_id,
                position=obj.position,
                quaternion=obj.quaternion,
                shape=obj.shape,
                size=obj.size,
                detected=obj.detected,
                perception_quality=obj.confidence,
            )
            for obj in perceptual.objects
        )
        return SceneState(
            objects,
            target_id,
            task.ground_height,
            task.bounds,
            perceptual.frame,
            perceptual.timestamp,
            task.protected_zone,
            task.target_safe_zone,
            task.obstacle_drop_zone,
        )


class GeometricRelationEstimator:
    """Baseline di candidati di supporto; score geometrico non calibrato.

    Non implementa il predittore di dinamica locale appreso di Li et al.
    """

    def __init__(self, tolerance=0.03):
        self.tolerance = float(tolerance)

    def estimate(self, scene: SceneState, evidence: SensorEvidence) -> PhysicalRelationState:
        relations = []
        for lower in scene.objects:
            if lower.position is None or lower.size is None:
                continue
            for upper in scene.objects:
                if lower.object_id == upper.object_id or upper.position is None or upper.size is None:
                    continue
                vertical_gap = (
                    upper.position[2] - upper.size[2] / 2
                    - lower.position[2] - lower.size[2] / 2
                )
                if not -self.tolerance <= vertical_gap <= self.tolerance:
                    continue
                xy_distance = math.dist(lower.position[:2], upper.position[:2])
                xy_limit = math.hypot(*lower.size[:2]) / 2 + math.hypot(*upper.size[:2]) / 2
                if xy_distance > xy_limit:
                    continue
                score = max(0.0, 1.0 - abs(vertical_gap) / self.tolerance) if self.tolerance else 1.0
                relations.append(
                    PhysicalRelation(lower.object_id, upper.object_id, "candidate_support", score)
                )
        return PhysicalRelationState(
            tuple(obj.object_id for obj in scene.objects),
            tuple(relations),
            "geometric_candidate_v1",
            bool(scene.objects) and all(
                obj.position is not None and obj.size is not None for obj in scene.objects
            ),
            scene.frame,
            scene.timestamp,
        )


class ExactUncertaintyProvider:
    def update(
        self,
        scene: SceneState,
        evidence: SensorEvidence,
        previous_uncertainty: UncertaintyState | None = None,
        physical_relations: PhysicalRelationState | None = None,
    ) -> UncertaintyState:
        return UncertaintyState(
            tuple(ObjectUncertainty(obj.object_id, True, 1.0, 1.0) for obj in scene.objects),
            False,
            "exact",
            scene.frame,
            scene.timestamp,
        )


class DegradedUncertaintyProvider:
    def update(
        self,
        scene: SceneState,
        evidence: SensorEvidence,
        previous_uncertainty: UncertaintyState | None = None,
        physical_relations: PhysicalRelationState | None = None,
    ) -> UncertaintyState:
        current = tuple(
                ObjectUncertainty(
                    obj.object_id,
                    obj.detected,
                    obj.perception_quality,
                    obj.perception_quality if obj.position is not None else None,
                )
                for obj in scene.objects
            )
        current_ids = {item.object_id for item in current}
        remembered = () if previous_uncertainty is None else tuple(
            ObjectUncertainty(item.object_id, False, None, None)
            for item in previous_uncertainty.objects
            if item.object_id not in current_ids
        )
        return UncertaintyState(
            current + remembered,
            evidence.observed_fraction is None or evidence.observed_fraction < 1.0 or bool(remembered),
            "degraded",
            scene.frame,
            scene.timestamp,
        )


class StereoUncertaintyProvider(DegradedUncertaintyProvider):
    """Stato iniziale non calibrato: non inventa confidence numeriche."""

    def update(
        self,
        scene: SceneState,
        evidence: SensorEvidence,
        previous_uncertainty: UncertaintyState | None = None,
        physical_relations: PhysicalRelationState | None = None,
    ) -> UncertaintyState:
        base = super().update(scene, evidence, previous_uncertainty, physical_relations)
        return replace(base, unknown_space=True, provider="stereo_uncalibrated")


class ObservationBuilder:
    """Compone e valida prodotti deployable; non seleziona feature PPO."""

    def build(
        self,
        scene: SceneState,
        relations: PhysicalRelationState,
        uncertainty: UncertaintyState,
    ) -> Observation:
        observation = Observation(scene, relations, uncertainty)
        validate_observation(observation)
        return observation
