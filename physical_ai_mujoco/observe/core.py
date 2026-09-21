"""API pubblica di OSSERVA e composizione dei blocchi interni."""

from abc import ABC, abstractmethod

from physical_ai_mujoco.contracts import (
    Observation,
    ObjectObservation,
    PerceptualObject,
    PerceptualState,
    PrivilegedState,
    SensorEvidence,
    SynchronizedSensorPacket,
    TaskContext,
)

from .pipeline import (
    DegradedStateExtractor,
    DegradedUncertaintyProvider,
    DepthBundleExtractor,
    ExactStateExtractor,
    ExactUncertaintyProvider,
    GeometricRelationEstimator,
    ObservationBuilder,
    SceneUnderstanding,
    StereoBundleExtractor,
    StereoUncertaintyProvider,
)


class Observer(ABC):
    def reset(self, seed=None):
        """Azzera la memoria temporale a inizio episodio."""

    @abstractmethod
    def observe(self, source, context: TaskContext) -> Observation:
        raise NotImplementedError


class PipelineObserver(Observer):
    """Coordina i cinque blocchi senza esporli a DECIDE."""

    def __init__(self, extractor, uncertainty_provider):
        self.extractor = extractor
        self.scene_understanding = SceneUnderstanding()
        self.relation_estimator = GeometricRelationEstimator()
        self.uncertainty_provider = uncertainty_provider
        self.builder = ObservationBuilder()
        self.previous_uncertainty = None

    def reset(self, seed=None):
        self.previous_uncertainty = None
        if hasattr(self.extractor, "reset"):
            self.extractor.reset(seed)

    def observe(self, source, context: TaskContext) -> Observation:
        if not isinstance(context, TaskContext):
            raise TypeError("Observer.observe richiede un TaskContext")
        perceptual, evidence = self.extractor.extract(source)
        scene = self.scene_understanding.build(perceptual, evidence, context)
        relations = self.relation_estimator.estimate(scene, evidence)
        uncertainty = self.uncertainty_provider.update(
            scene, evidence, self.previous_uncertainty, relations
        )
        observation = self.builder.build(scene, relations, uncertainty)
        self.previous_uncertainty = uncertainty
        return observation


class ExactObserver(PipelineObserver):
    def __init__(self):
        super().__init__(ExactStateExtractor(), ExactUncertaintyProvider())

    def privileged_state(self, simulator, target_id: str) -> PrivilegedState:
        """Ramo separato simulation-only per teacher e valutazione."""

        objects = []
        for item in simulator.scene.objects:
            state = simulator.get_object_state(item.instance_id)
            objects.append(
                ObjectObservation(
                    item.instance_id,
                    tuple(state.position),
                    tuple(state.quaternion),
                    tuple(state.linear_velocity),
                    tuple(state.angular_velocity),
                    item.mass,
                    item.friction[0],
                    simulator.is_present(item.instance_id),
                    item.instance_id == target_id,
                )
            )
        return PrivilegedState(tuple(objects))


class DegradedObserver(PipelineObserver):
    def __init__(self, position_sigma=0.01, drop_probability=0.1):
        super().__init__(
            DegradedStateExtractor(position_sigma, drop_probability),
            DegradedUncertaintyProvider(),
        )


class StereoObserver(PipelineObserver):
    """Baseline legacy stereo/depth, mantenuta solo per benchmark comparativi."""

    def __init__(self, reconstruction_mode: str = "stereo"):
        if reconstruction_mode not in ("stereo", "depth"):
            raise ValueError("reconstruction_mode deve essere 'stereo' o 'depth'")
        extractor = StereoBundleExtractor() if reconstruction_mode == "stereo" else DepthBundleExtractor()
        super().__init__(extractor, StereoUncertaintyProvider())


class SensorObserver(Observer):
    """Pipeline deployable: segmentazione 2D + localizzazione LiDAR 3D."""

    def __init__(self, detector, *, preprocessor=None, projector=None, geometry_estimator=None):
        from .geometry import LidarGeometryEstimator
        from .localization import LidarProjector
        from .preprocessing import LidarPreprocessor

        self.detector = detector
        self.preprocessor = preprocessor or LidarPreprocessor()
        self.projector = projector or LidarProjector()
        self.geometry_estimator = geometry_estimator or LidarGeometryEstimator()
        self.scene_understanding = SceneUnderstanding()
        self.relation_estimator = GeometricRelationEstimator()
        self.uncertainty_provider = StereoUncertaintyProvider()
        self.builder = ObservationBuilder()
        self.previous_uncertainty = None

    def reset(self, seed=None):
        self.previous_uncertainty = None
        if hasattr(self.detector, "reset"):
            self.detector.reset(seed)

    def observe(
        self,
        packet: SynchronizedSensorPacket,
        context: TaskContext,
    ) -> Observation:
        if not isinstance(packet, SynchronizedSensorPacket):
            raise TypeError("SensorObserver richiede un SynchronizedSensorPacket")
        if not isinstance(context, TaskContext):
            raise TypeError("SensorObserver richiede un TaskContext")
        detections = self.detector.detect(packet.stereo)
        if not detections.source:
            raise ValueError("La sorgente delle detection deve essere dichiarata")
        points = self.preprocessor.process(packet.lidar, context)
        localized = self.projector.project(points, detections.detections, packet.calibration)

        objects = []
        instance_points = []
        point_ids = []
        for item in localized:
            geometry = self.geometry_estimator.estimate(item, context.target_type_id)
            if geometry is None:
                continue
            detection = item.detection
            class_confidence = detection.class_confidence
            confidence = geometry.confidence if class_confidence is None else min(
                class_confidence, geometry.confidence
            )
            objects.append(
                PerceptualObject(
                    object_id=detection.track_id,
                    type_id=detection.class_id,
                    position=geometry.position,
                    quaternion=geometry.quaternion,
                    shape=geometry.shape,
                    size=geometry.size,
                    detected=True,
                    confidence=confidence,
                    classification_confidence=class_confidence,
                )
            )
            instance_points.append(item.points)
            point_ids.extend([detection.track_id] * len(item.points))

        objects.sort(key=lambda item: item.object_id)
        import numpy as np

        evidence = SensorEvidence(
            frame=packet.frame,
            timestamp=packet.timestamp,
            observed_fraction=None,
            point_cloud=(
                np.concatenate(instance_points, axis=0)
                if instance_points else np.empty((0, 3), dtype=float)
            ),
            instance_ids=tuple(item.object_id for item in objects),
            point_instance_ids=np.asarray(point_ids, dtype=object),
            instance_masks=detections.left_masks,
            camera_intrinsics=packet.calibration.intrinsics,
            camera_extrinsics=packet.calibration.world_from_left,
            lidar=packet.lidar.valid_points,
            rgb_left=packet.stereo.rgb_left,
            rgb_right=packet.stereo.rgb_right,
        )
        perceptual = PerceptualState(tuple(objects), packet.frame, packet.timestamp)
        scene = self.scene_understanding.build(perceptual, evidence, context)
        relations = self.relation_estimator.estimate(scene, evidence)
        uncertainty = self.uncertainty_provider.update(
            scene, evidence, self.previous_uncertainty, relations
        )
        result = self.builder.build(scene, relations, uncertainty)
        self.previous_uncertainty = uncertainty
        return result
