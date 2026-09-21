"""Compone frame, rilevazioni e disturbi senza conoscere MuJoCo."""

import numpy as np
import warnings

from physical_ai_mujoco.contracts import SensorBundle

from .calibration import perturb_rig
from .capture import StereoFrame
from .detector import Detector
from .noise import (CalibrationNoise, DepthNoise, DetectionNoise,
                    apply_depth_noise, apply_detection_noise, apply_type_confusion)
from .rig import StereoRig


class BundleBuilder:
    def __init__(self, detector: Detector, detection_noise: DetectionNoise | None = None,
                 calibration_noise: CalibrationNoise | None = None,
                 depth_noise: DepthNoise | None = None, seed: int | None = None):
        warnings.warn(
            "BundleBuilder/SensorBundle e' una baseline legacy riservata ai "
            "confronti stereo-only e RGB-D; il percorso deployable usa "
            "SensorSource + SynchronizedSensorPacket.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.detector = detector
        self.detection_noise = detection_noise or DetectionNoise()
        self.calibration_noise = calibration_noise or CalibrationNoise()
        self.depth_noise = depth_noise or DepthNoise()
        self.reset(seed)

    def reset(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def build(self, frame: StereoFrame) -> SensorBundle:
        detected = self.detector.detect(frame)
        if not detected.source:
            raise ValueError("La sorgente delle detection deve essere dichiarata")
        expected = frame.gray_left.shape
        for masks in (detected.left_masks, detected.right_masks):
            if any(np.asarray(mask).shape != expected for mask in masks.values()):
                raise ValueError("Maschera del detector non allineata al frame")
        rig = StereoRig(frame.intrinsics, frame.world_from_left, frame.baseline,
                        frame.gray_left.shape[0], frame.gray_left.shape[1])
        perceived = perturb_rig(rig, self.calibration_noise, self.rng)
        left = apply_detection_noise(detected.left_masks, self.detection_noise, self.rng)
        right = apply_detection_noise(detected.right_masks, self.detection_noise, self.rng)
        depth_left = (apply_depth_noise(frame.depth_left, self.depth_noise, self.rng)
                      if frame.depth_left is not None else None)
        depth_right = (apply_depth_noise(frame.depth_right, self.depth_noise, self.rng)
                       if frame.depth_right is not None else None)
        return SensorBundle(
            frame.rgb_left, frame.rgb_right, perceived.intrinsics,
            perceived.world_from_left, perceived.baseline, frame.timestamp,
            frame.frame, left, right,
            apply_type_confusion(detected.type_ids, self.detection_noise, self.rng),
            depth_left, depth_right, detected.source,
        )
