"""Sorgenti sensoriali che producono contratti consumabili da OSSERVA."""

from .capture import STEREO_BASELINE_M, SimulatedStereoCamera, StereoFrame
from .rig import StereoRig
from .detector import Detector, OracleDetector, StereoDetections
from .bundle import BundleBuilder
from .noise import DetectionNoise, CalibrationNoise, DepthNoise
from .source import SensorSource, SimulatedSensorSource
from .learned_detector import (
    LearnedDetector,
    SegmentationPrediction,
    resolve_detector_weights,
)
from .disturbance import ImageDisturbance, apply_image_disturbance
from .tracking import ObjectTracker, ObjectTrackerConfig

__all__ = [
    "STEREO_BASELINE_M", "SimulatedStereoCamera", "StereoFrame", "StereoRig",
    "Detector", "OracleDetector",
    "StereoDetections", "BundleBuilder", "DetectionNoise", "CalibrationNoise", "DepthNoise",
    "SensorSource", "SimulatedSensorSource",
    "LearnedDetector", "SegmentationPrediction", "resolve_detector_weights",
    "ImageDisturbance", "apply_image_disturbance",
    "ObjectTracker", "ObjectTrackerConfig",
]
