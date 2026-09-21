"""Confronta le sorgenti di OSSERVA sulle stesse scene, senza DECIDE/ESEGUE."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import secrets
import statistics
import subprocess
import sys
import time

import numpy as np

from physical_ai_mujoco.contracts import (
    Observation, ObservationInvariantError, TaskContext, validate_observation,
)
from physical_ai_mujoco.observe import (
    CADMatcher,
    DegradedObserver,
    ExactObserver,
    LidarGeometryEstimator,
    SensorObserver,
    StereoObserver,
)
from physical_ai_mujoco.sensors import (
    BundleBuilder, CalibrationNoise, DepthNoise, DetectionNoise, LearnedDetector,
    ImageDisturbance, OracleDetector,
    STEREO_BASELINE_M, SimulatedSensorSource, SimulatedStereoCamera,
    resolve_detector_weights,
)
from physical_ai_mujoco.sensors.lidar import LidarConfig, LidarNoise
from physical_ai_mujoco.evaluation.target_exposure import (
    measure_target_exposure,
    place_target_at_immersion,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs/observe_tests"
DETECTOR_WEIGHTS_DIR = PROJECT_ROOT / "outputs/detector_weights"
DEFAULT_DETECTOR_WEIGHTS = DETECTOR_WEIGHTS_DIR / "pfm_1_seg.pt"
MODES = ("exact", "degraded", "stereo")
ALL_MODES = MODES + ("rgbd", "fusion_oracle", "fusion_learned")
DEFAULT_MODES = tuple(mode for mode in ALL_MODES if mode != "fusion_learned")
STEREO_PROFILES = ("clean", "fixed", "random")
INTERACTIVE_PROFILE_KEYS = frozenset(
    {"name", "scene_count", "object_count", "seed", "degraded", "stereo"}
)


def _object_count_for_scene(spec: int | list[int], input_seed: int) -> int:
    """Numero fisso o estrazione riproducibile nell'intervallo inclusivo."""
    if isinstance(spec, int) and not isinstance(spec, bool) and spec >= 1:
        return spec
    if (
        isinstance(spec, list) and len(spec) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in spec)
        and 1 <= spec[0] <= spec[1]
    ):
        return int(np.random.default_rng(input_seed + 40_000_011).integers(spec[0], spec[1] + 1))
    raise ValueError("object_count deve essere un intero positivo o [min, max] positivi")


def _parameter(value, rng: np.random.Generator, name: str, *, maximum=None) -> float:
    if isinstance(value, list) and len(value) == 2:
        low, high = (float(item) for item in value)
        if not np.isfinite(low) or not np.isfinite(high) or low < 0 or low > high or (
            maximum is not None and high > maximum
        ):
            raise ValueError(f"Intervallo non valido per {name}")
        result = float(rng.uniform(low, high))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
    else:
        raise ValueError(f"Valore non valido per {name}: {value!r}")
    if not np.isfinite(result) or result < 0 or (maximum is not None and result > maximum):
        raise ValueError(f"Valore fuori range per {name}: {result}")
    return result


def _sample_parameters(config: dict, scene_seed: int) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(scene_seed + 10_000_019)
    degraded = config["degraded"]
    stereo = config["stereo"]
    fusion = config.get("fusion", {})
    return {
        "degraded": {
            "position_sigma": _parameter(degraded["position_sigma"], rng, "position_sigma"),
            "drop_probability": _parameter(degraded["drop_probability"], rng, "drop_probability", maximum=1),
        },
        "stereo": {
            "centroid_noise_px": _parameter(stereo["centroid_noise_px"], rng, "centroid_noise_px"),
            "drop_probability": _parameter(stereo["drop_probability"], rng, "stereo.drop_probability", maximum=1),
        },
        "rgbd": {
            "drop_probability": _parameter(config.get("rgbd", {}).get("drop_probability", 0.0), rng, "rgbd.drop_probability", maximum=1),
            "depth_sigma": _parameter(config.get("rgbd", {}).get("depth_sigma", 0.0), rng, "rgbd.depth_sigma"),
            "translation_sigma": _parameter(config.get("rgbd", {}).get("translation_sigma", 0.0), rng, "rgbd.translation_sigma"),
            "rotation_sigma_deg": _parameter(config.get("rgbd", {}).get("rotation_sigma_deg", 0.0), rng, "rgbd.rotation_sigma_deg"),
        },
        "fusion": {
            "contrast_delta": _parameter(fusion.get("contrast_delta", 0.0), rng, "fusion.contrast_delta", maximum=0.95),
            "brightness_abs": _parameter(fusion.get("brightness_abs", 0.0), rng, "fusion.brightness_abs"),
            "gamma_delta": _parameter(fusion.get("gamma_delta", 0.0), rng, "fusion.gamma_delta", maximum=0.95),
            "image_noise_sigma": _parameter(fusion.get("image_noise_sigma", 0.0), rng, "fusion.image_noise_sigma"),
            "blur_probability": _parameter(fusion.get("blur_probability", 0.0), rng, "fusion.blur_probability", maximum=1),
            "lidar_distance_sigma": _parameter(fusion.get("lidar_distance_sigma", 0.0), rng, "fusion.lidar_distance_sigma"),
            "lidar_angle_sigma_deg": _parameter(fusion.get("lidar_angle_sigma_deg", 0.0), rng, "fusion.lidar_angle_sigma_deg"),
            "lidar_drop_probability": _parameter(fusion.get("lidar_drop_probability", 0.0), rng, "fusion.lidar_drop_probability", maximum=1),
        },
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


class _RecordingDetector:
    """Registra l'ultima detection senza cambiare il contratto operativo."""

    def __init__(self, detector):
        self.detector = detector
        self.last = None

    def detect(self, frame):
        self.last = self.detector.detect(frame)
        return self.last


def _target_visual_metrics(
    detections, simulator, target_id: str, target_type_id: str
) -> dict:
    truth = simulator.render_instance_masks("cam_left").get(target_id)
    if truth is None:
        truth = np.zeros(
            (
                simulator.scene.stereo_camera.height,
                simulator.scene.stereo_camera.width,
            ),
            dtype=bool,
        )
    truth = np.asarray(truth, dtype=bool)
    candidates = [
        item for item in detections.detections
        if item.class_id == target_type_id
    ]
    overlaps = []
    for item in candidates:
        mask = np.asarray(item.left_mask, dtype=bool)
        union = int(np.logical_or(mask, truth).sum())
        overlaps.append(
            float(np.logical_and(mask, truth).sum() / union) if union else 0.0
        )
    best_iou = max(overlaps, default=0.0)
    observable = bool(truth.any())
    recognized = observable and best_iou >= 0.25
    return {
        "target_observable": observable,
        "target_visual_recognized": recognized,
        "target_visual_recall_observable": (
            float(recognized) if observable else None
        ),
        "target_best_mask_iou": best_iou,
        "target_prediction_count": len(candidates),
    }


def _true_object_size(description) -> np.ndarray:
    if description.shape in {"box", "mesh"}:
        values = tuple(description.size[key] for key in ("x", "y", "z"))
    elif description.shape == "cylinder":
        values = (
            2 * description.size["radius"],
            2 * description.size["radius"],
            description.size["height"],
        )
    else:
        values = (2 * description.size["radius"],) * 3
    return np.asarray(values, dtype=float)


def _sorted_size_ratios(estimated, truth) -> tuple[float, float, float]:
    estimated_sorted = np.sort(np.asarray(estimated, dtype=float))
    truth_sorted = np.sort(np.asarray(truth, dtype=float))
    ratios = estimated_sorted / np.maximum(truth_sorted, 1e-12)
    return tuple(float(value) for value in ratios)


def measure_observation(
    observation: Observation,
    simulator,
    target_type_id: str = "pfm_1_target",
) -> dict:
    """Misura un output gia' validato; MuJoCo e' solo il riferimento."""
    truth_ids = set(simulator.present_objects())
    observed = {item.object_id: item for item in observation.objects}
    detected_ids = set(observed)
    # Gli adapter oracle conservano gli ID MuJoCo; un detector reale produce
    # track ID propri. Il benchmark associa questi ultimi alla verita' soltanto
    # per calcolare le metriche, senza restituire la mappa alla pipeline.
    observed_to_truth = {
        object_id: object_id for object_id in truth_ids & detected_ids
    }
    unmatched_observed = [
        item for item in observation.objects
        if item.object_id not in observed_to_truth and item.position is not None
    ]
    unmatched_truth = truth_ids - set(observed_to_truth.values())
    candidates = []
    for item in unmatched_observed:
        for truth_id in unmatched_truth:
            distance = float(np.linalg.norm(
                np.asarray(item.position)
                - np.asarray(simulator.get_object_state(truth_id).position)
            ))
            if distance <= 0.15:
                candidates.append((distance, item.object_id, truth_id))
    used_observed = set(observed_to_truth)
    used_truth = set(observed_to_truth.values())
    for _, observed_id, truth_id in sorted(candidates):
        if observed_id in used_observed or truth_id in used_truth:
            continue
        observed_to_truth[observed_id] = truth_id
        used_observed.add(observed_id)
        used_truth.add(truth_id)
    matched = set(observed_to_truth.values())
    errors = [
        float(np.linalg.norm(
            np.asarray(observed[observed_id].position)
            - np.asarray(simulator.get_object_state(truth_id).position)
        ))
        for observed_id, truth_id in sorted(observed_to_truth.items())
        if observed[observed_id].position is not None
    ]
    descriptions = {item.instance_id: item for item in simulator.scene.objects}
    size_errors = []
    size_ratio_means = []
    volume_size_ratios = []
    shape_correct = []
    for observed_id, truth_id in sorted(observed_to_truth.items()):
        estimate = observed[observed_id]
        truth = descriptions[truth_id]
        if estimate.shape is not None:
            shape_correct.append(estimate.shape == truth.shape)
        if estimate.size is not None:
            true_size = _true_object_size(truth)
            estimated_size = np.asarray(estimate.size, dtype=float)
            size_errors.append(float(np.linalg.norm(np.sort(estimated_size)-np.sort(true_size))))
            ratios = _sorted_size_ratios(estimated_size, true_size)
            size_ratio_means.append(float(np.mean(ratios)))
            volume_size_ratios.append(
                float(np.prod(estimated_size) / max(np.prod(true_size), 1e-12))
            )

    # support_graph(): upper -> lower. Observation: lower -> upper.
    true_supports = {
        (lower, upper)
        for upper, lowers in simulator.support_graph().items()
        if upper in truth_ids
        for lower in lowers
        if lower != "terreno" and lower in truth_ids
    }
    observable_supports = {
        edge for edge in true_supports if edge[0] in matched and edge[1] in matched
    }
    predicted_supports = {
        (observed_to_truth[relation.source_id], observed_to_truth[relation.target_id])
        for relation in observation.relations.relations
        if relation.relation_type == "candidate_support"
        and relation.source_id in observed_to_truth
        and relation.target_id in observed_to_truth
    }
    true_positives = len(predicted_supports & true_supports)
    support_precision = (
        _ratio(true_positives, len(predicted_supports))
        if predicted_supports else (0.0 if true_supports else None)
    )
    support_recall = _ratio(true_positives, len(true_supports))
    conditional_recall = _ratio(true_positives, len(observable_supports))
    support_f1 = None
    if support_precision is not None and support_recall is not None:
        total = support_precision + support_recall
        support_f1 = 2 * support_precision * support_recall / total if total else 0.0
    if not observation.relations.available:
        support_precision = support_recall = conditional_recall = support_f1 = None

    target_truth_ids = {
        item.instance_id for item in simulator.scene.objects
        if item.type_id == target_type_id and item.instance_id in truth_ids
    }
    predicted_target = observation.scene.target_id
    matched_target = (
        None if predicted_target is None else observed_to_truth.get(predicted_target)
    )
    # Il matching globale e' volutamente uno-a-uno per la precisione oggetti.
    # Un detector puo' pero' produrre due maschere della stessa mina: se il
    # candidato scelto e' il duplicato non assegnato, non va contato come
    # target sbagliato. Valutiamo quindi il target scelto direttamente contro
    # la posa vera, mantenendo i duplicati penalizzati nelle metriche oggetto.
    if (
        matched_target not in target_truth_ids
        and predicted_target in observed
        and observed[predicted_target].position is not None
    ):
        target_candidates = [
            (
                float(
                    np.linalg.norm(
                        np.asarray(observed[predicted_target].position)
                        - np.asarray(simulator.get_object_state(truth_id).position)
                    )
                ),
                truth_id,
            )
            for truth_id in target_truth_ids
        ]
        if target_candidates:
            distance, truth_id = min(target_candidates)
            if distance <= 0.15:
                matched_target = truth_id
    target_detected = matched_target in target_truth_ids
    target_size_ratios = None
    target_volume_size_ratio = None
    if (
        target_detected
        and predicted_target in observed
        and observed[predicted_target].size is not None
    ):
        target_truth_size = _true_object_size(descriptions[matched_target])
        target_estimated_size = np.asarray(observed[predicted_target].size, dtype=float)
        target_size_ratios = _sorted_size_ratios(
            target_estimated_size, target_truth_size
        )
        target_volume_size_ratio = float(
            np.prod(target_estimated_size) / max(np.prod(target_truth_size), 1e-12)
        )
    return {
        "true_objects": len(truth_ids),
        "detected_objects": len(detected_ids),
        "matched_objects": len(matched),
        "true_object_ids": sorted(truth_ids),
        "detected_object_ids": sorted(detected_ids),
        "target_detected": target_detected,
        "target_recall": float(target_detected),
        "target_prediction_precision": (
            None if predicted_target is None else float(target_detected)
        ),
        "predicted_target_id": predicted_target,
        "matched_target_truth_id": matched_target,
        "observation_to_truth": observed_to_truth,
        "object_recall": _ratio(len(matched), len(truth_ids)),
        "object_precision": _ratio(len(matched), len(detected_ids)),
        "mean_position_error_m": statistics.mean(errors) if errors else None,
        "mean_size_error_m": statistics.mean(size_errors) if size_errors else None,
        "mean_size_ratio": statistics.mean(size_ratio_means) if size_ratio_means else None,
        "mean_volume_size_ratio": statistics.mean(volume_size_ratios) if volume_size_ratios else None,
        "target_size_ratio_small": None if target_size_ratios is None else target_size_ratios[0],
        "target_size_ratio_medium": None if target_size_ratios is None else target_size_ratios[1],
        "target_size_ratio_large": None if target_size_ratios is None else target_size_ratios[2],
        "target_size_ratio_mean": None if target_size_ratios is None else statistics.mean(target_size_ratios),
        "target_volume_size_ratio": target_volume_size_ratio,
        "shape_accuracy": statistics.mean(shape_correct) if shape_correct else None,
        "position_errors_m": errors,
        "relations_available": observation.relations.available,
        "true_supports": len(true_supports),
        "true_support_edges": [list(edge) for edge in sorted(true_supports)],
        "observable_supports": len(observable_supports),
        "candidate_supports": len(predicted_supports),
        "candidate_support_edges": [list(edge) for edge in sorted(predicted_supports)],
        "support_precision": support_precision,
        "support_recall": support_recall,
        "support_recall_when_both_seen": conditional_recall,
        "support_f1": support_f1,
        "unknown_space": observation.uncertainty.unknown_space,
        "unseen_object_ids": list(observation.uncertainty.unseen_object_ids),
        "object_uncertainty": [
            {
                "object_id": item.object_id,
                "visible": item.visible,
                "remembered": not item.visible,
                "detection_quality": item.detection_quality,
                "pose_quality": item.pose_quality,
            }
            for item in observation.uncertainty.objects
        ],
        "invariants_ok": True,
        "invariant_violations": [],
    }


def _invalid_metrics(error: ObservationInvariantError) -> dict:
    """Conserva il guasto senza trasformarlo in una falsa misura di accuratezza."""
    return {"invariants_ok": False, "invariant_violations": [str(error)]}


def _summary(records: list[dict]) -> dict:
    summary = {}
    for mode in ALL_MODES:
        items = [item for item in records if item["mode"] == mode]
        if not items:
            continue
        valid = [item for item in items if item["metrics"]["invariants_ok"]]
        metrics = {}
        for key in (
            "object_recall", "object_precision", "mean_position_error_m",
            "mean_size_error_m", "shape_accuracy",
            "mean_size_ratio", "mean_volume_size_ratio",
            "target_size_ratio_small", "target_size_ratio_medium",
            "target_size_ratio_large", "target_size_ratio_mean",
            "target_volume_size_ratio",
            "target_recall", "target_prediction_precision",
            "target_visual_recall_observable", "target_best_mask_iou",
            "support_precision", "support_recall", "support_recall_when_both_seen",
            "support_f1",
        ):
            values = [
                item["metrics"].get(key)
                for item in valid
                if item["metrics"].get(key) is not None
            ]
            metrics[key] = {
                "mean": statistics.mean(values) if values else None,
                "std": statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None),
                "scene_count": len(values),
            }
        worst = sorted(
            (item for item in valid if item["metrics"]["mean_position_error_m"] is not None),
            key=lambda item: item["metrics"]["mean_position_error_m"],
            reverse=True,
        )[:3]
        worst_supports = sorted(
            (item for item in valid if item["metrics"]["support_f1"] is not None),
            key=lambda item: item["metrics"]["support_f1"],
        )[:3]
        summary[mode] = {
            "metrics": metrics,
            "worst_position_scene_seeds": [item["scene_seed"] for item in worst],
            "worst_support_scene_seeds": [item["scene_seed"] for item in worst_supports],
            "relation_unavailable_scenes": sum(not item["metrics"]["relations_available"] for item in valid),
            "unknown_space_scenes": sum(item["metrics"]["unknown_space"] for item in valid),
            "invariant_violations": sum(not item["metrics"]["invariants_ok"] for item in items),
            "target_exposure_curve": _target_exposure_curve(valid),
        }
    return summary


def _target_exposure_curve(records: list[dict]) -> list[dict]:
    """Riconoscimento visivo per intervalli di sagoma realmente visibile."""

    def recognized(item):
        return bool(
            item["metrics"].get(
                "target_visual_recognized",
                item["metrics"].get("target_detected", False),
            )
        )

    def mean_metric(items, key):
        values = [
            item["metrics"].get(key)
            for item in items
            if item["metrics"].get(key) is not None
        ]
        return statistics.mean(values) if values else None

    available = [
        item for item in records
        if item["metrics"].get("target_camera_visible_fraction") is not None
    ]
    hidden = [
        item for item in available
        if item["metrics"]["target_camera_visible_fraction"] <= 1e-9
    ]
    result = [
        {
            "visible_fraction_min": 0.0,
            "visible_fraction_max": 0.0,
            "fully_hidden": True,
            "scene_count": len(hidden),
            "target_detections": sum(recognized(item) for item in hidden),
            "target_recall": (
                sum(recognized(item) for item in hidden) / len(hidden)
                if hidden else None
            ),
            "target_size_ratio_mean": mean_metric(hidden, "target_size_ratio_mean"),
            "target_volume_size_ratio": mean_metric(hidden, "target_volume_size_ratio"),
        }
    ]
    for lower in np.arange(0.0, 1.0, 0.1):
        upper = float(lower + 0.1)
        selected = [
            item for item in available
            if lower < item["metrics"]["target_camera_visible_fraction"]
            and (
                item["metrics"]["target_camera_visible_fraction"] <= upper
                or (upper >= 1.0 and item["metrics"]["target_camera_visible_fraction"] <= 1.0)
            )
        ]
        detected = sum(recognized(item) for item in selected)
        result.append(
            {
                "visible_fraction_min": round(float(lower), 1),
                "visible_fraction_max": round(min(upper, 1.0), 1),
                "fully_hidden": False,
                "scene_count": len(selected),
                "target_detections": detected,
                "target_recall": detected / len(selected) if selected else None,
                "target_size_ratio_mean": mean_metric(selected, "target_size_ratio_mean"),
                "target_volume_size_ratio": mean_metric(selected, "target_volume_size_ratio"),
            }
        )
    return result


def run_benchmark(
    config: dict,
    modes: tuple[str, ...] = MODES,
    *,
    show_viewer: bool = False,
    view_seconds: float = 2.0,
    detector_weights: str | Path | None = None,
    detector_confidence_threshold: float = 0.25,
) -> dict:
    """Esegue osservatori diversi sulla medesima scena assestata per seed."""
    import gymnasium as gym
    import physical_ai_mujoco.envs  # noqa: F401 (registra l'environment)

    if not modes or any(mode not in ALL_MODES for mode in modes):
        raise ValueError(f"Modalita' non valide: {modes}")
    scene_count = int(config["scene_count"])
    object_count_spec = config["object_count"]
    seed = int(config["seed"])
    if scene_count < 1 or seed < 0:
        raise ValueError("scene_count deve essere positivo; seed >= 0")
    _object_count_for_scene(object_count_spec, seed)
    if view_seconds < 0:
        raise ValueError("view_seconds non puo' essere negativo")
    if not 0 <= detector_confidence_threshold <= 1:
        raise ValueError("detector_confidence_threshold deve essere in [0, 1]")
    immersion_range = config.get("target_immersion_range")
    if immersion_range is not None and (
        not isinstance(immersion_range, list)
        or len(immersion_range) != 2
        or not 0 <= immersion_range[0] <= immersion_range[1] <= 1
    ):
        raise ValueError("target_immersion_range deve essere [min, max] in [0, 1]")

    env = None
    active_object_count = None
    records = []
    cad_matcher_cache = {}
    try:
        for index in range(scene_count):
            object_count = _object_count_for_scene(object_count_spec, seed + index)
            if env is None or object_count != active_object_count:
                if env is not None:
                    env.close()
                env = gym.make(
                    "TargetExtraction-v0", object_count=object_count,
                    scene_rules_path=(
                        str(CONFIG_DIR / "immersed_scene_rules.json")
                        if immersion_range is not None
                        else None
                    ),
                    obs_mode="state", highlight_target=False, disable_env_checker=True,
                    render_mode="human" if show_viewer else None,
                    stereo_baseline=STEREO_BASELINE_M,
                )
                active_object_count = object_count
            _, info = env.reset(seed=seed + index)
            simulator = env.unwrapped.simulator
            scene_seed = int(info["scene_seed"])
            target_id = env.unwrapped.target_id
            target_type_id = env.unwrapped.session.scene_rules[
                "object_selection"
            ].get("target_type_id")
            if not target_type_id:
                raise ValueError("Le regole della scena non dichiarano target_type_id")
            exposure = None
            if immersion_range is not None:
                immersion_fraction = float(
                    np.random.default_rng(scene_seed + 94_000_049).uniform(
                        *immersion_range
                    )
                )
                placement = place_target_at_immersion(
                    simulator, target_id, immersion_fraction
                )
                exposure = measure_target_exposure(
                    simulator, target_id, immersion_fraction, placement
                )
            parameters = _sample_parameters(config, scene_seed)
            for mode in modes:
                recording_detector = None
                chosen = {} if mode == "exact" else (
                    parameters["fusion"]
                    if mode in ("fusion_oracle", "fusion_learned")
                    else parameters[mode]
                )
                try:
                    if mode == "exact":
                        observer = ExactObserver()
                        observation = observer.observe(
                            simulator,
                            TaskContext(
                                target_id=target_id,
                                ground_height=float(
                                    simulator.scene.ground.pose.position[2]
                                ),
                                target_type_id=target_type_id,
                            ),
                        )
                    elif mode == "degraded":
                        observer = DegradedObserver(**chosen)
                        observer.reset(scene_seed + 20_000_033)
                        observation = observer.observe(
                            simulator,
                            TaskContext(
                                target_id=target_id,
                                ground_height=float(
                                    simulator.scene.ground.pose.position[2]
                                ),
                                target_type_id=target_type_id,
                            ),
                        )
                    elif mode in ("fusion_oracle", "fusion_learned"):
                        lidar_config = LidarConfig.from_file(
                            PROJECT_ROOT / "configs/sensors/livox_avia.json"
                        )
                        packet = SimulatedSensorSource(
                            simulator,
                            lidar_config=lidar_config,
                            image_disturbance=ImageDisturbance(
                                contrast_range=(1.0-chosen["contrast_delta"], 1.0+chosen["contrast_delta"]),
                                brightness_range=(-chosen["brightness_abs"], chosen["brightness_abs"]),
                                gamma_range=(1.0-chosen["gamma_delta"], 1.0+chosen["gamma_delta"]),
                                noise_sigma_range=(0.0, chosen["image_noise_sigma"]),
                                blur_probability=chosen["blur_probability"],
                            ),
                            lidar_noise=LidarNoise(
                                distance_sigma=chosen["lidar_distance_sigma"],
                                angle_sigma_deg=chosen["lidar_angle_sigma_deg"],
                                dropout_probability=chosen["lidar_drop_probability"],
                            ),
                            seed=scene_seed + 50_000_101,
                        ).capture()
                        detector = OracleDetector(simulator)
                        if mode == "fusion_learned":
                            recording_detector = _RecordingDetector(
                                LearnedDetector(
                                    weights_path=detector_weights,
                                    confidence_threshold=detector_confidence_threshold,
                                )
                            )
                            detector = recording_detector
                        if target_type_id not in cad_matcher_cache:
                            definition = next(
                                item
                                for item in env.unwrapped.session._object_dataset[
                                    "object_types"
                                ]
                                if item["id"] == target_type_id
                            )
                            cad_matcher_cache[target_type_id] = (
                                CADMatcher.from_stl(
                                    definition["mesh_file"],
                                    tuple(definition["mesh_scale"]),
                                )
                                if definition.get("shape") == "mesh"
                                else None
                            )
                        matcher = cad_matcher_cache[target_type_id]
                        observer = SensorObserver(
                            detector,
                            geometry_estimator=LidarGeometryEstimator(
                                mesh_target_type_ids=(target_type_id,),
                                cad_matchers=(
                                    {target_type_id: matcher}
                                    if matcher is not None
                                    else {}
                                ),
                            ),
                        )
                        observation = observer.observe(
                            packet,
                            TaskContext(
                                target_id=None,
                                ground_height=float(
                                    simulator.scene.ground.pose.position[2]
                                ),
                                target_type_id=target_type_id,
                            ),
                        )
                    else:
                        with_depth = mode == "rgbd"
                        frame = SimulatedStereoCamera(with_depth=with_depth).capture(simulator)
                        noise = DetectionNoise(
                            centroid_noise_px=chosen.get("centroid_noise_px", 0.0),
                            drop_probability=chosen["drop_probability"],
                        )
                        builder = BundleBuilder(
                            OracleDetector(simulator), detection_noise=noise,
                            depth_noise=DepthNoise(distance_sigma=chosen.get("depth_sigma", 0.0)),
                            calibration_noise=CalibrationNoise(
                                translation_sigma=chosen.get("translation_sigma", 0.0),
                                rotation_sigma_deg=chosen.get("rotation_sigma_deg", 0.0),
                            ),
                            seed=scene_seed + (40_000_079 if with_depth else 30_000_059),
                        )
                        bundle = builder.build(frame)
                        observer = StereoObserver(reconstruction_mode="depth" if with_depth else "stereo")
                        observation = observer.observe(
                            bundle,
                            TaskContext(target_id=target_id,
                                        ground_height=float(simulator.scene.ground.pose.position[2])),
                        )
                    validate_observation(observation)
                except ObservationInvariantError as error:
                    metrics = _invalid_metrics(error)
                else:
                    metrics = measure_observation(
                        observation, simulator, target_type_id
                    )
                    if recording_detector is not None:
                        metrics.update(
                            _target_visual_metrics(
                                recording_detector.last,
                                simulator,
                                target_id,
                                target_type_id,
                            )
                        )
                if exposure is not None:
                    metrics.update(
                        target_immersion_fraction=exposure.immersion_fraction,
                        target_geometric_exposure=exposure.geometric_exposure_fraction,
                        target_ground_visible_fraction=exposure.ground_visible_fraction,
                        target_camera_visible_fraction=exposure.camera_visible_fraction,
                        target_obstacle_visibility_fraction=exposure.obstacle_visibility_fraction,
                        target_visible_pixels=exposure.visible_pixels,
                        target_fully_exposed_pixels=exposure.fully_exposed_pixels,
                    )
                records.append({
                    "mode": mode,
                    "scene_index": index,
                    "scene_seed": scene_seed,
                    "input_seed": seed + index,
                    "object_count": object_count,
                    "target_id": target_id,
                    "parameters": chosen,
                    "metrics": metrics,
                })
            if show_viewer:
                deadline = time.monotonic() + view_seconds
                while time.monotonic() < deadline:
                    env.render()
                    time.sleep(0.03)
    finally:
        if env is not None:
            env.close()
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "stereo_identity_source": "MuJoCo segmentation oracle; no RGB detector",
        "rgbd_depth_source": "MuJoCo rendered depth; not stereo triangulation",
        "target_identity_source": (
            "fusion modes use the configured target type; fusion_oracle uses oracle masks; "
            "legacy modes use the simulator target instance id"
        ),
        "stereo_baseline_m": STEREO_BASELINE_M,
        "config": config,
        "modes": list(modes),
        "scenes": records,
        "summary": _summary(records),
    }


def export_graphs(report: dict, directory: Path) -> list[Path]:
    """Esporta DOT e, quando Graphviz e presente, SVG dei supporti per scena."""
    def quote(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    directory.mkdir(parents=True, exist_ok=True)
    files = []
    dot_binary = shutil.which("dot")
    for record in report["scenes"]:
        metrics = record["metrics"]
        if not metrics["invariants_ok"]:
            stem = f"scene_{record['scene_index']:03d}_{record['mode']}"
            dot_path = directory / f"{stem}.dot"
            reason = "\n".join(metrics["invariant_violations"])
            dot_path.write_text(
                "digraph osservazione_invalida {\n"
                f"  invalid [shape=note, label={quote('Observation non valida\n' + reason)}];\n"
                "}\n"
            )
            files.append(dot_path)
            record["graph_dot"] = str(dot_path)
            if dot_binary:
                svg_path = directory / f"{stem}.svg"
                subprocess.run([dot_binary, "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True)
                files.append(svg_path)
                record["graph_svg"] = str(svg_path)
            continue
        true_edges = {tuple(edge) for edge in metrics["true_support_edges"]}
        candidate_edges = {tuple(edge) for edge in metrics["candidate_support_edges"]}
        detected_ids = set(metrics["detected_object_ids"])
        object_ids = sorted(
            set(metrics["true_object_ids"]) | detected_ids
        )
        title = f"Scena {record['scene_seed']} · {record['mode']} — basso → alto"
        lines = [
            "digraph supporti {",
            '  graph [rankdir=LR, fontname="Helvetica", labelloc=t];',
            '  node [shape=box, style=rounded, fontname="Helvetica"];',
            f"  label={quote(title)};",
        ]
        for side, title in (("truth", "Contatti MuJoCo"), ("observed", "OSSERVA")):
            lines.append(f"  subgraph cluster_{side} {{")
            lines.append(f"    label={quote(title)};")
            for object_id in object_ids:
                node = f"{side}:{object_id}"
                absent = side == "observed" and object_id not in detected_ids
                fill = "#eeeeee" if absent else (
                    "#fff2b3" if object_id == record["target_id"] else "#ffffff"
                )
                label = f"{object_id}\nnon visto" if absent else object_id
                style = "rounded,dashed,filled" if absent else "rounded,filled"
                lines.append(
                    f"    {quote(node)} [label={quote(label)}, "
                    f'fillcolor="{fill}", style="{style}"];'
                )
            lines.append("  }")
        for lower, upper in sorted(true_edges):
            lines.append(
                f"  {quote('truth:' + lower)} -> {quote('truth:' + upper)} "
                '[color="#444444"];'
            )
        if metrics["relations_available"]:
            for lower, upper in sorted(candidate_edges | true_edges):
                edge = (lower, upper)
                if edge in candidate_edges and edge in true_edges:
                    style = 'color="#208050", label="confermato"'
                elif edge in candidate_edges:
                    style = 'color="#dc7800", label="aggiunto"'
                else:
                    style = 'color="#c0392b", style=dashed, label="mancante"'
                lines.append(
                    f"  {quote('observed:' + lower)} -> "
                    f"{quote('observed:' + upper)} [{style}];"
                )
        else:
            lines.append(
                '  unavailable [label="Relazioni non disponibili", shape=note];'
            )
            if object_ids:
                lines.append(
                    f"  unavailable -> {quote('observed:' + object_ids[0])} "
                    "[style=invis];"
                )
        lines.append("}")
        stem = f"scene_{record['scene_index']:03d}_{record['mode']}"
        dot_path = directory / f"{stem}.dot"
        dot_path.write_text("\n".join(line for line in lines if line) + "\n")
        files.append(dot_path)
        record["graph_dot"] = str(dot_path)
        if dot_binary:
            svg_path = directory / f"{stem}.svg"
            subprocess.run([dot_binary, "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True)
            files.append(svg_path)
            record["graph_svg"] = str(svg_path)
    return files


def _ask_int(label: str, default: int, minimum: int, maximum: int | None = None) -> int:
    while True:
        answer = input(f"{label} [{default}]: ").strip()
        if not answer:
            return default
        if answer.isdigit() and int(answer) >= minimum and (
            maximum is None or int(answer) <= maximum
        ):
            return int(answer)
        interval = (
            f"tra {minimum} e {maximum}"
            if maximum is not None else f"maggiore o uguale a {minimum}"
        )
        print(f"Inserire un intero {interval}.")


def _interactive_profiles(directory: Path = CONFIG_DIR) -> list[tuple[Path, dict]]:
    """Restituisce solo i JSON che descrivono un profilo del benchmark.

    La cartella contiene anche configurazioni ausiliarie, per esempio le regole
    della scena con target interrato. Questi file non devono comparire nel menu.
    """
    profiles = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if not INTERACTIVE_PROFILE_KEYS.issubset(payload):
            continue
        if not isinstance(payload["name"], str) or not payload["name"].strip():
            continue
        profiles.append((path, payload))
    return profiles


def _resolve_detector_weights(
    requested: str | Path | None,
    directory: Path = DETECTOR_WEIGHTS_DIR,
) -> Path:
    """Compatibilita' del benchmark: usa il resolver condiviso dal runtime."""
    return resolve_detector_weights(requested, directory=directory)


def _interactive_selection() -> tuple[Path, tuple[str, ...], str | None]:
    profiles = _interactive_profiles()
    if not profiles:
        raise RuntimeError(
            f"Nessun profilo OSSERVA valido trovato in {CONFIG_DIR}"
        )
    print("\nTest autonomo di OSSERVA: scegli il profilo iniziale dei disturbi")
    for index, (path, payload) in enumerate(profiles, 1):
        print(f"  {index}. {path.stem} — {payload['name']}")
    selected, _ = profiles[_ask_int("Configurazione", 1, 1, len(profiles)) - 1]
    print(
        "  1. Tutte le sorgenti\n  2. Exact\n  3. Degraded\n"
        "  4. Stereo simulata\n  5. RGB-D simulata\n"
        "  6. Fusione camera + LiDAR (maschere oracle)\n"
        "  7. Fusione camera + LiDAR (detector appreso)"
    )
    choice = _ask_int("Sorgente", 1, 1, 7)
    choices = {1: DEFAULT_MODES, 2: ("exact",), 3: ("degraded",),
               4: ("stereo",), 5: ("rgbd",), 6: ("fusion_oracle",),
               7: ("fusion_learned",)}
    if choice not in choices:
        raise ValueError("Sorgente non valida")
    stereo_profile = None
    if "stereo" in choices[choice]:
        print("Disturbo delle rilevazioni stereo (le immagini restano grezze):")
        print("  1. Nessuno\n  2. Fisso\n  3. Casuale per scena")
        default = STEREO_PROFILES.index(selected.stem) + 1 if selected.stem in STEREO_PROFILES else 1
        stereo_profile = STEREO_PROFILES[_ask_int("Disturbo stereo", default, 1, 3) - 1]
    return selected, choices[choice], stereo_profile


def _ask_object_count(default: int) -> int | list[int]:
    while True:
        answer = input(f"Oggetti per scena: numero oppure R=casuale [{default}]: ").strip().lower()
        if not answer:
            return default
        if answer == "r":
            minimum = _ask_int("Minimo oggetti", 1, 1)
            maximum = _ask_int("Massimo oggetti", 6, minimum)
            return [minimum, maximum]
        if answer.isdigit() and int(answer) >= 1:
            return int(answer)
        print("Inserire un intero positivo oppure R.")


def _ask_visual_seed(default: int) -> int:
    """Invio = scena nuova; un intero = scena riproducibile."""
    while True:
        answer = input(
            f"Seed scena (Invio = nuovo casuale, numero = riproducibile; "
            f"profilo {default}): "
        ).strip()
        if not answer:
            selected = secrets.randbelow(2**31 - 1)
            print(f"Seed scelto: {selected} (riusalo per rivedere la stessa scena)")
            return selected
        if answer.isdigit() and 0 <= int(answer) < 2**31 - 1:
            return int(answer)
        print("Inserire un seed intero fra 0 e 2147483646, oppure Invio.")


def review_scene(
    report: dict, scene_index: int, *, seconds: float | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Riapre una scena del report con viewer MuJoCo e viste stereo affiancate."""
    import cv2
    import gymnasium as gym
    import physical_ai_mujoco.envs  # noqa: F401 (registra l'environment)

    if seconds is not None and seconds <= 0:
        raise ValueError("La durata della revisione deve essere positiva")
    records = [item for item in report["scenes"] if item["scene_index"] == scene_index]
    if not records:
        raise ValueError(f"Scena {scene_index} assente dal report")
    first = records[0]
    stereo_record = next((item for item in records if item["mode"] == "stereo"), None)
    parameters = stereo_record["parameters"] if stereo_record else {}
    env = gym.make(
        "TargetExtraction-v0", object_count=int(first.get("object_count") or report["config"]["object_count"]),
        obs_mode="state", render_mode="human", highlight_target=False,
        realtime_factor=0.0, disable_env_checker=True,
        stereo_baseline=float(report["stereo_baseline_m"]),
    )
    window = f"OSSERVA scena {scene_index + 1} - stereo (q per chiudere)"
    opened = False
    try:
        _, info = env.reset(seed=first["input_seed"])
        if info["scene_seed"] != first["scene_seed"] or env.unwrapped.target_id != first["target_id"]:
            raise RuntimeError("La scena ricostruita non coincide con il report")
        simulator = env.unwrapped.simulator
        frame = SimulatedStereoCamera(
            with_depth=any(item["mode"] == "rgbd" for item in records)
        ).capture(simulator)
        bundle = BundleBuilder(
            OracleDetector(simulator),
            detection_noise=DetectionNoise(**parameters),
            seed=first["scene_seed"] + 30_000_059,
        ).build(frame)
        panels = []
        for image, masks, label in (
            (bundle.rgb_left, bundle.left_masks, "SINISTRA"),
            (bundle.rgb_right, bundle.right_masks, "DESTRA"),
        ):
            panel = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for object_id, mask in sorted(masks.items()):
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(panel, contours, -1, (255, 255, 255), 1)
                y, x = np.rint(np.argwhere(mask).mean(axis=0)).astype(int)
                tag = object_id + (" [T]" if object_id == first["target_id"] else "")
                cv2.putText(panel, tag, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.34, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(panel, (0, 0), (panel.shape[1], 22), (30, 30, 30), -1)
            cv2.putText(panel, label, (8, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(panel)
        preview = np.concatenate(panels, axis=1)
        if bundle.depth_left is not None and bundle.depth_right is not None:
            depth_panels = []
            for depth, label in ((bundle.depth_left, "DEPTH SINISTRA"),
                                 (bundle.depth_right, "DEPTH DESTRA")):
                valid = np.isfinite(depth) & (depth > 0) & (depth < 20)
                gray = np.zeros(depth.shape, dtype=np.uint8)
                if valid.any():
                    low, high = np.percentile(depth[valid], (5, 95))
                    gray[valid] = np.uint8(255*np.clip((depth[valid]-low)/max(high-low, 1e-6), 0, 1))
                panel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(panel, (0, 0), (panel.shape[1], 22), (30, 30, 30), -1)
                cv2.putText(panel, label, (8, 16), cv2.FONT_HERSHEY_SIMPLEX,
                            .48, (255, 255, 255), 1)
                depth_panels.append(panel)
            preview = np.concatenate((preview, np.concatenate(depth_panels, axis=1)), axis=0)
        directory = output_dir or PROJECT_ROOT / "outputs/observe_tests"
        destination = directory / f"scene_{scene_index:03d}_stereo_preview.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), preview):
            raise RuntimeError(f"Impossibile salvare {destination}")
        print(
            f"\nScena {scene_index + 1}: {env.unwrapped.object_count} oggetti, "
            f"seed {first['scene_seed']}, target {first['target_id']}"
        )
        for item in records:
            metrics = item["metrics"]
            if not metrics["invariants_ok"]:
                print(
                    f"  {item['mode']:<8} OBSERVATION NON VALIDA: "
                    f"{'; '.join(metrics['invariant_violations'])} | "
                    f"grafo: {item.get('graph_svg') or item.get('graph_dot', '-')}"
                )
                continue
            position = metrics["mean_position_error_m"]
            f1 = metrics["support_f1"]
            position_text = "-" if position is None else f"{position:.3f} m"
            size_error = metrics["mean_size_error_m"]
            size_text = "-" if size_error is None else f"{size_error:.3f} m"
            shape_accuracy = metrics["shape_accuracy"]
            shape_text = "-" if shape_accuracy is None else f"{shape_accuracy:.2f}"
            print(
                f"  {item['mode']:<8} visti {metrics['detected_object_ids']} | "
                f"target visto: {'si' if metrics['target_detected'] else 'no'} | "
                f"errore: {position_text}"
            )
            print(
                f"    supporti F1: {'-' if f1 is None else f'{f1:.3f}'} | "
                f"errore dimensioni: {size_text} | "
                f"forme corrette: {shape_text} | "
                f"spazio incerto: {'si' if metrics['unknown_space'] else 'no'} | "
                f"grafo: {item.get('graph_svg') or item.get('graph_dot', '-')}"
            )
            for uncertainty in metrics.get("object_uncertainty", []):
                state = "ricordato" if uncertainty["remembered"] else "visibile"
                quality = uncertainty["detection_quality"]
                print(
                    f"      {uncertainty['object_id']}: {state}, "
                    f"qualita' {'sconosciuta' if quality is None else f'{quality:.2f}'}"
                )
        print(f"Maschere camera: sinistra {sorted(bundle.left_masks)}, destra {sorted(bundle.right_masks)}")
        print("Le etichette e [T] nella preview sono ground truth MuJoCo, non riconoscimento visivo.")
        print(f"Anteprima stereo: {destination}")
        print("Viewer MuJoCo e camere aperti; premi q o Esc nella finestra stereo per continuare.")
        cv2.imshow(window, preview)
        opened = True
        started = time.monotonic()
        while True:
            env.render()
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if seconds is not None and time.monotonic() - started >= seconds:
                break
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
        return destination
    finally:
        if opened:
            cv2.destroyWindow(window)
        env.close()


def _review_menu(
    report: dict, *, seconds: float | None = None, output_dir: Path | None = None,
) -> None:
    scene_count = int(report["config"]["scene_count"])
    print("\nApro la prima scena nel viewer MuJoCo/Gymnasium con le viste stereo.")
    review_scene(report, 0, seconds=seconds, output_dir=output_dir)
    while True:
        selected = _ask_int("Altra scena da rivedere (0 per uscire)", 0, 0, scene_count)
        if selected == 0:
            return
        review_scene(report, selected - 1, seconds=seconds, output_dir=output_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Banco di prova autonomo di OSSERVA")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=("all",) + ALL_MODES, default="all")
    parser.add_argument("--scenes", type=int)
    parser.add_argument("--objects", help="Numero di oggetti per scena oppure 'random'")
    parser.add_argument("--min-objects", type=int, default=1)
    parser.add_argument("--max-objects", type=int, default=6)
    parser.add_argument("--stereo-profile", choices=STEREO_PROFILES,
                        help="Disturbo delle rilevazioni stereo indipendente dal profilo Degraded")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--headless", action="store_true", help="Non aprire il viewer Gymnasium")
    parser.add_argument("--view-seconds", type=float, default=2.0)
    parser.add_argument("--review-report", type=Path, help="Riapre un report gia' salvato")
    parser.add_argument("--scene", type=int, default=1, help="Scena (da 1) per --review-report")
    parser.add_argument("--review-seconds", type=float, help="Chiude la revisione dopo N secondi")
    parser.add_argument(
        "--weights",
        type=Path,
        help="Pesi del detector per fusion_learned",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
        help="Soglia di confidenza del detector learned (default: 0.25)",
    )
    parser.add_argument("--min-target-immersion", type=float)
    parser.add_argument("--max-target-immersion", type=float)
    args = parser.parse_args(argv)
    if args.review_report is not None:
        report = json.loads(args.review_report.read_text())
        review_scene(
            report, args.scene - 1, seconds=args.review_seconds,
            output_dir=args.review_report.parent / args.review_report.stem,
        )
        return 0
    if args.config is None and sys.stdin.isatty():
        path, modes, interactive_stereo_profile = _interactive_selection()
    else:
        path = args.config or CONFIG_DIR / "clean.json"
        modes = DEFAULT_MODES if args.mode == "all" else (args.mode,)
        interactive_stereo_profile = None
    config = json.loads(path.read_text())
    stereo_profile = args.stereo_profile or interactive_stereo_profile
    if stereo_profile is not None:
        config["stereo"] = json.loads((CONFIG_DIR / f"{stereo_profile}.json").read_text())["stereo"]
        config["stereo_profile"] = stereo_profile
    if args.scenes is not None:
        config["scene_count"] = args.scenes
    elif args.config is None and sys.stdin.isatty():
        config["scene_count"] = _ask_int("Numero di scene", int(config["scene_count"]), 1)
    if args.objects is not None:
        if args.objects.lower() == "random":
            config["object_count"] = [args.min_objects, args.max_objects]
        else:
            try:
                config["object_count"] = int(args.objects)
            except ValueError:
                parser.error("--objects richiede un intero positivo oppure 'random'")
    elif args.config is None and sys.stdin.isatty():
        config["object_count"] = _ask_object_count(int(config["object_count"]))
    if args.seed is not None:
        config["seed"] = args.seed
    elif args.config is None and sys.stdin.isatty():
        config["seed"] = _ask_visual_seed(int(config["seed"]))
    if (args.min_target_immersion is None) != (args.max_target_immersion is None):
        parser.error(
            "--min-target-immersion e --max-target-immersion vanno specificati insieme"
        )
    if args.min_target_immersion is not None:
        config["target_immersion_range"] = [
            args.min_target_immersion,
            args.max_target_immersion,
        ]
    _object_count_for_scene(config["object_count"], int(config["seed"]))
    print(
        f"\nImpostazioni: {config['scene_count']} scene, oggetti per scena "
        f"{config['object_count']}, sorgenti {', '.join(modes)}"
    )
    if "stereo" in modes:
        print(f"Disturbo stereo: {stereo_profile or path.stem} (maschere/rilevazioni)")
    if "rgbd" in modes:
        print("RGB-D: depth renderizzata da MuJoCo; identita' da segmentazione oracle")
    if "fusion_oracle" in modes:
        print("Fusione: immagini B/N + punti LiDAR; solo le maschere sono oracle")
    if "fusion_learned" in modes:
        print("Fusione learned: maschere dalla camera B/N e geometria dal LiDAR")
    if config.get("target_immersion_range") is not None:
        print(
            "Immersione target casuale: "
            f"{config['target_immersion_range'][0]:.0%} - "
            f"{config['target_immersion_range'][1]:.0%}"
        )
    detector_weights = None
    if "fusion_learned" in modes:
        try:
            detector_weights = _resolve_detector_weights(args.weights)
        except FileNotFoundError as error:
            print(f"\nImpossibile avviare fusion_learned: {error}")
            return 1
        print(f"Modello detector: {detector_weights.name}")
    interactive_review = sys.stdin.isatty() and not args.headless
    report = run_benchmark(
        config, modes, show_viewer=not args.headless and not interactive_review,
        view_seconds=args.view_seconds,
        detector_weights=detector_weights,
        detector_confidence_threshold=args.confidence_threshold,
    )
    destination = args.output or PROJECT_ROOT / "outputs/observe_tests" / (
        datetime.now().astimezone().strftime("%Y%m%d_%H%M%S") + ".json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Il report resta disponibile anche se l'esportatore dei grafi fallisce.
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    graph_files = export_graphs(report, destination.parent / destination.stem)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("\nRisultati OSSERVA (media sulle scene; '-' = non misurabile)")
    for mode, summary in report["summary"].items():
        metrics = summary["metrics"]
        def show(key):
            value = metrics[key]["mean"]
            return "-" if value is None else f"{value:.3f}"
        print(
            f"  {mode:<8} oggetti recall={show('object_recall')}  "
            f"target recall={show('target_recall')}  "
            f"target precision={show('target_prediction_precision')}  "
            f"errore posizione={show('mean_position_error_m')} m  "
            f"errore dimensioni={show('mean_size_error_m')} m  "
            f"forme corrette={show('shape_accuracy')}  "
            f"supporti F1={show('support_f1')}  "
            f"relazioni non disponibili={summary['relation_unavailable_scenes']}  "
            f"scene incerte={summary['unknown_space_scenes']}  "
            f"invarianti violati={summary['invariant_violations']}"
        )
        curve = [
            item for item in summary["target_exposure_curve"]
            if item["scene_count"]
        ]
        if curve:
            print(
                "    curva esposizione target: "
                + ", ".join(
                    ((
                        "nascosta: "
                        if item.get("fully_hidden")
                        else f"{item['visible_fraction_min']:.0%}-"
                        f"{item['visible_fraction_max']:.0%}: "
                    ) + f"{item['target_recall']:.0%} (n={item['scene_count']})")
                    for item in curve
                )
            )
    print(f"Report JSON: {destination}")
    print(f"Grafi DOT/SVG: {destination.parent / destination.stem} ({len(graph_files)} file)")
    if "stereo" in modes:
        print("Stereo: ID e maschere ideali di MuJoCo; il detector RGB non e' ancora testato.")
    if "fusion_oracle" in modes:
        print(
            "Fusione: posizione e geometria provengono dal LiDAR; "
            "classi e maschere sono ancora oracle."
        )
    if "fusion_learned" in modes:
        print("Fusione learned: maschere e classi prodotte dal modello addestrato.")
    if interactive_review:
        _review_menu(
            report, seconds=args.review_seconds,
            output_dir=destination.parent / destination.stem,
        )
    return 1 if any(item["metrics"]["invariants_ok"] is False for item in report["scenes"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
