"""Disturbi espliciti e riproducibili per detection, depth e calibrazione."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DetectionNoise:
    centroid_noise_px: float = 0.0
    edge_erosion_px: int = 0
    edge_dilation_px: int = 0
    drop_probability: float = 0.0
    drop_small_objects: bool = False
    min_area_px: int = 16
    fuse_nearby: bool = False
    false_positive_probability: float = 0.0
    type_confusion: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        if not np.isfinite(self.centroid_noise_px) or self.centroid_noise_px < 0:
            raise ValueError("Rumore dei centroidi non valido")
        if min(self.edge_erosion_px, self.edge_dilation_px) < 0 or self.min_area_px < 1:
            raise ValueError("Dimensioni del disturbo detection non valide")
        if not 0 <= self.drop_probability <= 1 or not 0 <= self.false_positive_probability <= 1:
            raise ValueError("Probabilita' detection fuori [0,1]")
        for distribution in self.type_confusion.values():
            if any(value < 0 for value in distribution.values()) or sum(distribution.values()) > 1 + 1e-9:
                raise ValueError("Confusione di tipo non valida")


@dataclass(frozen=True)
class CalibrationNoise:
    translation_sigma: float = 0.0
    rotation_sigma_deg: float = 0.0
    baseline_sigma: float = 0.0

    def __post_init__(self):
        if min(self.translation_sigma, self.rotation_sigma_deg, self.baseline_sigma) < 0:
            raise ValueError("Sigma di calibrazione negativo")


@dataclass(frozen=True)
class DepthNoise:
    distance_sigma: float = 0.0
    dropout_probability: float = 0.0
    edge_sigma_multiplier: float = 1.0

    def __post_init__(self):
        if self.distance_sigma < 0 or not 0 <= self.dropout_probability <= 1 or self.edge_sigma_multiplier < 1:
            raise ValueError("Disturbo depth non valido")


def _shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    height, width = mask.shape
    shifted = np.zeros_like(mask)
    source_x0, source_x1 = max(0, -dx), min(width, width - dx)
    source_y0, source_y1 = max(0, -dy), min(height, height - dy)
    if source_x0 < source_x1 and source_y0 < source_y1:
        shifted[source_y0 + dy:source_y1 + dy, source_x0 + dx:source_x1 + dx] = (
            mask[source_y0:source_y1, source_x0:source_x1]
        )
    return shifted


def apply_detection_noise(masks, noise: DetectionNoise, rng: np.random.Generator):
    import cv2

    result = {}
    for object_id, mask in sorted(masks.items()):
        image = np.asarray(mask, dtype=bool)
        if image.ndim != 2:
            raise ValueError("La maschera deve essere bidimensionale")
        if noise.drop_small_objects and image.sum() < noise.min_area_px:
            continue
        if rng.random() < noise.drop_probability:
            continue
        out = image.astype(np.uint8)
        if noise.centroid_noise_px:
            dx, dy = np.rint(rng.normal(0, noise.centroid_noise_px, 2)).astype(int)
            out = _shift_mask(out, int(dx), int(dy))
        if noise.edge_erosion_px:
            out = cv2.erode(out, np.ones((3, 3), np.uint8), iterations=noise.edge_erosion_px)
        if noise.edge_dilation_px:
            out = cv2.dilate(out, np.ones((3, 3), np.uint8), iterations=noise.edge_dilation_px)
        if out.any():
            result[object_id] = out.astype(bool)
    if noise.fuse_nearby:
        # Fusioni solo tra maschere che si toccano dopo dilatazione di 1 px.
        pending = dict(result)
        result = {}
        while pending:
            key, merged = pending.popitem()
            names = [key]
            changed = True
            while changed:
                changed = False
                rim = cv2.dilate(merged.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                for other, candidate in list(pending.items()):
                    if np.any(rim & candidate):
                        merged = merged | pending.pop(other)
                        names.append(other)
                        changed = True
            result["fused:" + "+".join(sorted(names)) if len(names) > 1 else key] = merged
    if masks and rng.random() < noise.false_positive_probability:
        shape = next(iter(masks.values())).shape
        y = int(rng.integers(0, shape[0]))
        x = int(rng.integers(0, shape[1]))
        false_mask = np.zeros(shape, dtype=bool)
        false_mask[max(0, y-2):y+3, max(0, x-2):x+3] = True
        result["false_positive"] = false_mask
    return result


def apply_type_confusion(type_ids: dict[str, str] | None, noise: DetectionNoise,
                         rng: np.random.Generator) -> dict[str, str] | None:
    if type_ids is None:
        return None
    result = {}
    for object_id, original in sorted(type_ids.items()):
        distribution = noise.type_confusion.get(original, {})
        value = rng.random()
        result[object_id] = original
        for candidate, probability in distribution.items():
            value -= probability
            if value < 0:
                result[object_id] = candidate
                break
    return result


def apply_depth_noise(depth: np.ndarray, noise: DepthNoise,
                      rng: np.random.Generator) -> np.ndarray:
    image = np.asarray(depth, dtype=float)
    if image.ndim != 2:
        raise ValueError("La depth deve essere bidimensionale")
    valid = np.isfinite(image) & (image > 0)
    result = image.copy()
    if noise.distance_sigma:
        filled = np.where(valid, image, 0.0)
        gy, gx = np.gradient(filled)
        edges = (np.abs(gx) + np.abs(gy)) > 0.02
        scale = np.where(edges, noise.edge_sigma_multiplier, 1.0)
        result[valid] += rng.normal(0, noise.distance_sigma, image.shape)[valid] * scale[valid]
    result[valid & (result <= 0)] = np.nan
    if noise.dropout_probability:
        result[valid & (rng.random(image.shape) < noise.dropout_probability)] = np.nan
    return result
