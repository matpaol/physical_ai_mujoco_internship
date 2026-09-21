"""Disturbi fotometrici riproducibili per immagini monocromatiche."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageDisturbance:
    contrast_range: tuple[float, float] = (0.75, 1.25)
    brightness_range: tuple[float, float] = (-25.0, 25.0)
    gamma_range: tuple[float, float] = (0.8, 1.2)
    noise_sigma_range: tuple[float, float] = (0.0, 8.0)
    blur_probability: float = 0.25
    blur_kernel: int = 3

    def __post_init__(self):
        ranges = (
            (self.contrast_range, True),
            (self.brightness_range, False),
            (self.gamma_range, True),
            (self.noise_sigma_range, False),
        )
        for interval, positive in ranges:
            if (
                len(interval) != 2
                or not np.isfinite(interval).all()
                or interval[0] > interval[1]
                or (positive and interval[0] <= 0)
            ):
                raise ValueError("Intervallo del disturbo immagine non valido")
        if self.noise_sigma_range[0] < 0 or not 0 <= self.blur_probability <= 1:
            raise ValueError("Rumore o probabilita' di blur non validi")
        if self.blur_kernel < 1 or self.blur_kernel % 2 == 0:
            raise ValueError("blur_kernel deve essere dispari e positivo")


def apply_image_disturbance(
    image: np.ndarray,
    policy: ImageDisturbance,
    rng: np.random.Generator,
) -> np.ndarray:
    raw = np.asarray(image)
    if raw.dtype != np.uint8 or raw.ndim not in (2, 3):
        raise ValueError("L'immagine deve essere uint8 HxW oppure HxWxC")
    result = raw.astype(np.float32)
    contrast = rng.uniform(*policy.contrast_range)
    brightness = rng.uniform(*policy.brightness_range)
    gamma = rng.uniform(*policy.gamma_range)
    result = np.clip(result * contrast + brightness, 0, 255)
    result = 255.0 * np.power(result / 255.0, gamma)
    sigma = rng.uniform(*policy.noise_sigma_range)
    if sigma:
        result += rng.normal(0.0, sigma, result.shape)
    result = np.clip(result, 0, 255).astype(np.uint8)
    if policy.blur_kernel > 1 and rng.random() < policy.blur_probability:
        result = cv2.GaussianBlur(result, (policy.blur_kernel, policy.blur_kernel), 0)
    return result
