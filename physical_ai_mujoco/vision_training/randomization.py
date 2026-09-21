"""Campionamento della domain randomization: dati puri, nessun simulatore.

Ogni aspetto ha il suo flusso casuale, derivato da (seed della ricetta, scopo,
scena, vista, ...). Cosi' aggiungere il rumore d'immagine non sposta le
estrazioni della camera, e cambiare la camera non cambia quale scena si
genera: i confronti fra ricette restano confronti fra la sola cosa cambiata.
"""

from __future__ import annotations

import math

import numpy as np

from physical_ai_mujoco.sensors import ImageDisturbance, apply_image_disturbance
from physical_ai_mujoco.simulation.visual_conditions import (
    BackgroundAppearance,
    CameraPerturbation,
    GroundAppearance,
    LightingConditions,
    VisualConditions,
)

from .recipe import ImageRandomization, RandomizationRecipe


# Scopi dei flussi casuali: numeri fissi, mai riusati per altro.
STREAM_OBJECT_COUNT = 1
STREAM_IMMERSION = 2
STREAM_VIEW = 3
STREAM_IMAGE = 4
STREAM_SPLIT = 5
STREAM_ABSENT = 6


def stream(seed: int, purpose: int, *keys: int) -> np.random.Generator:
    """Generatore indipendente per (seed, scopo, chiavi)."""
    return np.random.default_rng([int(seed), int(purpose), *(int(key) for key in keys)])


def _uniform(rng: np.random.Generator, interval) -> float:
    low, high = interval
    return float(rng.uniform(low, high)) if high > low else float(low)


def _jitter_rgb(rng, rgb, amount: float) -> tuple[float, float, float]:
    values = np.asarray(rgb[:3], dtype=float) + rng.uniform(-amount, amount, 3)
    return tuple(float(value) for value in np.clip(values, 0.0, 1.0))


def sample_visual_conditions(
    recipe: RandomizationRecipe,
    rng: np.random.Generator,
    *,
    object_rgb: dict[str, tuple[float, float, float]],
    target_id: str | None,
) -> VisualConditions:
    """Estrae le condizioni di una vista.

    `object_rgb` sono i colori nominali delle istanze in scena: gli ostacoli
    vengono perturbati attorno al proprio colore, il target pesca dalla
    tavolozza della ricetta (colori realistici della PFM-1).
    """
    camera = lighting = ground = background = None
    colors: dict[str, tuple[float, float, float]] = {}

    if recipe.camera is not None:
        item = recipe.camera
        camera = CameraPerturbation(
            tuple(float(v) for v in rng.uniform(-item.position_jitter_m, item.position_jitter_m, 3)),
            tuple(float(v) for v in rng.uniform(-item.target_jitter_m, item.target_jitter_m, 3)),
            float(rng.uniform(-item.roll_deg, item.roll_deg)),
            None if item.fovy_deg is None else _uniform(rng, item.fovy_deg),
        )
    if recipe.lighting is not None:
        item = recipe.lighting
        azimuth = math.radians(_uniform(rng, item.azimuth_deg))
        elevation = math.radians(_uniform(rng, item.elevation_deg))
        # Direzione di propagazione: dalla luce, in alto, verso la scena.
        direction = (
            -math.cos(elevation) * math.cos(azimuth),
            -math.cos(elevation) * math.sin(azimuth),
            -math.sin(elevation),
        )
        lighting = LightingConditions(
            direction,
            _uniform(rng, item.diffuse),
            _uniform(rng, item.ambient),
            _uniform(rng, item.headlight),
            bool(rng.random() < item.shadow_probability),
        )
    if recipe.ground is not None:
        item = recipe.ground
        kinds = sorted(item.kinds)
        weights = np.asarray([item.kinds[kind] for kind in kinds], dtype=float)
        kind = kinds[int(rng.choice(len(kinds), p=weights / weights.sum()))]
        ground = GroundAppearance(
            kind,
            _uniform(rng, item.gray),
            _uniform(rng, item.contrast),
            int(rng.integers(item.grain_px[0], item.grain_px[1] + 1)),
            int(rng.integers(0, 2**31 - 1)),
            None if item.reflectance is None else _uniform(rng, item.reflectance),
        )
    if recipe.background is not None:
        item = recipe.background
        background = BackgroundAppearance(
            _uniform(rng, item.gray),
            _uniform(rng, item.contrast),
            int(rng.integers(0, 2**31 - 1)),
        )
    if recipe.objects is not None:
        item = recipe.objects
        for instance_id in sorted(object_rgb):
            if instance_id == target_id and item.target_palette:
                base = item.target_palette[int(rng.integers(len(item.target_palette)))]
                colors[instance_id] = _jitter_rgb(rng, base, item.target_color_jitter)
            else:
                colors[instance_id] = _jitter_rgb(rng, object_rgb[instance_id], item.obstacle_color_jitter)

    return VisualConditions(camera, lighting, ground, background, colors)


def apply_image_effects(
    image: np.ndarray,
    recipe: ImageRandomization,
    rng: np.random.Generator,
) -> np.ndarray:
    """Disturbi della catena di acquisizione su un'immagine in grigio.

    Riusa il disturbo fotometrico dei sensori (contrasto, luminosita', gamma,
    rumore, sfocatura) e aggiunge la vignettatura dell'ottica.
    """
    policy = ImageDisturbance(
        contrast_range=recipe.contrast,
        brightness_range=recipe.brightness,
        gamma_range=recipe.gamma,
        noise_sigma_range=recipe.noise_sigma,
        blur_probability=recipe.blur_probability,
        blur_kernel=recipe.blur_kernel,
    )
    result = apply_image_disturbance(image, policy, rng)
    strength = _uniform(rng, recipe.vignette)
    if strength > 0:
        height, width = result.shape[:2]
        rows = (np.arange(height) - (height - 1) / 2.0) / (height / 2.0)
        columns = (np.arange(width) - (width - 1) / 2.0) / (width / 2.0)
        radius = np.sqrt(rows[:, None] ** 2 + columns[None, :] ** 2) / math.sqrt(2.0)
        gain = 1.0 - strength * radius**2
        if result.ndim == 3:
            gain = gain[:, :, None]
        result = np.clip(result.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return result
