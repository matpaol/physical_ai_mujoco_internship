"""Addestramento del visore (detector di segmentazione) per OSSERVA.

Non e' codice deployable: produce dataset, modelli e valutazioni che il
detector di `sensors/learned_detector.py` poi carica. Il ciclo e' sempre lo
stesso, sim o reale:

    ricetta dataset --> generate_dataset --> dataset (train/val/test)
    ricetta training --> train --> pesi .pt + scheda .json
    evaluate --> richiamo del target per fasce di % visibile, scritto nella scheda

Questo pacchetto non importa MuJoCo: l'aspetto della scena si cambia solo
tramite `Simulator.apply_visual_conditions`.
"""

from .recipe import (
    DatasetRecipe,
    EvaluationSettings,
    RecipeError,
    TrainingRecipe,
    available_recipes,
    load_recipe,
)

__all__ = [
    "DatasetRecipe",
    "EvaluationSettings",
    "RecipeError",
    "TrainingRecipe",
    "available_recipes",
    "load_recipe",
    "generate_dataset",
    "train",
    "evaluate",
]


def generate_dataset(*args, **kwargs):
    """Vedi `vision_training.dataset.generate_dataset`."""
    from .dataset import generate_dataset as implementation

    return implementation(*args, **kwargs)


def train(*args, **kwargs):
    """Vedi `vision_training.training.train`."""
    from .training import train as implementation

    return implementation(*args, **kwargs)


def evaluate(*args, **kwargs):
    """Vedi `vision_training.evaluation.evaluate`."""
    from .evaluation import evaluate as implementation

    return implementation(*args, **kwargs)
