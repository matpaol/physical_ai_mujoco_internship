"""Profili di avvio. Le fasi sono dati, non rami degli algoritmi scientifici."""

from dataclasses import dataclass
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[2]
PROFILE_VARIABLE = "PHYSICAL_AI_EXPERIMENT"


@dataclass(frozen=True)
class ExperimentProfile:
    path: Path
    name: str
    available: bool
    description: str
    default_decider: str
    operations: tuple[str, ...]
    env_overrides: dict

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            path,
            data["name"],
            data["available"],
            data["description"],
            data["default_decider"],
            tuple(data["operations"]),
            data["env_overrides"],
        )

    def activate(self):
        if not self.available:
            raise ValueError(self.description)
        # I processi di training/video lanciati dal menu ereditano il profilo.
        os.environ[PROFILE_VARIABLE] = str(self.path)


def available_profiles():
    return [
        ExperimentProfile.load(p)
        for p in sorted((ROOT / "configs/experiments").glob("*.json"))
    ]


def selected_profile():
    path = os.environ.get(PROFILE_VARIABLE)
    return ExperimentProfile.load(path) if path else None
