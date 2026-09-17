"""Configurazione e provenienza salvate insieme agli artefatti."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from physical_ai_mujoco.infrastructure.builder import PROJECT_ROOT


def save_run_metadata(path, parameters, metrics=None):
    versions = {}
    for name in ("mujoco", "numpy", "gymnasium", "stable-baselines3"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    configs = {
        str(p.relative_to(PROJECT_ROOT)): json.loads(p.read_text())
        for folder in ("configs", "datasets")
        for p in sorted((PROJECT_ROOT / folder).rglob("*.json"))
    }
    digest = hashlib.sha256()
    for p in sorted((PROJECT_ROOT / "physical_ai_mujoco").rglob("*.py")):
        digest.update(str(p.relative_to(PROJECT_ROOT)).encode())
        digest.update(p.read_bytes())

    def git(*args):
        result = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else None

    from physical_ai_mujoco.infrastructure.experiment import selected_profile

    profile = selected_profile()
    document = dict(
        experiment=None
        if profile is None
        else {
            "path": str(profile.path),
            "name": profile.name,
            "env_overrides": profile.env_overrides,
        },
        timestamp=datetime.now(timezone.utc).isoformat(),
        parameters=parameters,
        versions=versions,
        git_commit=git("rev-parse", "HEAD"),
        git_dirty=bool(git("status", "--porcelain")),
        source_sha256=digest.hexdigest(),
        inputs=configs,
        metrics=metrics,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n")
