#!/usr/bin/env bash

if [[ -n "${BASH_VERSION:-}" && "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'Run this script with: source start_project.sh\n' >&2
    exit 1
fi

if [[ -n "${ZSH_VERSION:-}" && "${ZSH_EVAL_CONTEXT:-}" != *:file ]]; then
    printf 'Run this script with: source start_project.sh\n' >&2
    exit 1
fi

if [[ -n "${BASH_VERSION:-}" ]]; then
    SCRIPT_PATH="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
    SCRIPT_PATH="${(%):-%N}"
else
    printf 'This script requires Bash or Zsh.\n' >&2
    return 1
fi

PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
ENV_NAME="mujoco-tirocinio"

find_conda_base() {
    if command -v conda >/dev/null 2>&1; then
        conda info --base 2>/dev/null
        return
    fi

    local candidate
    for candidate in \
        "/opt/miniconda3" \
        "$HOME/miniconda3" \
        "$HOME/miniforge3" \
        "$HOME/anaconda3"
    do
        if [[ -x "$candidate/bin/conda" ]]; then
            "$candidate/bin/conda" info --base
            return
        fi
    done

    return 1
}

CONDA_BASE="$(find_conda_base)"
if [[ -z "$CONDA_BASE" || ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    printf 'Conda was not found. Install Miniconda or Miniforge first.\n' >&2
    return 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda run -n "$ENV_NAME" python --version >/dev/null 2>&1; then
    printf 'Creating Conda environment %s...\n' "$ENV_NAME"
    conda create -n "$ENV_NAME" python=3.12 pip -y || return 1
fi

conda activate "$ENV_NAME" || return 1
cd "$PROJECT_ROOT" || return 1

printf 'Installing the project from %s...\n' "$PROJECT_ROOT"
python -m pip install -e ".[test,train,video]" || return 1

PROJECT_PYTHON="$(python -c 'import sys; print(sys.executable)')"
VSCODE_SETTINGS="$PROJECT_ROOT/.vscode/settings.json"
mkdir -p "$PROJECT_ROOT/.vscode"

python - "$VSCODE_SETTINGS" "$PROJECT_PYTHON" <<'PY'
import json
import sys
from pathlib import Path

settings_path = Path(sys.argv[1])
python_path = sys.argv[2]

if settings_path.exists():
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
else:
    settings = {}

settings.update(
    {
        "python.defaultInterpreterPath": python_path,
        "python.analysis.enableEditableInstalls": True,
        "python.testing.pytestEnabled": True,
        "python.testing.pytestPath": str(Path(python_path).parent / "pytest"),
        "python.testing.pytestArgs": ["tests"],
        "python.testing.unittestEnabled": False,
    }
)

settings_path.write_text(
    json.dumps(settings, indent=2) + "\n",
    encoding="utf-8",
)
PY

python -c "import mujoco, numpy, pytest, physical_ai_mujoco" || return 1

printf 'Project ready.\n'
printf 'Environment: %s\n' "$CONDA_DEFAULT_ENV"
printf 'Python: %s\n' "$PROJECT_PYTHON"
printf 'Working directory: %s\n' "$PROJECT_ROOT"
