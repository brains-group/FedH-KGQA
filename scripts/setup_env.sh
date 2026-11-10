#!/usr/bin/env bash
# Create the conda environment "FedH-KGQA" and install the requirements.
# Falls back to a .venv if conda is not available.
set -euo pipefail
cd "$(dirname "$0")/.."
ENV_NAME=${ENV_NAME:-FedH-KGQA}
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda env list | grep -q "^$ENV_NAME " || conda create -n "$ENV_NAME" python=3.11 -y
  conda activate "$ENV_NAME"
  pip install -r requirements.txt
  python -c "import numpy, networkx; print('numpy', numpy.__version__, '| networkx', networkx.__version__)"
  echo "Environment ready. Activate with: conda activate $ENV_NAME"
else
  python3 -m venv --prompt "$ENV_NAME" .venv
  source .venv/bin/activate
  pip install --upgrade pip && pip install -r requirements.txt
  echo "Environment ready. Activate with: source .venv/bin/activate"
fi
