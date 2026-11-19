#!/bin/bash
# Setup environment variables for local PRL pipeline execution
# Source this template from the checkout:
#   source resources/setup_env_template.sh

export PRL_PROJECT_ROOT="${PRL_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PRL_DATA_ROOT="${PRL_DATA_ROOT:-/media/smbshare/srs-9/prl_project/data}"
export PRL_TRAIN_ROOT="${PRL_TRAIN_ROOT:-/media/smbshare/srs-9/prl_project/training}"
export PRL_TRAIN_HOME="${PRL_TRAIN_HOME:-$PRL_PROJECT_ROOT/training}"

# Install this checkout from its root with python -m pip install -e . once.
source "${PRL_VENV:-$PRL_PROJECT_ROOT/.venv}/bin/activate"

echo "PRL Pipeline environment configured:"
echo "  PRL_PROJECT_ROOT=$PRL_PROJECT_ROOT"
echo "  PRL_DATA_ROOT=$PRL_DATA_ROOT"
echo "  PRL_TRAIN_ROOT=$PRL_TRAIN_ROOT"
