#!/bin/bash
# Setup environment for PRL pipeline on HPC (UMass Chan SCI Cluster, IBM LSF)
# Source this in job scripts before running pipeline commands

# Load required modules (uncomment and adjust versions as needed)
# module load python/3.x
# module load cuda/11.x

# Export environment variables for HPC paths
# Override these if your HPC paths differ from local
export PRL_PROJECT_ROOT="${PRL_PROJECT_ROOT:-/home/shridhar.singh9-umw/prl_project/prl_project}"
export PRL_DATA_ROOT="${PRL_DATA_ROOT:-/home/shridhar.singh9-umw/prl_project/data}"
export PRL_TRAIN_ROOT="${PRL_TRAIN_ROOT:-/home/shridhar.singh9-umw/prl_project/training}"

# Install this checkout with python -m pip install -e . in the environment once.
source "${PRL_PROJECT_ROOT}/.venv/bin/activate"

echo "HPC environment configured:"
echo "  PRL_PROJECT_ROOT=$PRL_PROJECT_ROOT"
echo "  PRL_DATA_ROOT=$PRL_DATA_ROOT"
echo "  PRL_TRAIN_ROOT=$PRL_TRAIN_ROOT"
echo "  Python: $(which python)"
