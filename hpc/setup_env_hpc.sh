#!/bin/bash
# Setup environment for PRL pipeline on HPC (UMass Chan SCI Cluster, IBM LSF)
# Source this in job scripts before running pipeline commands

# Load required modules (uncomment and adjust versions as needed)
# module load python/3.x
# module load cuda/11.x

# Activate the MONAI virtual environment
source ~/.virtualenvs/monai/bin/activate

# Export environment variables for HPC paths
# Override these if your HPC paths differ from local
export PRL_PROJECT_ROOT="${PRL_PROJECT_ROOT:-/home/srs-9/Projects/prl_project}"
export PRL_DATA_ROOT="${PRL_DATA_ROOT:-/media/smbshare/srs-9/prl_project/data}"
export PRL_TRAIN_ROOT="${PRL_TRAIN_ROOT:-/media/smbshare/srs-9/prl_project/training}"

# Ensure Python can import the project
export PYTHONPATH="${PRL_PROJECT_ROOT}/src:${PRL_PROJECT_ROOT}:${PYTHONPATH}"

echo "HPC environment configured:"
echo "  PRL_PROJECT_ROOT=$PRL_PROJECT_ROOT"
echo "  PRL_DATA_ROOT=$PRL_DATA_ROOT"
echo "  PRL_TRAIN_ROOT=$PRL_TRAIN_ROOT"
echo "  Python: $(which python)"
