#!/bin/bash
# Setup environment for PRL pipeline on HPC (IBM LSF)
# This script loads required modules and sets environment variables
# Source this in job scripts before running pipeline commands

# Load required modules (adjust based on your HPC environment)
# module load fsl/6.x.x          # FSL for image processing
# module load c3d/1.x.x          # C3D for image operations
# module load python/3.x         # Python
# module load cuda/11.x          # CUDA if using GPU

ml cuda

# Export environment variables for local/HPC setups
export PRL_PROJECT_ROOT="${PRL_PROJECT_ROOT:-/home/shridhar.singh9-umw/prl_project/prl_project}"
export PRL_DATA_ROOT="${PRL_DATA_ROOT:-/home/shridhar.singh9-umw/prl_project/data}"
export PRL_TRAIN_ROOT="${PRL_TRAIN_ROOT:-/home/shridhar.singh9-umw/prl_project/training}"
source $PRL_PROJECT_ROOT/.venv/bin/activate

# Ensure Python can import the project
export PYTHONPATH="${PRL_PROJECT_ROOT}/src:${PYTHONPATH}"

echo "HPC environment configured:"
echo "  PRL_PROJECT_ROOT=$PRL_PROJECT_ROOT"
echo "  PRL_DATA_ROOT=$PRL_DATA_ROOT"
echo "  PRL_TRAIN_ROOT=$PRL_TRAIN_ROOT"
