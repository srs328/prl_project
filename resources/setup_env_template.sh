#!/bin/bash
# Setup environment variables for local PRL pipeline execution
# Source this before running any pipeline scripts:
#   source setup_env.sh

export PRL_PROJECT_ROOT="/home/srs-9/Projects/prl_project"
export PRL_DATA_ROOT="/media/smbshare/srs-9/prl_project/data"
export PRL_TRAIN_ROOT="/media/smbshare/srs-9/prl_project/training"
export PRL_PROJECT_ROOT="/home/shridhar.singh9-umw/prl_project/prl_project"
export PRL_DATA_ROOT="/home/shridhar.singh9-umw/prl_project/data"
export PRL_TRAIN_ROOT="/home/shridhar.singh9-umw/prl_project/training"
export PRL_TRAIN_HOME="/home/shridhar.singh9-umw/prl_project/prl_project/training"

source $PRL_PROJECT_ROOT/.venv/bin/activate
export PYTHONPATH="${PRL_PROJECT_ROOT}/src:$PYTHONPATH"


echo "PRL Pipeline environment configured:"
echo "  PRL_PROJECT_ROOT=$PRL_PROJECT_ROOT"
echo "  PRL_DATA_ROOT=$PRL_DATA_ROOT"
echo "  PRL_TRAIN_ROOT=$PRL_TRAIN_ROOT"
