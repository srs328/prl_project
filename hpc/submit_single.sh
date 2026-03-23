#!/bin/bash
# Template for submitting a single training run to LSF (UMass Chan SCI Cluster)
# Usage: bash submit_single.sh --run-dir /path/to/run<N>

# Parse arguments
run_dir=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)
            run_dir="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 --run-dir <path>"
            exit 1
            ;;
    esac
done

if [ -z "$run_dir" ]; then
    echo "ERROR: --run-dir is required"
    exit 1
fi

# Create a temporary job script
job_script="/tmp/prl_train_$(date +%s).sh"

cat > "$job_script" <<'JOBEOF'
#!/bin/bash
#BSUB -n 1
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -R "rusage[mem=32G]"
#BSUB -W 24:00
#BSUB -o RUN_DIR/train_%J.log
#BSUB -e RUN_DIR/train_%J.err

# Source HPC environment setup
source PROJECT_ROOT/hpc/setup_env_hpc.sh

echo "Starting training in RUN_DIR"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
cd RUN_DIR
python PROJECT_ROOT/training/roi_train2/train.py --run-dir RUN_DIR
echo "Training completed"
JOBEOF

# Replace placeholders
sed -i "s|PROJECT_ROOT|${PRL_PROJECT_ROOT}|g" "$job_script"
sed -i "s|RUN_DIR|${run_dir}|g" "$job_script"

chmod +x "$job_script"

echo "Submitting job script: $job_script"
bsub < "$job_script"
