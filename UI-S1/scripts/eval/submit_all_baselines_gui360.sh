#!/bin/bash
# Submit all baseline model evaluations on GUI-360
# Usage: bash submit_all_baselines_gui360.sh [EVAL_WORKERS]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/eval_baseline_gui360.slurm"
WORKERS="${1:-16}"

echo "Submitting baseline GUI-360 evaluations with $WORKERS workers..."
echo ""

for MODEL_TYPE in os_atlas os_genesis ui_tars; do
    JOB_ID=$(sbatch --export=ALL,MODEL_TYPE=$MODEL_TYPE,EVAL_WORKERS=$WORKERS \
        "$SLURM_SCRIPT" 2>&1 | grep -oP '\d+')
    echo "  $MODEL_TYPE: Job $JOB_ID"
done

echo ""
echo "Monitor with: squeue -u \$USER"
