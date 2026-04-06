#!/bin/bash
# Submit all baseline model evaluations on AndroidControl
# Usage: bash submit_all_baselines_ac.sh [EVAL_WORKERS]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/eval_baseline_ac.slurm"
WORKERS="${1:-32}"

echo "Submitting baseline AC evaluations with $WORKERS workers..."
echo ""

for MODEL_TYPE in os-atlas ui-tars os-genesis; do
    JOB_ID=$(sbatch --export=ALL,MODEL_TYPE=$MODEL_TYPE,EVAL_WORKERS=$WORKERS \
        "$SLURM_SCRIPT" 2>&1 | grep -oP '\d+')
    echo "  $MODEL_TYPE: Job $JOB_ID"
done

echo ""
echo "Monitor with: squeue -u \$USER"
