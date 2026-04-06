#!/bin/bash
#SBATCH --job-name=probing_diag
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/probing_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/probing_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================================
# Cross-Modal Probing Diagnostic
# ============================================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="$PROJECT_DIR/evaluation"
LOG_DIR="$PROJECT_DIR/train_GUI_360/logs"

MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/train_GUI_360/llamafactory/output/gui360_full_sft_v2/checkpoint-796}"
PARQUET="${PARQUET:-$PROJECT_DIR/train_GUI_360/data/gui360_test_sft_eval_format_with_bbox.parquet}"
N_SAMPLES="${N_SAMPLES:-300}"

MODEL_SHORT=$(basename "$MODEL_PATH")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/evaluation/results/probing_${MODEL_SHORT}_${TIMESTAMP}}"

echo "============================================================"
echo "Cross-Modal Probing Diagnostic"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Model: $MODEL_PATH"
echo "Samples: $N_SAMPLES"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

export TMPDIR=/tmp/${SLURM_JOB_ID}
mkdir -p $TMPDIR 2>/dev/null || true
if ! touch "$TMPDIR/.test_write" 2>/dev/null; then
    export TMPDIR=/scratch/a5l/shuqing.a5l/tmp/${SLURM_JOB_ID}
    mkdir -p $TMPDIR
fi
rm -f "$TMPDIR/.test_write" 2>/dev/null
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor
export TORCH_HOME=$TMPDIR/torch
export XDG_CACHE_HOME=$TMPDIR/cache
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TORCH_HOME $XDG_CACHE_HOME

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

cd "$EVAL_DIR"

export PYTHONUNBUFFERED=1

python -u probing_diagnostic.py \
    --model_path "$MODEL_PATH" \
    --parquet_file "$PARQUET" \
    --n_samples "$N_SAMPLES" \
    --output_dir "$OUTPUT_DIR"

echo "Done at $(date)"
