#!/bin/bash
#SBATCH --job-name=probe_bind_aux
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/probe_bind_aux_%j.out
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/probe_bind_aux_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:1

# ============================================================
# Step 2α Eval Track 1: Probing Diagnostic (Probe C gap)
# Tests whether binding subspace improved after auxiliary loss
# ============================================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
LORA_PATH="$PROJECT_DIR/train_GUI_360/llamafactory/output/bind_aux_r32/final"
PARQUET="$PROJECT_DIR/train_GUI_360/data/gui360_test_sft_eval_format_with_bbox.parquet"
N_SAMPLES=300

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_DIR/evaluation/results/probing_bind_aux_r32_${TIMESTAMP}"

echo "============================================================"
echo "Step 2α Eval: Probing Diagnostic (bind_aux_r32)"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Base model: $BASE_MODEL"
echo "LoRA: $LORA_PATH"
echo "Samples: $N_SAMPLES"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export PYTHONUNBUFFERED=1

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

mkdir -p "$OUTPUT_DIR" "$PROJECT_DIR/scripts/eval/logs"

cd "$PROJECT_DIR/evaluation"

python -u probing_diagnostic.py \
    --model_path "$BASE_MODEL" \
    --lora_path "$LORA_PATH" \
    --parquet_file "$PARQUET" \
    --n_samples "$N_SAMPLES" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "Probing complete at $(date)"
echo "Results: $OUTPUT_DIR"
echo "============================================================"
