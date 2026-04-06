#!/bin/bash
#SBATCH --job-name=ac_diag
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/ac_diag_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/ac_diag_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================================
# AC Full Diagnostic: Phase 1 (Probing) + Phase 1.75 (Binding)
# ============================================================
# Runs probing_diagnostic.py (Probe A/B/C) and
# binding_analysis.py (Exp E+F) on AndroidControl data
# using the base model (Qwen2.5-VL-7B-Instruct).
#
# No vLLM needed — loads model directly via transformers.
# ============================================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="$PROJECT_DIR/evaluation"

MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct}"
IMAGE_BASE="${IMAGE_BASE:-$PROJECT_DIR/datasets}"
N_SAMPLES="${N_SAMPLES:-300}"

PARQUET="$EVAL_DIR/dataset/ac_diagnostic_click_steps.parquet"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "AndroidControl Diagnostic (Probing + Binding Analysis)"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Model: $MODEL_PATH"
echo "Parquet: $PARQUET"
echo "Samples: $N_SAMPLES"
echo "============================================================"

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export PYTHONUNBUFFERED=1

# Temp directory setup
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

mkdir -p "$PROJECT_DIR/scripts/eval/logs"

# ── Step 0: Generate parquet if needed ──
if [ ! -f "$PARQUET" ]; then
    echo "Generating AC diagnostic parquet..."
    cd "$EVAL_DIR"
    python -u prepare_ac_diagnostic_data.py \
        --input "$PROJECT_DIR/evaluation/dataset/android_control_evaluation_std.jsonl" \
        --output "$PARQUET" \
        --image_base "$IMAGE_BASE"
    echo "Parquet generated: $(python -c "import pandas as pd; print(len(pd.read_parquet('$PARQUET')), 'rows')")"
fi

cd "$EVAL_DIR"

# ── Step 1: Probing Diagnostic (Probe A/B/C) ──
PROBING_OUTPUT="$PROJECT_DIR/evaluation/results/ac_probing_base_${TIMESTAMP}"
mkdir -p "$PROBING_OUTPUT"
echo ""
echo "============================================================"
echo "Phase 1: Probing Diagnostic"
echo "Output: $PROBING_OUTPUT"
echo "============================================================"

python -u probing_diagnostic.py \
    --model_path "$MODEL_PATH" \
    --parquet_file "$PARQUET" \
    --image_base "$IMAGE_BASE" \
    --n_samples "$N_SAMPLES" \
    --output_dir "$PROBING_OUTPUT"

echo "Probing complete at $(date)"

# ── Step 2: Binding Analysis (Exp E+F) ──
BINDING_OUTPUT="$PROJECT_DIR/evaluation/results/ac_binding_base_${TIMESTAMP}"
mkdir -p "$BINDING_OUTPUT"
echo ""
echo "============================================================"
echo "Phase 1.75: Binding Analysis (Exp E+F)"
echo "Output: $BINDING_OUTPUT"
echo "============================================================"

python -u binding_analysis.py \
    --model_path "$MODEL_PATH" \
    --parquet_file "$PARQUET" \
    --image_base "$IMAGE_BASE" \
    --n_samples "$N_SAMPLES" \
    --output_dir "$BINDING_OUTPUT"

echo "Binding analysis complete at $(date)"
echo ""
echo "============================================================"
echo "All diagnostics complete!"
echo "Probing results: $PROBING_OUTPUT"
echo "Binding results: $BINDING_OUTPUT"
echo "============================================================"
