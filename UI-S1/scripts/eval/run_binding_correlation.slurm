#!/bin/bash
#SBATCH --job-name=binding_corr
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/binding_corr_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/binding_corr_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:1

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct}"
N_SAMPLES="${N_SAMPLES:-500}"

GUI360_PARQUET="$PROJECT_DIR/train_GUI_360/data/gui360_test_sft_eval_format_with_bbox.parquet"
AC_PARQUET="$PROJECT_DIR/evaluation/dataset/ac_diagnostic_click_steps.parquet"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SHORT=$(basename "$MODEL_PATH")

echo "============================================================"
echo "Phase 1.8: Binding Quality vs Accuracy Correlation"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Model: $MODEL_PATH"
echo "N samples: $N_SAMPLES"
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

export PYTHONUNBUFFERED=1

cd "$PROJECT_DIR"

# ── GUI-360 ──
echo ""
echo "============================================================"
echo "Running GUI-360 binding correlation..."
echo "============================================================"

GUI360_OUTPUT="$PROJECT_DIR/evaluation/results/binding_corr_gui360_${MODEL_SHORT}_${TIMESTAMP}"
mkdir -p "$GUI360_OUTPUT"

python -u evaluation/binding_accuracy_correlation.py \
    --model_path "$MODEL_PATH" \
    --parquet_file "$GUI360_PARQUET" \
    --n_samples "$N_SAMPLES" \
    --output_dir "$GUI360_OUTPUT"

echo "GUI-360 done."

# ── AC ──
echo ""
echo "============================================================"
echo "Running AC binding correlation..."
echo "============================================================"

AC_OUTPUT="$PROJECT_DIR/evaluation/results/binding_corr_ac_${MODEL_SHORT}_${TIMESTAMP}"
mkdir -p "$AC_OUTPUT"

python -u evaluation/binding_accuracy_correlation.py \
    --model_path "$MODEL_PATH" \
    --parquet_file "$AC_PARQUET" \
    --image_base "$PROJECT_DIR/datasets" \
    --n_samples "$N_SAMPLES" \
    --output_dir "$AC_OUTPUT"

echo "AC done."

echo ""
echo "============================================================"
echo "All done at $(date)"
echo "GUI-360 results: $GUI360_OUTPUT"
echo "AC results: $AC_OUTPUT"
echo "============================================================"
