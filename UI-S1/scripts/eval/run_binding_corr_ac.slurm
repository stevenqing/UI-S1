#!/bin/bash
#SBATCH --job-name=bc_ac
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/bc_ac_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/bc_ac_%j.err
#SBATCH --time=1:30:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:1

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_DIR/evaluation/results/binding_corr_ac_base_${TIMESTAMP}"

echo "Phase 1.8: AC Binding Correlation | Job ${SLURM_JOB_ID} | $(hostname) | $(date)"

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
mkdir -p "$OUTPUT_DIR"

python -u evaluation/binding_accuracy_correlation.py \
    --model_path "$MODEL_PATH" \
    --parquet_file "$PROJECT_DIR/evaluation/dataset/ac_diagnostic_click_steps.parquet" \
    --image_base "$PROJECT_DIR/datasets" \
    --n_samples 500 \
    --output_dir "$OUTPUT_DIR"

echo "Done at $(date) | Output: $OUTPUT_DIR"
