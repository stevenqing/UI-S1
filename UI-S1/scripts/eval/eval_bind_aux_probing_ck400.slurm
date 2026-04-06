#!/bin/bash
#SBATCH --job-name=probe_ba_ck400
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/probe_ba_ck400_%j.out
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/probe_ba_ck400_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:1

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
LORA_PATH="$PROJECT_DIR/train_GUI_360/llamafactory/output/bind_aux_r32/checkpoint-400"
PARQUET="$PROJECT_DIR/train_GUI_360/data/gui360_test_sft_eval_format_with_bbox.parquet"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_DIR/evaluation/results/probing_bind_aux_r32_ck400_${TIMESTAMP}"

echo "Step 2α Probing: bind_aux_r32 checkpoint-400 | Job ${SLURM_JOB_ID} | $(hostname) | $(date)"

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

python -u probing_diagnostic.py     --model_path "$BASE_MODEL"     --lora_path "$LORA_PATH"     --parquet_file "$PARQUET"     --n_samples 300     --output_dir "$OUTPUT_DIR"

echo "Done at $(date) | Results: $OUTPUT_DIR"
