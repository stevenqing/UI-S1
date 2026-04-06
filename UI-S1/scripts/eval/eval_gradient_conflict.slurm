#!/bin/bash
#SBATCH --job-name=grad_conflict
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/grad_conflict_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/grad_conflict_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:1

# ============================================
# Gradient Conflict Analysis: L_bind vs L_act
# Sequential runs: SFT v2 then Base model
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_SCRIPT="$PROJECT_DIR/evaluation/gradient_conflict_analysis.py"
PARQUET_FILE="$PROJECT_DIR/train_GUI_360/data/gui360_eval_sft.parquet"
IMAGE_BASE="$PROJECT_DIR"
LOG_DIR="$PROJECT_DIR/scripts/eval/logs"

# Model paths
SFT_V2_PATH="$PROJECT_DIR/train_GUI_360/llamafactory/output/gui360_full_sft_v2"
BASE_PATH="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "Gradient Conflict Analysis"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "SFT v2: $SFT_V2_PATH"
echo "Base:   $BASE_PATH"
echo "=========================================="

# Environment setup
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

# CUDA/Library paths
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

# Fix TMPDIR/cache
export TMPDIR=/tmp/${SLURM_JOB_ID}
mkdir -p $TMPDIR
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor
export TORCH_HOME=$TMPDIR/torch
export XDG_CACHE_HOME=$TMPDIR/cache
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TORCH_HOME $XDG_CACHE_HOME

mkdir -p "$LOG_DIR"

# ============================================
# Run 1: SFT v2
# ============================================
SFT_OUTPUT="$PROJECT_DIR/evaluation/results/gradient_conflict_sft_v2_${TIMESTAMP}"

echo ""
echo "=== Run 1: SFT v2 ==="
echo "Start: $(date)"
echo "Output: $SFT_OUTPUT"

python "$EVAL_SCRIPT" \
    --model_path "$SFT_V2_PATH" \
    --parquet_file "$PARQUET_FILE" \
    --image_base "$IMAGE_BASE" \
    --n_samples 200 --seed 42 \
    --output_dir "$SFT_OUTPUT"

echo "SFT v2 done: $(date)"

# ============================================
# Run 2: Base model
# ============================================
BASE_OUTPUT="$PROJECT_DIR/evaluation/results/gradient_conflict_base_${TIMESTAMP}"

echo ""
echo "=== Run 2: Base (Qwen2.5-VL-7B-Instruct) ==="
echo "Start: $(date)"
echo "Output: $BASE_OUTPUT"

python "$EVAL_SCRIPT" \
    --model_path "$BASE_PATH" \
    --parquet_file "$PARQUET_FILE" \
    --image_base "$IMAGE_BASE" \
    --n_samples 200 --seed 42 \
    --output_dir "$BASE_OUTPUT"

echo "Base done: $(date)"

# ============================================
# Cross-model comparison
# ============================================
echo ""
echo "=========================================="
echo "Cross-Model Comparison"
echo "=========================================="

python3 -c "
import json
import sys

sft_path = '$SFT_OUTPUT/per_layer_conflict.json'
base_path = '$BASE_OUTPUT/per_layer_conflict.json'

try:
    with open(sft_path) as f:
        sft = json.load(f)
    with open(base_path) as f:
        base = json.load(f)
except FileNotFoundError as e:
    print(f'Error loading results: {e}')
    sys.exit(1)

print(f\"{'Layer':>6} | {'SFT v2 conflict':>16} | {'Base conflict':>14} | {'Delta':>8} | {'Flag':>12}\")
print('-' * 75)

for l in range(28):
    key = str(l)
    sft_c = sft.get(key, {}).get('conflict_mean', float('nan'))
    base_c = base.get(key, {}).get('conflict_mean', float('nan'))
    delta = sft_c - base_c if not (sft_c != sft_c or base_c != base_c) else float('nan')

    flag = ''
    if sft_c < -0.1 or base_c < -0.1:
        flag = 'INTERFERENCE'
    elif abs(delta) > 0.1 and delta == delta:
        flag = 'DIVERGED'

    print(f'{l:>6} | {sft_c:>+16.4f} | {base_c:>+14.4f} | {delta:>+8.4f} | {flag:>12}')

# Summary
print()
sft_late = [sft.get(str(l), {}).get('conflict_mean', 0) for l in range(19, 28)]
base_late = [base.get(str(l), {}).get('conflict_mean', 0) for l in range(19, 28)]
import numpy as np
print(f'Late layers (L19-27) mean conflict:')
print(f'  SFT v2: {np.mean(sft_late):+.4f}')
print(f'  Base:   {np.mean(base_late):+.4f}')
print(f'  Delta:  {np.mean(sft_late) - np.mean(base_late):+.4f}')
"

echo ""
echo "=========================================="
echo "Gradient Conflict Analysis Complete"
echo "=========================================="
echo "End: $(date)"
echo "SFT v2 results: $SFT_OUTPUT"
echo "Base results:   $BASE_OUTPUT"
echo "=========================================="
