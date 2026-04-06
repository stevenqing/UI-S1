#!/bin/bash
#SBATCH --job-name=grad_proj
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/grad_conflict_proj_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/grad_conflict_proj_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:1

# ============================================
# Step 1b: Projection-Level Gradient Conflict
# Sequential: SFT v2 → Base model
# 200 samples each, 1 GPU, ~2h per model
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_SCRIPT="$PROJECT_DIR/evaluation/gradient_conflict_projection.py"
PARQUET_FILE="$PROJECT_DIR/train_GUI_360/data/gui360_eval_sft.parquet"
IMAGE_BASE="$PROJECT_DIR"

SFT_V2_PATH="$PROJECT_DIR/train_GUI_360/llamafactory/output/gui360_full_sft_v2"
BASE_PATH="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "Step 1b: Projection-Level Gradient Conflict"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Models: SFT v2, Base"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

export TMPDIR=/tmp/${SLURM_JOB_ID}
mkdir -p $TMPDIR
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor
export TORCH_HOME=$TMPDIR/torch
export XDG_CACHE_HOME=$TMPDIR/cache
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TORCH_HOME $XDG_CACHE_HOME

mkdir -p "$PROJECT_DIR/scripts/eval/logs"

# ============================================
# Run 1: SFT v2
# ============================================
SFT_OUTPUT="$PROJECT_DIR/evaluation/results/grad_conflict_projection_sft_v2_${TIMESTAMP}"

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
BASE_OUTPUT="$PROJECT_DIR/evaluation/results/grad_conflict_projection_base_${TIMESTAMP}"

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
echo "Cross-Model Projection-Level Comparison"
echo "=========================================="

python3 -c "
import json, numpy as np

MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
GROUPS = {
    'attn_routing': ['q_proj', 'k_proj'],
    'attn_content': ['v_proj', 'o_proj'],
    'ffn': ['gate_proj', 'up_proj', 'down_proj'],
}

sft = json.load(open('$SFT_OUTPUT/per_module_conflict.json'))
base = json.load(open('$BASE_OUTPUT/per_module_conflict.json'))

print(f\"{'':>6} | {'SFT v2':>50} | {'Base':>50}\")
print(f\"{'Layer':>6} | \", end='')
for _ in range(2):
    for m in ['q', 'k', 'v', 'o', 'gate', 'up', 'down']:
        print(f'{m:>7}', end='')
    print(' | ', end='')
print()
print('-' * 130)

for l in range(19, 28):
    print(f'{l:>6} | ', end='')
    for data in [sft, base]:
        for mod in MODULES:
            key = f'{l}_{mod}'
            val = data.get(key, {}).get('conflict_mean', float('nan'))
            print(f'{val:>+7.3f}', end='')
        print(' | ', end='')
    print()

# Group summary
print()
print('Late layers (L19-27) group means:')
print(f\"{'Group':>14} | {'SFT v2':>10} | {'Base':>10} | {'Delta':>10}\")
print('-' * 55)
for gname, gmods in GROUPS.items():
    sft_vals = [sft.get(f'{l}_{m}', {}).get('conflict_mean', float('nan'))
                for l in range(19, 28) for m in gmods]
    base_vals = [base.get(f'{l}_{m}', {}).get('conflict_mean', float('nan'))
                 for l in range(19, 28) for m in gmods]
    sm = np.nanmean(sft_vals)
    bm = np.nanmean(base_vals)
    print(f'{gname:>14} | {sm:>+10.4f} | {bm:>+10.4f} | {sm-bm:>+10.4f}')
"

echo ""
echo "=========================================="
echo "Step 1b Complete"
echo "End: $(date)"
echo "SFT v2: $SFT_OUTPUT"
echo "Base:   $BASE_OUTPUT"
echo "=========================================="
