#!/bin/bash
#SBATCH --job-name=grad_proj_ba
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/grad_conflict_proj_bindaux_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/grad_conflict_proj_bindaux_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:1

# ============================================
# Step 1b: Projection-Level Gradient Conflict
# bind_aux model (after L_bind training)
# Compare with SFT v2 and Base to see if
# L_bind changed the conflict structure
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_SCRIPT="$PROJECT_DIR/evaluation/gradient_conflict_projection.py"
PARQUET_FILE="$PROJECT_DIR/train_GUI_360/data/gui360_eval_sft.parquet"
IMAGE_BASE="$PROJECT_DIR"

# bind_aux final merged model (LoRA already merged into base weights)
BIND_AUX_PATH="$PROJECT_DIR/train_GUI_360/llamafactory/output/bind_aux_r32/final_merged"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "Step 1b: Projection-Level Gradient Conflict (bind_aux)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Model: $BIND_AUX_PATH"
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

# Check model exists
if [ ! -d "$BIND_AUX_PATH" ]; then
    echo "ERROR: bind_aux model not found at $BIND_AUX_PATH"
    echo "Training may still be running. Check job 3451004."
    exit 1
fi

OUTPUT_DIR="$PROJECT_DIR/evaluation/results/grad_conflict_projection_bindaux_${TIMESTAMP}"

echo "Output: $OUTPUT_DIR"

python "$EVAL_SCRIPT" \
    --model_path "$BIND_AUX_PATH" \
    --parquet_file "$PARQUET_FILE" \
    --image_base "$IMAGE_BASE" \
    --n_samples 200 --seed 42 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "bind_aux projection analysis complete"
echo "End: $(date)"
echo "Results: $OUTPUT_DIR"
echo "=========================================="
