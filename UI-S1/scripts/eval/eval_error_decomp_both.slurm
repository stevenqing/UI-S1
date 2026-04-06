#!/bin/bash
#SBATCH --job-name=ed_both
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/ed_both_%j.out
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/ed_both_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:1

# Error decomposition for bind_aux_v2 + lora_baseline (sequential, 1 GPU)

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
PARQUET="$PROJECT_DIR/train_GUI_360/data/gui360_test_sft_eval_format_with_bbox.parquet"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export PYTHONUNBUFFERED=1

export TMPDIR=/tmp/${SLURM_JOB_ID}
mkdir -p $TMPDIR 2>/dev/null || true
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor
export TORCH_HOME=$TMPDIR/torch
export XDG_CACHE_HOME=$TMPDIR/cache
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TORCH_HOME $XDG_CACHE_HOME 2>/dev/null || true

cd "$PROJECT_DIR"

echo "============================================================"
echo "Error Decomposition: bind_aux_v2 + baseline"
echo "Job ${SLURM_JOB_ID} | $(hostname) | $(date)"
echo "============================================================"

# ── bind_aux_v2 ──
echo ""
echo "=== bind_aux_v2_r32 ==="
ED1="$PROJECT_DIR/evaluation/results/error_decomp_bind_aux_v2_r32_${TIMESTAMP}"
mkdir -p "$ED1"

python -u evaluation/error_decomposition.py collect \
    --model_path "$BASE_MODEL" \
    --lora_path "$PROJECT_DIR/train_GUI_360/llamafactory/output/bind_aux_v2_r32/final" \
    --parquet_file "$PARQUET" \
    --n_samples 500 \
    --output_dir "$ED1"

python -u evaluation/error_decomposition.py analyze \
    --predictions "$ED1/predictions.json" \
    --output_dir "$ED1"

echo "bind_aux_v2 done at $(date)"

# ── lora_baseline ──
echo ""
echo "=== lora_baseline_r32 ==="
ED2="$PROJECT_DIR/evaluation/results/error_decomp_lora_baseline_r32_${TIMESTAMP}"
mkdir -p "$ED2"

python -u evaluation/error_decomposition.py collect \
    --model_path "$BASE_MODEL" \
    --lora_path "$PROJECT_DIR/train_GUI_360/llamafactory/output/lora_baseline_r32/final" \
    --parquet_file "$PARQUET" \
    --n_samples 500 \
    --output_dir "$ED2"

python -u evaluation/error_decomposition.py analyze \
    --predictions "$ED2/predictions.json" \
    --output_dir "$ED2"

echo "baseline done at $(date)"

echo ""
echo "============================================================"
echo "All error decomposition complete at $(date)"
echo "bind_aux_v2: $ED1"
echo "baseline: $ED2"
echo "============================================================"
