#!/bin/bash
#SBATCH --job-name=merge_ba
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/merge_ba_%j.out
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/merge_ba_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:1

# ============================================================
# Merge all 3 bind_aux_r32 LoRA checkpoints using llamafactory-cli export
# Same approach as eval_lora_v4_ckpt250.slurm etc.
# ============================================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
BIND_DIR="$PROJECT_DIR/train_GUI_360/llamafactory/output/bind_aux_r32"

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export PYTHONUNBUFFERED=1

if [ -f "$PROJECT_DIR/train/env_config.sh" ]; then
    source "$PROJECT_DIR/train/env_config.sh"
fi

echo "Merge all bind_aux_r32 checkpoints | Job ${SLURM_JOB_ID} | $(hostname) | $(date)"

for CKPT in checkpoint-400 checkpoint-600 final; do
    LORA="${BIND_DIR}/${CKPT}"
    MERGED="${BIND_DIR}/${CKPT}_merged"

    if [ -f "$MERGED/model.safetensors.index.json" ]; then
        echo "[$CKPT] Already merged, skip."
        continue
    fi

    echo ""
    echo "[$CKPT] Merging at $(date)..."
    rm -rf "$MERGED" 2>/dev/null

    llamafactory-cli export \
        --model_name_or_path $BASE_MODEL \
        --adapter_name_or_path $LORA \
        --template qwen2_vl \
        --finetuning_type lora \
        --export_dir $MERGED \
        --export_size 5 \
        --export_legacy_format False

    echo "[$CKPT] Merge done at $(date)"
done

echo ""
echo "All merges complete at $(date)"
