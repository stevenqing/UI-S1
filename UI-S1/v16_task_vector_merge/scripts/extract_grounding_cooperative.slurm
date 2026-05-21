#!/bin/bash
#SBATCH --job-name=v16_grd
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v16_task_vector_merge/scripts/logs/extract_grounding_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v16_task_vector_merge/scripts/logs/extract_grounding_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:1

# ============================================
# Step 1: Extract LoRA from gui_grounding via SVD (PEFT format)
# Step 2: Convert PEFT LoRA to cooperative V13 format
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
GROUNDING_MODEL="$PROJECT_ROOT/checkpoints/gui_grounding"
PEFT_OUTPUT="$PROJECT_ROOT/checkpoints/v16_peft_grounding_r128"
COOP_OUTPUT="$PROJECT_ROOT/checkpoints/v16_peft_grounding_cooperative_r128"

echo "=========================================="
echo "V16: Extract Grounding LoRA + Convert to Cooperative"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start:  $(date)"
echo "=========================================="

# Step 1: SVD extract gui_grounding → PEFT standard LoRA
echo ""
echo "=== Step 1: SVD Extract gui_grounding → PEFT LoRA (r=128) ==="
if [ -f "$PEFT_OUTPUT/adapter_model.safetensors" ]; then
    echo "  Already exists: $PEFT_OUTPUT"
else
    python Option-incentivized-MoE/extract_fullsft_to_lora.py \
        --sft_model "$GROUNDING_MODEL" \
        --base_model "$BASE_MODEL" \
        --output "$PEFT_OUTPUT" \
        --rank 128 \
        --alpha 256
fi

# Step 2: Convert PEFT LoRA → Cooperative V13
echo ""
echo "=== Step 2: Convert PEFT → Cooperative (noise=0.1, T=2) ==="
if [ -f "$COOP_OUTPUT/cooperative_config.json" ]; then
    echo "  Already exists: $COOP_OUTPUT"
else
    python v15_gui_360/convert_peft_to_cooperative.py \
        --peft_checkpoint "$PEFT_OUTPUT" \
        --output "$COOP_OUTPUT" \
        --num_comm_rounds 2 \
        --noise_scale 0.1
fi

echo ""
echo "=========================================="
echo "Done: $(date)"
echo ""
echo "Grounding PEFT LoRA:        $PEFT_OUTPUT"
echo "Grounding Cooperative LoRA: $COOP_OUTPUT"
echo "=========================================="
