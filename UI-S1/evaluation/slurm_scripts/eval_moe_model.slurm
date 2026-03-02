#!/bin/bash
#SBATCH --job-name=eval_moe
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/results/eval_moe_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/results/eval_moe_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ========================================
# MoE Model Evaluation Script
# ========================================
# Evaluates MoE checkpoints from gui_traj_grpo_moe
#
# Usage:
#   sbatch eval_moe_model.slurm [CHECKPOINT_NAME] [STEP]
#
# Example:
#   sbatch eval_moe_model.slurm qwenvl_uis1_MoE_4experts_r16_balance0.1_gamma0.5 20
#
# ========================================

# Activate environment
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

# Set CUDA library path
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

# Set PYTHONPATH to prioritize local verl package (includes MoE module)
export PYTHONPATH="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1:$PYTHONPATH"

# Paths
BASE_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="${BASE_DIR}/evaluation"
DATASET_PATH="${EVAL_DIR}/dataset/android_control_evaluation_std.jsonl"
OUTPUT_DIR="${EVAL_DIR}/results"
BASE_MODEL="${BASE_DIR}/checkpoints/Qwen2.5-VL-7B-Instruct"

# Default checkpoint configuration
CHECKPOINT_NAME="${1:-qwenvl_uis1_MoE_4experts_r16_balance0.1_gamma0.5}"
STEP="${2:-20}"

CHECKPOINT_DIR="${BASE_DIR}/checkpoints/gui_traj_grpo_moe/${CHECKPOINT_NAME}/global_step_${STEP}"
MODEL_NAME="MoE_${CHECKPOINT_NAME}_step${STEP}"

echo "=========================================="
echo "Starting MoE Model Evaluation"
echo "Checkpoint: ${CHECKPOINT_NAME}"
echo "Step: ${STEP}"
echo "Checkpoint Dir: ${CHECKPOINT_DIR}"
echo "Base Model: ${BASE_MODEL}"
echo "Start time: $(date)"
echo "=========================================="

# Check if MoE checkpoint exists
if [ ! -f "${CHECKPOINT_DIR}/actor/moe/moe_config.json" ]; then
    echo "ERROR: MoE checkpoint not found at ${CHECKPOINT_DIR}"
    echo "Available checkpoints:"
    ls -la "${BASE_DIR}/checkpoints/gui_traj_grpo_moe/${CHECKPOINT_NAME}/" 2>/dev/null || echo "Checkpoint directory not found"
    exit 1
fi

# Run MoE evaluation directly (no vLLM server needed)
echo ""
echo "Running MoE model evaluation..."
echo "=========================================="

cd $BASE_DIR

python $EVAL_DIR/eval_moe_model.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --base_model "$BASE_MODEL" \
    --jsonl_file "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --n_history_image_limit 2

EVAL_EXIT_CODE=$?

echo "=========================================="
echo "Evaluation completed at $(date)"
echo "Exit code: $EVAL_EXIT_CODE"
echo "Results saved to: $OUTPUT_DIR/${MODEL_NAME}.jsonl"
echo "=========================================="

# Analyze results
if [ -f "$OUTPUT_DIR/${MODEL_NAME}.jsonl" ]; then
    echo ""
    echo "Results Summary:"
    echo "=========================================="
    python -c "
import json
results = []
with open('$OUTPUT_DIR/${MODEL_NAME}.jsonl', 'r') as f:
    for line in f:
        results.append(json.loads(line))
success = sum(1 for r in results if r.get('task_success', False))
total = len(results)
print(f'Total tasks: {total}')
print(f'Successful tasks: {success}')
print(f'Success rate: {success/total*100:.2f}%' if total > 0 else 'N/A')
"
fi

exit $EVAL_EXIT_CODE
