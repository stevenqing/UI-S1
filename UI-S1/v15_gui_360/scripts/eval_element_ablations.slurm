#!/bin/bash
#SBATCH --job-name=elem_ablate
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/elem_ablations_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/elem_ablations_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Element Ablation Experiments
#
# 4 modes × 200 steps = 800 inferences
# Tests what is the real bottleneck:
#   1. one_pass (baseline)
#   2. oracle_top5 (short element list)
#   3. oracle_subgoal (tell model WHAT to click)
#   4. subgoal_cot (force step-by-step reasoning)
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CKPT_PATH="$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_balanced_full_sft/checkpoint-272"
AP_DATA="$PROJECT_ROOT/datasets/GUI-360/processed_data/action_prediction_train_resize/training_data.json"
SP_DATA="$PROJECT_ROOT/datasets/GUI-360/processed_data/screen_parsing_train_resize/training_data.json"
IMAGE_DIR="$PROJECT_ROOT/datasets/GUI-360/processed_data/action_prediction_train_resize"
OUTPUT_DIR="$PROJECT_ROOT/v15_gui_360/outputs/preliminary_tests/element_ablations"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=========================================="
echo "Element Ablation Experiments"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"

python "$PROJECT_ROOT/v15_gui_360/eval_element_ablations.py" \
    --model_path "$CKPT_PATH" \
    --ap_data "$AP_DATA" \
    --sp_data "$SP_DATA" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps 200

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
