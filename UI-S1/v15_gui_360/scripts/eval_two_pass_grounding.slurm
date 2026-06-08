#!/bin/bash
#SBATCH --job-name=two_pass_grnd
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/two_pass_grounding_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/two_pass_grounding_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Two-Pass Grounding Experiment
#
# Tests whether decomposing action prediction into:
#   Pass 1: Element detection (list all UI elements)
#   Pass 2: Action selection (choose element + generate action)
# improves click accuracy vs one-pass direct prediction.
#
# Evaluates 100 click steps from test set.
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CKPT_PATH="$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_balanced_full_sft/checkpoint-272"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/v15_gui_360/outputs/preliminary_tests/two_pass_grounding"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=========================================="
echo "Two-Pass Grounding Experiment"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"

python "$PROJECT_ROOT/v15_gui_360/eval_two_pass_grounding.py" \
    --model_path "$CKPT_PATH" \
    --test_data "$TEST_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps 100

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
