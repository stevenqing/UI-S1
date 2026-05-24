#!/bin/bash
#SBATCH --job-name=attn_align
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/attention_analysis_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/attention_analysis_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Attention Alignment + Crop Verification Analysis
#
# Loads model directly (not vLLM) with eager attention.
# For 100 partial click steps (from BoN data):
#   1. Attention analysis: where does the model attend when
#      generating coordinate tokens? Does CoM align with prediction?
#   2. Crop verify: crop region around predicted coord, ask model
#      "what element is here?", check consistency with instruction.
#
# Tests whether attention alignment or crop-verify can serve
# as a verifier to close the Oracle gap (58.8% → 71.7%).
# ============================================

set -e

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CKPT_PATH="$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_balanced_full_sft/checkpoint-272"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
BON_DATA="$PROJECT_ROOT/v15_gui_360/outputs/preliminary_tests/bon_metrics/bon_raw_*.json"
OUTPUT_DIR="$PROJECT_ROOT/v15_gui_360/outputs/preliminary_tests/attention_analysis"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Attention Alignment + Crop Verify Analysis"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"

python "$PROJECT_ROOT/v15_gui_360/analyze_attention_alignment.py" \
    --model_path "$CKPT_PATH" \
    --test_data "$TEST_DATA" \
    --bon_data "$BON_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples 100

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
