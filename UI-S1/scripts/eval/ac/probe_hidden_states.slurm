#!/bin/bash
#SBATCH --job-name=probe_hs
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/probe_hs_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/probe_hs_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:2

# ============================================
# Q1: Probing Experiment - Task Progress State
# Phase 1: Extract hidden states from oracle condition
# Phase 2: Train linear probes
# ============================================

set -e  # Stop on error so Phase 2 doesn't run if Phase 1 fails

source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
MODEL_PATH="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
DATASET_PATH="$PROJECT_ROOT/datasets/android_control_evaluation_std.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/outputs/probe_hidden_states/Qwen2.5-VL-7B"

echo "=========================================="
echo "Q1: Probing Experiment"
echo "Model: $MODEL_PATH"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p $OUTPUT_DIR

cd $PROJECT_ROOT/scripts/eval/ac

# Phase 1: Extract hidden states
echo "Phase 1: Extracting hidden states..."
python probe_hidden_states.py \
    --mode extract \
    --model_path $MODEL_PATH \
    --jsonl_file $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --max_episodes 200 \
    --max_steps_per_episode 20 \
    --max_samples 2000

echo "Phase 1 completed at $(date)"

# Phase 2: Train probes
echo ""
echo "Phase 2: Training linear probes..."
pip install scikit-learn -q 2>/dev/null

python probe_hidden_states.py \
    --mode probe \
    --output_dir $OUTPUT_DIR

echo ""
echo "=========================================="
echo "Probing experiment completed at $(date)"
echo "Results: $OUTPUT_DIR/probe_results.json"
echo "=========================================="
