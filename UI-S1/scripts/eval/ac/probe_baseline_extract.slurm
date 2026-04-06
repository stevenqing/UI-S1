#!/bin/bash
#SBATCH --job-name=probe_bl
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/probe_baseline_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/probe_baseline_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:2

set -e

source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
MODEL_PATH="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
DATASET_PATH="$PROJECT_ROOT/datasets/android_control_evaluation_std.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/outputs/probe_hidden_states_baseline/Qwen2.5-VL-7B"
ORACLE_DIR="$PROJECT_ROOT/outputs/probe_hidden_states/Qwen2.5-VL-7B"

cd "$PROJECT_ROOT/scripts/eval/ac"

echo "===== Phase 1: Extract baseline hidden states ====="
python probe_baseline_extract.py \
    --mode extract \
    --model_path $MODEL_PATH \
    --jsonl_file $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --max_episodes 200 \
    --max_steps_per_episode 20 \
    --max_samples 2000

echo "===== Phase 2: Train probes on baseline data ====="
pip install scikit-learn -q
python probe_baseline_extract.py \
    --mode probe \
    --output_dir $OUTPUT_DIR

echo "===== Phase 3: Cross-apply oracle probes to baseline data ====="
python probe_baseline_extract.py \
    --mode cross_apply \
    --output_dir $OUTPUT_DIR \
    --oracle_dir $ORACLE_DIR

echo "===== All phases complete ====="
