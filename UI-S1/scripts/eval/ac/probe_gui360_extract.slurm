#!/bin/bash
#SBATCH --job-name=probe_g360
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/probe_gui360_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/probe_gui360_%j.err
#SBATCH --time=6:00:00
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
DATASET_PATH="$PROJECT_ROOT/datasets/GUI-360/rl_data/gui360_train.jsonl"
ORACLE_DIR="$PROJECT_ROOT/outputs/probe_hidden_states_gui360/Qwen2.5-VL-7B"
BASELINE_DIR="$PROJECT_ROOT/outputs/probe_hidden_states_gui360_baseline/Qwen2.5-VL-7B"

cd "$PROJECT_ROOT/scripts/eval/ac"
mkdir -p logs

echo "===== Phase 1: Extract Oracle Hidden States ====="
python probe_gui360_extract.py \
    --mode extract_oracle \
    --model_path "$MODEL_PATH" \
    --jsonl_file "$DATASET_PATH" \
    --output_dir "$ORACLE_DIR" \
    --max_episodes 200 --max_steps_per_episode 20 --max_samples 2000

echo "===== Phase 2: Extract Baseline Hidden States ====="
python probe_gui360_extract.py \
    --mode extract_baseline \
    --model_path "$MODEL_PATH" \
    --jsonl_file "$DATASET_PATH" \
    --output_dir "$BASELINE_DIR" \
    --max_episodes 200 --max_steps_per_episode 20 --max_samples 2000

echo "===== Phase 3: Train Probes (Oracle) ====="
pip install scikit-learn -q 2>/dev/null
python probe_gui360_extract.py \
    --mode probe \
    --output_dir "$ORACLE_DIR"

echo "===== Phase 4: Train Probes (Baseline) ====="
python probe_gui360_extract.py \
    --mode probe \
    --output_dir "$BASELINE_DIR"

echo "===== Phase 5: Cross-Apply Oracle → Baseline ====="
python probe_gui360_extract.py \
    --mode cross_apply \
    --output_dir "$BASELINE_DIR" \
    --oracle_dir "$ORACLE_DIR"

echo "===== All phases complete ====="
