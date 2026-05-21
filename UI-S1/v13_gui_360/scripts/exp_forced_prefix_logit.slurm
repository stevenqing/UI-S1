#!/bin/bash
#SBATCH --job-name=forced_logit
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/forced_logit_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/forced_logit_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

set -e
PROJECT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
OUTPUT_DIR="$PROJECT/v13_gui_360/outputs/forced_prefix_logit"

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo "Forced-Prefix + Logit Gap — 968 eps, 4 GPUs | Job ${SLURM_JOB_ID} | $(date)"
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT"

for SHARD in 0 1 2 3; do
    python v13_gui_360/exp_forced_prefix_and_logit_gap.py \
        --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
        --coop_checkpoint checkpoints/v13_gui360_rl/epoch-3/cooperative \
        --test_data v13_gui_360/data/gui360_test_968.jsonl \
        --gpu $SHARD \
        --shard_id $SHARD \
        --num_shards 4 \
        --output_dir "$OUTPUT_DIR" \
        &
done

echo "Waiting for all 4 shards..."
wait
echo "All shards done: $(date)"

echo ""
echo "=========================================="
echo "Merging results..."
echo "=========================================="
python v13_gui_360/exp_forced_prefix_and_logit_gap.py --merge "$OUTPUT_DIR"
echo "Done: $(date)"
