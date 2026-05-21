#!/bin/bash
#SBATCH --job-name=gate_perturb
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/gate_perturb_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/gate_perturb_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

set -e
PROJECT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
OUTPUT_DIR="$PROJECT/v13_gui_360/outputs/gate_perturbation"

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo "Gate Perturbation — Full 968, 4 GPUs | Job ${SLURM_JOB_ID} | $(date)"
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT"

for SHARD in 0 1 2 3; do
    python v13_gui_360/exp_gate_perturbation.py \
        --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
        --coop_checkpoint checkpoints/v13_gui360_rl/epoch-3/cooperative \
        --test_data v13_gui_360/data/gui360_test_968.jsonl \
        --max_episodes 968 \
        --deltas="-0.5,-0.2,-0.1,0,0.1,0.2,0.5" \
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
python v13_gui_360/exp_gate_perturbation.py --merge "$OUTPUT_DIR"

echo "Done: $(date)"
