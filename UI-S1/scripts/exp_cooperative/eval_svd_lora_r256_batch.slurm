#!/bin/bash
#SBATCH --job-name=eval_svd256
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_svd256_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_svd256_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
NUM_SHARDS=4
BATCH_SIZE=32

echo "=========================================="
echo "SVD LoRA r=256 Eval (same pipeline as coop)"
echo "Job ID: ${SLURM_JOB_ID} | Start: $(date)"
echo "=========================================="

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

cd "$PROJECT_DIR"

BASE_MODEL="checkpoints/Qwen2.5-VL-7B-Instruct"
PEFT_CKPT="train_GUI_360/moe_rl/extracted_lora_fullsft_r256"
RESULTS_BASE="train_GUI_360/GUI-360-eval/results/svd_lora_r256_same_pipeline"
LOGDIR="$PROJECT_DIR/scripts/exp_cooperative/logs"

echo "=== Action Prediction (batch_size=$BATCH_SIZE) ==="
PIDS=()
for shard in $(seq 0 $((NUM_SHARDS-1))); do
    python -u evaluation/eval_peft_batch.py \
        --base_model "$BASE_MODEL" \
        --peft_checkpoint "$PEFT_CKPT" \
        --eval_data_root datasets/GUI-360/test \
        --eval_type action_prediction \
        --output_dir "$RESULTS_BASE/action_prediction" \
        --gpu_id $shard \
        --shard_id $shard \
        --num_shards $NUM_SHARDS \
        --batch_size $BATCH_SIZE \
        > "$LOGDIR/svd256_ap_shard${shard}_${SLURM_JOB_ID}.log" 2>&1 &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do wait $pid || true; done
echo "Action prediction done: $(date)"
for shard in $(seq 0 $((NUM_SHARDS-1))); do
    tail -3 "$LOGDIR/svd256_ap_shard${shard}_${SLURM_JOB_ID}.log" 2>/dev/null
done
echo "All done: $(date)"
