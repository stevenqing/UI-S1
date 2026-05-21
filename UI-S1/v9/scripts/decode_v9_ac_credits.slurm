#!/bin/bash
#SBATCH --job-name=v9_decode_ac
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v9/scripts/logs/decode_v9_ac_%A_%a.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v9/scripts/logs/decode_v9_ac_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:1
#SBATCH --array=0-7

# ============================================
# v9 step-1: Offline credit extraction over AC train set.
#
# Decodes ac_train_thought.jsonl with the v6.5 AC ep4 checkpoint, parses
# pred_action, compares with GT, writes (w_v, w_a, r_g, r_a, ...) per sample
# to v9/credits/v6_5_ac_ep4/credit_cache_shard{N}.jsonl.
#
# Sharded across 8 GPU jobs (~6482 samples / 8 ≈ 810 per shard).
# After all shards finish, run v9/scripts/merge_credit_shards.sh to produce
# v9/credits/v6_5_ac_ep4/credit_cache.jsonl.
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CONDA_ENV="ui-s1"

BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
COOP_CKPT="$PROJECT_DIR/train_GUI_360/llamafactory/output/cooperative_v6_5_ac/epoch-4"
TRAIN_JSONL="$PROJECT_DIR/datasets/cooperative_thought_ac/ac_train_thought.jsonl"
OUT_DIR="$PROJECT_DIR/v9/credits/v6_5_ac_ep4"

mkdir -p "$OUT_DIR" "$PROJECT_DIR/v9/scripts/logs"

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/${CONDA_ENV}/bin:$PATH"
export CONDA_DEFAULT_ENV=$CONDA_ENV
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1

cd "$PROJECT_DIR"

NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-8}
SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "=========================================="
echo "v9 decode_credits — shard $SHARD_ID / $NUM_SHARDS"
echo "Base:  $BASE_MODEL"
echo "Coop:  $COOP_CKPT"
echo "Train: $TRAIN_JSONL"
echo "Out:   $OUT_DIR"
echo "Start: $(date)"
echo "=========================================="

python v9/decode_credits.py \
    --base_model "$BASE_MODEL" \
    --coop_checkpoint "$COOP_CKPT" \
    --train_jsonl "$TRAIN_JSONL" \
    --out_dir "$OUT_DIR" \
    --shard_id "$SHARD_ID" \
    --num_shards "$NUM_SHARDS" \
    --max_new_tokens 512 \
    --resume

echo "=========================================="
echo "Done shard $SHARD_ID: $(date)"
echo "=========================================="
