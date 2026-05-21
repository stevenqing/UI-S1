#!/bin/bash
#SBATCH --job-name=v63t_e1_ap
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_v6_3t_epoch1_ap_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_v6_3t_epoch1_ap_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Proper-routing eval for v6.3 cooperative LoRA
# Checkpoint: epoch-1 (~762 steps, r=256, tanh gate, eff_bs=128)
# Only action_prediction, 1 node, 4 GPUs, 4 shards
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="checkpoints/Qwen2.5-VL-7B-Instruct"
COOP_CKPT="train_GUI_360/llamafactory/output/cooperative_v6_3_comm_thought/epoch-1"
RESULTS_BASE="train_GUI_360/GUI-360-eval/results/cooperative_v6_3_thought_epoch1_proper"
LOGDIR="$PROJECT_DIR/scripts/exp_cooperative/logs"
NUM_SHARDS=4
BATCH_SIZE=32
EVAL_TYPE="action_prediction"

echo "=========================================="
echo "Cooperative LoRA v6.3 (thought, tanh, eff_bs=128, epoch-1) Proper Routing Eval"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "=========================================="

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

cd "$PROJECT_DIR"

OUT_DIR="$RESULTS_BASE/${EVAL_TYPE}"
mkdir -p "$OUT_DIR"

echo "Launching ${NUM_SHARDS} shards for ${EVAL_TYPE}"
PIDS=()
for shard in $(seq 0 $((NUM_SHARDS-1))); do
    LOG="$LOGDIR/v6_3t_epoch1_${EVAL_TYPE}_shard${shard}_${SLURM_JOB_ID}.log"
    echo "  launching shard=${shard} gpu=${shard} -> ${LOG}"
    python -u evaluation/eval_cooperative_batch.py \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$COOP_CKPT" \
        --eval_data_root datasets/GUI-360/test \
        --eval_type "$EVAL_TYPE" \
        --output_dir "$OUT_DIR" \
        --gpu_id $shard \
        --shard_id $shard \
        --num_shards $NUM_SHARDS \
        --batch_size $BATCH_SIZE \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
    wait $pid || echo "shard PID=$pid exited non-zero"
done

echo "=========================================="
echo "All shards done: $(date)"
echo "=========================================="

for shard in $(seq 0 $((NUM_SHARDS-1))); do
    echo "--- ${EVAL_TYPE} shard${shard} last 5 lines ---"
    tail -5 "$LOGDIR/v6_3t_epoch1_${EVAL_TYPE}_shard${shard}_${SLURM_JOB_ID}.log" 2>/dev/null || true
done

echo ""
echo "=== ${EVAL_TYPE} aggregate summary ==="
python - << PYEOF
import json, glob, os
out_dir = "$OUT_DIR"
shards = sorted(glob.glob(os.path.join(out_dir, "summary_shard*.json")))
total, success = 0, 0
for s in shards:
    d = json.load(open(s))
    total += d.get("total_samples", 0)
    success += d.get("success_count", 0)
    print(f"  {os.path.basename(s)}: total={d.get('total_samples')}, success={d.get('success_count')}, rate={d.get('success_rate', 0):.2f}%")
if total:
    print(f"\n=== AGGREGATE: total={total}, success={success}, rate={100*success/total:.2f}% ===")
PYEOF

echo ""
echo "Done: $(date)"
echo "=========================================="
