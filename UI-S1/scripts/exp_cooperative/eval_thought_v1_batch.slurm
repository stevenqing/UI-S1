#!/bin/bash
#SBATCH --job-name=eval_batch
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_batch_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_batch_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Batched multi-GPU cooperative LoRA eval
# 4 independent processes, one per GPU
# batch_size=32 per GPU for GPU-parallel speedup
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
NUM_SHARDS=4
BATCH_SIZE=32

echo "=========================================="
echo "Batched Cooperative LoRA Eval (4 GPU, bs=$BATCH_SIZE)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

cd "$PROJECT_DIR"

BASE_MODEL="checkpoints/Qwen2.5-VL-7B-Instruct"
COOP_CKPT="train_GUI_360/llamafactory/output/cooperative_thought_v1/final"
RESULTS_BASE="train_GUI_360/GUI-360-eval/results/cooperative_thought_v1_batch"

LOGDIR="$PROJECT_DIR/scripts/exp_cooperative/logs"

# ── Action Prediction (4 parallel processes) ──
echo ""
echo "=== Action Prediction (batch_size=$BATCH_SIZE) ==="
PIDS=()
for shard in $(seq 0 $((NUM_SHARDS-1))); do
    echo "Launching shard $shard on GPU $shard..."
    python -u evaluation/eval_cooperative_batch.py \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$COOP_CKPT" \
        --eval_data_root datasets/GUI-360/test \
        --eval_type action_prediction \
        --output_dir "$RESULTS_BASE/action_prediction" \
        --gpu_id $shard \
        --shard_id $shard \
        --num_shards $NUM_SHARDS \
        --batch_size $BATCH_SIZE \
        > "$LOGDIR/batch_ap_shard${shard}_${SLURM_JOB_ID}.log" 2>&1 &
    PIDS+=($!)
done

echo "Waiting for action_prediction..."
for pid in "${PIDS[@]}"; do wait $pid || true; done
echo "Action prediction done: $(date)"

for shard in $(seq 0 $((NUM_SHARDS-1))); do
    echo "--- Shard $shard ---"
    tail -3 "$LOGDIR/batch_ap_shard${shard}_${SLURM_JOB_ID}.log" 2>/dev/null
done

# ── Grounding (4 parallel processes) ──
echo ""
echo "=== Grounding (batch_size=$BATCH_SIZE) ==="
PIDS=()
for shard in $(seq 0 $((NUM_SHARDS-1))); do
    echo "Launching shard $shard on GPU $shard..."
    python -u evaluation/eval_cooperative_batch.py \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$COOP_CKPT" \
        --eval_data_root datasets/GUI-360/test \
        --eval_type grounding \
        --output_dir "$RESULTS_BASE/grounding" \
        --gpu_id $shard \
        --shard_id $shard \
        --num_shards $NUM_SHARDS \
        --batch_size $BATCH_SIZE \
        > "$LOGDIR/batch_gr_shard${shard}_${SLURM_JOB_ID}.log" 2>&1 &
    PIDS+=($!)
done

echo "Waiting for grounding..."
for pid in "${PIDS[@]}"; do wait $pid || true; done
echo "Grounding done: $(date)"

for shard in $(seq 0 $((NUM_SHARDS-1))); do
    echo "--- Shard $shard ---"
    tail -3 "$LOGDIR/batch_gr_shard${shard}_${SLURM_JOB_ID}.log" 2>/dev/null
done

# ── Merge results ──
echo ""
echo "=== Merging results ==="
python -c "
import json, glob, os

for eval_type in ['action_prediction', 'grounding']:
    rdir = '$RESULTS_BASE/' + eval_type
    files = sorted(glob.glob(os.path.join(rdir, 'results_shard*_*.json')))
    if not files:
        print(f'No shard files for {eval_type}')
        continue

    all_results = []
    for f in files:
        with open(f) as fh:
            all_results.extend(json.load(fh))

    total = len(all_results)
    success = sum(1 for r in all_results if r.get('success'))
    rate = 100 * success / total if total > 0 else 0

    merged = {'total': total, 'success': success, 'rate': rate, 'eval_type': eval_type}
    with open(os.path.join(rdir, 'merged_summary.json'), 'w') as fh:
        json.dump(merged, fh, indent=2)

    print(f'{eval_type}: {success}/{total} = {rate:.1f}%')
"

echo "=========================================="
echo "All done: $(date)"
echo "=========================================="
