#!/bin/bash
#SBATCH --job-name=eval_v2_batch
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_v2_batch_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_v2_batch_%j.err
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
echo "Cooperative LoRA v2 Eval (r=32, 7 modules, 161M params)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start: $(date)"

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

cd "$PROJECT_DIR"

BASE_MODEL="checkpoints/Qwen2.5-VL-7B-Instruct"
COOP_CKPT="train_GUI_360/llamafactory/output/cooperative_thought_v2/final"
RESULTS_BASE="train_GUI_360/GUI-360-eval/results/cooperative_thought_v2_batch"

LOGDIR="$PROJECT_DIR/scripts/exp_cooperative/logs"

# ── Action Prediction ──
echo "=== Action Prediction (batch_size=$BATCH_SIZE) ==="
PIDS=()
for shard in $(seq 0 $((NUM_SHARDS-1))); do
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
        > "$LOGDIR/v2_batch_ap_shard${shard}_${SLURM_JOB_ID}.log" 2>&1 &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do wait $pid || true; done
echo "Action prediction done: $(date)"
for shard in $(seq 0 $((NUM_SHARDS-1))); do
    echo "--- Shard $shard ---"
    tail -3 "$LOGDIR/v2_batch_ap_shard${shard}_${SLURM_JOB_ID}.log" 2>/dev/null
done

# ── Grounding ──
echo "=== Grounding (batch_size=$BATCH_SIZE) ==="
PIDS=()
for shard in $(seq 0 $((NUM_SHARDS-1))); do
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
        > "$LOGDIR/v2_batch_gr_shard${shard}_${SLURM_JOB_ID}.log" 2>&1 &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do wait $pid || true; done
echo "Grounding done: $(date)"
for shard in $(seq 0 $((NUM_SHARDS-1))); do
    echo "--- Shard $shard ---"
    tail -3 "$LOGDIR/v2_batch_gr_shard${shard}_${SLURM_JOB_ID}.log" 2>/dev/null
done

# ── Merge results ──
echo "=== Merging ==="
python -c "
import json, glob, os
for eval_type in ['action_prediction', 'grounding']:
    rdir = '$RESULTS_BASE/' + eval_type
    files = sorted(glob.glob(os.path.join(rdir, 'results_shard*_*.json')))
    if not files:
        print(f'No shard files for {eval_type}'); continue
    all_results = []
    for f in files:
        with open(f) as fh: all_results.extend(json.load(fh))
    total = len(all_results)
    success = sum(1 for r in all_results if r.get('success'))
    rate = 100 * success / total if total > 0 else 0
    with open(os.path.join(rdir, 'merged_summary.json'), 'w') as fh:
        json.dump({'total': total, 'success': success, 'rate': rate}, fh, indent=2)
    print(f'{eval_type}: {success}/{total} = {rate:.1f}%')
"
echo "All done: $(date)"
