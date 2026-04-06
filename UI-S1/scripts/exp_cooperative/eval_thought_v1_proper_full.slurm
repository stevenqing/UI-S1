#!/bin/bash
#SBATCH --job-name=eval_coop_full
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_proper_full_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_proper_full_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Full proper routing eval — 4 GPU parallel
# Each GPU runs 1/4 of the data
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
NUM_SHARDS=4

echo "=========================================="
echo "Cooperative LoRA v1 — Full Proper Routing Eval"
echo "4 GPUs x 2 eval_types = 8 parallel processes"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

cd "$PROJECT_DIR"

RESULTS_BASE="$PROJECT_DIR/train_GUI_360/GUI-360-eval/results/cooperative_thought_v1_proper"
BASE_MODEL="checkpoints/Qwen2.5-VL-7B-Instruct"
COOP_CKPT="train_GUI_360/llamafactory/output/cooperative_thought_v1/final"

# ── Phase 1: Grounding (4 shards on 4 GPUs) ──
echo ""
echo "=== Phase 1: Grounding ==="
PIDS=()
for shard in $(seq 0 $((NUM_SHARDS-1))); do
    echo "Launching grounding shard $shard on GPU $shard..."
    python evaluation/eval_cooperative_proper.py \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$COOP_CKPT" \
        --eval_data_root datasets/GUI-360/test \
        --eval_type grounding \
        --output_dir "$RESULTS_BASE/grounding" \
        --gpu_id $shard \
        --shard_id $shard \
        --num_shards $NUM_SHARDS &
    PIDS+=($!)
done

echo "Waiting for grounding shards..."
FAIL=0
for pid in "${PIDS[@]}"; do
    wait $pid || FAIL=1
done
if [ $FAIL -eq 1 ]; then
    echo "WARNING: Some grounding shards failed"
fi
echo "Grounding done: $(date)"

# ── Phase 2: Action Prediction (4 shards on 4 GPUs) ──
echo ""
echo "=== Phase 2: Action Prediction ==="
PIDS=()
for shard in $(seq 0 $((NUM_SHARDS-1))); do
    echo "Launching action_prediction shard $shard on GPU $shard..."
    python evaluation/eval_cooperative_proper.py \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$COOP_CKPT" \
        --eval_data_root datasets/GUI-360/test \
        --eval_type action_prediction \
        --output_dir "$RESULTS_BASE/action_prediction" \
        --gpu_id $shard \
        --shard_id $shard \
        --num_shards $NUM_SHARDS &
    PIDS+=($!)
done

echo "Waiting for action_prediction shards..."
FAIL=0
for pid in "${PIDS[@]}"; do
    wait $pid || FAIL=1
done
if [ $FAIL -eq 1 ]; then
    echo "WARNING: Some action_prediction shards failed"
fi
echo "Action prediction done: $(date)"

# ── Phase 3: Merge shard results ──
echo ""
echo "=== Merging results ==="
python -c "
import json, glob, os

for eval_type in ['grounding', 'action_prediction']:
    result_dir = '$RESULTS_BASE/' + eval_type
    shard_files = sorted(glob.glob(os.path.join(result_dir, 'evaluation_results_shard*_*.json')))
    if not shard_files:
        print(f'No shard files for {eval_type}')
        continue

    all_results = []
    for f in shard_files:
        with open(f) as fh:
            all_results.extend(json.load(fh))

    total = len(all_results)
    success = sum(1 for r in all_results if r.get('success'))
    errors = sum(1 for r in all_results if 'error' in r)
    rate = 100 * success / total if total > 0 else 0

    summary = {
        'total_samples': total,
        'success_count': success,
        'success_rate': rate,
        'error_count': errors,
        'eval_type': eval_type,
        'inference_mode': 'proper_routing (no merge)',
        'num_shards': $NUM_SHARDS,
    }

    with open(os.path.join(result_dir, 'merged_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(result_dir, 'merged_results.json'), 'w') as fh:
        json.dump(all_results, fh, indent=2)

    print(f'{eval_type}: {success}/{total} = {rate:.1f}% (errors={errors})')
"

echo "=========================================="
echo "All done: $(date)"
echo "=========================================="
