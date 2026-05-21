#!/bin/bash
#SBATCH --job-name=v9_eval
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v9/scripts/logs/eval_v9_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v9/scripts/logs/eval_v9_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# v9 AR Trajectory eval on AndroidControl.
# Parameterized via env vars:
#   COOP_CKPT     — cooperative checkpoint path (relative to PROJECT_DIR)
#   RESULTS_BASE  — output directory for results
#
# Usage:
#   COOP_CKPT=v9/output/v9_ac_comm/epoch-1 \
#   RESULTS_BASE=v9/results/v9_ac_comm/epoch1 \
#   sbatch v9/scripts/eval_v9_ac_trajectory.slurm
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="checkpoints/Qwen2.5-VL-7B-Instruct"
COOP_CKPT="${COOP_CKPT:?ERROR: set COOP_CKPT}"
RESULTS_BASE="${RESULTS_BASE:?ERROR: set RESULTS_BASE}"
JSONL_FILE="${JSONL_FILE:-datasets/android_control_evaluation_std.jsonl}"
LOGDIR="$PROJECT_DIR/v9/scripts/logs"
NUM_SHARDS=4
N_HISTORY_IMG=2
MAX_NEW_TOKENS=512

mkdir -p "$LOGDIR" "$PROJECT_DIR/$RESULTS_BASE"

echo "=========================================="
echo "v9 AC AR-Trajectory Eval"
echo "=========================================="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "COOP_CKPT: $COOP_CKPT"
echo "OUT:       $RESULTS_BASE"
echo "=========================================="

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

cd "$PROJECT_DIR"

echo "Launching ${NUM_SHARDS} shards"
PIDS=()
for shard in $(seq 0 $((NUM_SHARDS-1))); do
    LOG="$LOGDIR/v9_eval_shard${shard}_${SLURM_JOB_ID}.log"
    echo "  shard=${shard} gpu=${shard} -> ${LOG}"
    python -u evaluation/eval_cooperative_ac_trajectory.py \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$COOP_CKPT" \
        --jsonl_file "$JSONL_FILE" \
        --output_dir "$RESULTS_BASE" \
        --gpu_id $shard \
        --shard_id $shard \
        --num_shards $NUM_SHARDS \
        --n_history_image_limit $N_HISTORY_IMG \
        --max_new_tokens $MAX_NEW_TOKENS \
        --no_stop \
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
    echo "--- shard${shard} last 8 lines ---"
    tail -8 "$LOGDIR/v9_eval_shard${shard}_${SLURM_JOB_ID}.log" 2>/dev/null || true
done

echo ""
echo "=== aggregate trajectory summary ==="
python - << PYEOF
import json, glob, os
out_dir = "$RESULTS_BASE"
shards = sorted(glob.glob(os.path.join(out_dir, "summary_shard*.json")))
total_ep, success, sum_prog = 0, 0, 0.0
all_action = {}
for s in shards:
    d = json.load(open(s))
    n = d.get('total_episodes', 0)
    succ = d.get('success_count', 0)
    tsr = d.get('tsr', 0.0)
    avg_prog = d.get('avg_progress', 0.0)
    sc_prog = d.get('scattered_progress', 0.0)
    print(f"  {os.path.basename(s)}: n={n} TSR={tsr:.4f} ({succ}/{n}) "
          f"AvgProg={avg_prog:.4f} Scattered={sc_prog:.4f}")
    total_ep += n
    success += succ
    sum_prog += avg_prog * n
    for at, st in d.get('action_type_stats', {}).items():
        if at not in all_action:
            all_action[at] = {'total': 0, 'type_match': 0, 'extract_match': 0}
        all_action[at]['total'] += st.get('total', 0)
        all_action[at]['type_match'] += st.get('type_match', 0)
        all_action[at]['extract_match'] += st.get('extract_match', 0)

if total_ep:
    overall_tsr = success / total_ep
    overall_avg_prog = sum_prog / total_ep
    print(f"\n=== AGGREGATE ===")
    print(f"  episodes:       {total_ep}")
    print(f"  TSR:            {100*overall_tsr:.2f}%  ({success}/{total_ep})")
    print(f"  Avg Progress:   {overall_avg_prog:.4f}")
    print(f"\n  per-action extract_match rate:")
    for at in sorted(all_action.keys()):
        st = all_action[at]
        rate = st['extract_match'] / st['total'] if st['total'] else 0
        print(f"    {at:18s}: {rate:.4f}  ({st['extract_match']}/{st['total']})")
PYEOF

echo ""
echo "Done: $(date)"
echo "=========================================="
