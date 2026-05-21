#!/bin/bash
#SBATCH --job-name=v65_ody_traj
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_v6_5_odyssey_traj_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_v6_5_odyssey_traj_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# AR Trajectory eval for v6.5 cooperative LoRA on GUI-Odyssey
# Uses HF generate (vLLM incompatible with cooperative wrapper).
# 4 shards across 4 GPUs, sequential within each shard.
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="checkpoints/Qwen2.5-VL-7B-Instruct"
# IMPORTANT: this checkpoint must be the Odyssey-trained v6.5 (not GUI-360 / not AC).
COOP_CKPT="${COOP_CKPT:-train_GUI_360/llamafactory/output/cooperative_v6_5_odyssey/epoch-4}"
RESULTS_BASE="${RESULTS_BASE:-outputs/eval_cooperative_v6_5_odyssey/epoch4}"
JSONL_FILE="${JSONL_FILE:-datasets/GUI-Odyssey/gui_odyssey_random_split_test.jsonl}"
LOGDIR="$PROJECT_DIR/scripts/exp_cooperative/logs"
NUM_SHARDS=4
N_HISTORY_IMG=2
MAX_NEW_TOKENS=512
NO_STOP="${NO_STOP:-1}"  # default no_stop=on

mkdir -p "$LOGDIR"

echo "=========================================="
echo "Cooperative LoRA v6.5 Odyssey AR-Trajectory Eval"
echo "=========================================="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "BASE:      $BASE_MODEL"
echo "COOP_CKPT: $COOP_CKPT"
echo "JSONL:     $JSONL_FILE"
echo "OUT:       $RESULTS_BASE"
echo "no_stop:   $NO_STOP"
echo "=========================================="

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

cd "$PROJECT_DIR"

mkdir -p "$RESULTS_BASE"

NO_STOP_FLAG=""
if [ "$NO_STOP" = "1" ]; then
    NO_STOP_FLAG="--no_stop"
fi

echo "Launching ${NUM_SHARDS} shards"
PIDS=()
for shard in $(seq 0 $((NUM_SHARDS-1))); do
    LOG="$LOGDIR/v6_5_odyssey_traj_shard${shard}_${SLURM_JOB_ID}.log"
    echo "  launching shard=${shard} gpu=${shard} -> ${LOG}"
    python -u evaluation/eval_cooperative_odyssey_trajectory.py \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$COOP_CKPT" \
        --jsonl_file "$JSONL_FILE" \
        --output_dir "$RESULTS_BASE" \
        --gpu_id $shard \
        --shard_id $shard \
        --num_shards $NUM_SHARDS \
        --n_history_image_limit $N_HISTORY_IMG \
        --max_new_tokens $MAX_NEW_TOKENS \
        $NO_STOP_FLAG \
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
    tail -8 "$LOGDIR/v6_5_odyssey_traj_shard${shard}_${SLURM_JOB_ID}.log" 2>/dev/null || true
done

echo ""
echo "=== aggregate trajectory summary ==="
python - << PYEOF
import json, glob, os
from collections import defaultdict

out_dir = "$RESULTS_BASE"
shards = sorted(glob.glob(os.path.join(out_dir, "summary_shard*.json")))
total_ep, success, sum_prog = 0, 0, 0.0
all_action = {}
cat_agg = defaultdict(lambda: {'n': 0, 'success': 0})
dev_agg = defaultdict(lambda: {'n': 0, 'success': 0})

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
    for cat, m in d.get('category_stats', {}).items():
        cat_agg[cat]['n'] += m.get('n', 0)
        cat_agg[cat]['success'] += m.get('success_count', 0)
    for dev, m in d.get('device_stats', {}).items():
        dev_agg[dev]['n'] += m.get('n', 0)
        dev_agg[dev]['success'] += m.get('success_count', 0)

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
    print(f"\n  per-category TSR:")
    for cat in sorted(cat_agg.keys()):
        a = cat_agg[cat]
        r = a['success']/a['n'] if a['n'] else 0
        print(f"    {cat:24s}: {r:.4f}  ({a['success']}/{a['n']})")
    print(f"\n  per-device TSR:")
    for dev in sorted(dev_agg.keys()):
        a = dev_agg[dev]
        r = a['success']/a['n'] if a['n'] else 0
        print(f"    {dev:24s}: {r:.4f}  ({a['success']}/{a['n']})")
PYEOF

echo ""
echo "Done: $(date)"
echo "=========================================="
