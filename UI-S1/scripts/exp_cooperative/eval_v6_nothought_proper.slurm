#!/bin/bash
#SBATCH --job-name=eval_v6nt_p
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_v6nt_proper_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_v6nt_proper_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Proper-routing eval for v6 cooperative LoRA (nothought)
# Uses eval_cooperative_batch.py with true per-token routing
# (NOT the merge+vllm approximation)
#
# Layout: 2 nodes × 4 GPUs = 8 shards total
#   - Node 0: shards 0-3 grounding       (4 GPUs)
#   - Node 1: shards 0-3 action_prediction (4 GPUs)
# Both eval types run in parallel -> total time = max(grounding, action_pred)
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="checkpoints/Qwen2.5-VL-7B-Instruct"
COOP_CKPT="train_GUI_360/llamafactory/output/cooperative_v6_comm_nothought/final"
RESULTS_BASE="train_GUI_360/GUI-360-eval/results/cooperative_v6_nothought_proper"
LOGDIR="$PROJECT_DIR/scripts/exp_cooperative/logs"
NUM_SHARDS=4
BATCH_SIZE=32

echo "=========================================="
echo "Cooperative LoRA v6 (nothought) Proper Routing Eval"
echo "=========================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Nodes:       ${SLURM_NNODES}"
echo "Node list:   ${SLURM_NODELIST}"
echo "Start:       $(date)"
echo "=========================================="

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Worker script: launched once per node via srun
WORKER_SCRIPT="$LOGDIR/eval_v6nt_proper_worker_${SLURM_JOB_ID}.sh"
cat > "$WORKER_SCRIPT" << 'WORKEREOF'
#!/bin/bash
set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="checkpoints/Qwen2.5-VL-7B-Instruct"
COOP_CKPT="train_GUI_360/llamafactory/output/cooperative_v6_comm_nothought/final"
RESULTS_BASE="train_GUI_360/GUI-360-eval/results/cooperative_v6_nothought_proper"
LOGDIR="$PROJECT_DIR/scripts/exp_cooperative/logs"
NUM_SHARDS=4
BATCH_SIZE=32

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

cd "$PROJECT_DIR"

# Decide eval type per node (node 0 = grounding, node 1 = action_prediction)
if [ "${SLURM_NODEID}" = "0" ]; then
    EVAL_TYPE="grounding"
else
    EVAL_TYPE="action_prediction"
fi

mkdir -p "$RESULTS_BASE/${EVAL_TYPE}"
OUT_DIR="$RESULTS_BASE/${EVAL_TYPE}"

echo "[node ${SLURM_NODEID} / $(hostname)] EVAL_TYPE=${EVAL_TYPE}  NUM_SHARDS=${NUM_SHARDS}"

PIDS=()
for shard in $(seq 0 $((NUM_SHARDS-1))); do
    LOG="$LOGDIR/v6nt_proper_${EVAL_TYPE}_shard${shard}_${SLURM_JOB_ID}.log"
    echo "[node ${SLURM_NODEID}] launching shard=${shard} gpu=${shard} -> ${LOG}"
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

# Wait for all shards on this node
for pid in "${PIDS[@]}"; do
    wait $pid || echo "[node ${SLURM_NODEID}] shard PID=$pid exited non-zero"
done

echo "[node ${SLURM_NODEID}] ${EVAL_TYPE} all shards done: $(date)"

# Print tails of shard logs for quick inspection
for shard in $(seq 0 $((NUM_SHARDS-1))); do
    echo "--- [node ${SLURM_NODEID}] ${EVAL_TYPE} shard${shard} last 5 lines ---"
    tail -5 "$LOGDIR/v6nt_proper_${EVAL_TYPE}_shard${shard}_${SLURM_JOB_ID}.log" 2>/dev/null || true
done
WORKEREOF

chmod +x "$WORKER_SCRIPT"

echo "=========================================="
echo "Dispatching workers to both nodes"
echo "=========================================="

srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=144 \
    bash "$WORKER_SCRIPT"

echo "=========================================="
echo "Both eval types completed: $(date)"
echo "=========================================="

# Aggregate shards for each eval type
cd "$PROJECT_DIR"
for EVAL_TYPE in grounding action_prediction; do
    OUT_DIR="$RESULTS_BASE/${EVAL_TYPE}"
    echo ""
    echo "=== ${EVAL_TYPE} shard summaries ==="
    ls "$OUT_DIR"/summary_shard*.json 2>/dev/null || true

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
    print(f"\n=== $EVAL_TYPE AGGREGATE: total={total}, success={success}, rate={100*success/total:.2f}% ===")
PYEOF
done

echo ""
echo "Done: $(date)"
echo "=========================================="
