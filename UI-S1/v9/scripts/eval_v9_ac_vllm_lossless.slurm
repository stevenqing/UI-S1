#!/bin/bash
#SBATCH --job-name=v9_ll_eval
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v9/scripts/logs/eval_v9_vllm_ll_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v9/scripts/logs/eval_v9_vllm_ll_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# v9 AR Trajectory eval on AndroidControl via vLLM (LOSSLESS)
#
# 1) Lossless merge: base weights + correction adapters
# 2) Start corrected vLLM server (per-token correction hooks)
# 3) Run AR trajectory eval (multi-threaded, fast)
#
# Usage:
#   COOP_CKPT=v9/output/v9_ac_comm/epoch-1 \
#   RESULTS_BASE=v9/results/v9_ac_comm_ll/epoch1 \
#   sbatch v9/scripts/eval_v9_ac_vllm_lossless.slurm
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
COOP_CKPT="${COOP_CKPT:?ERROR: set COOP_CKPT}"
RESULTS_BASE="${RESULTS_BASE:?ERROR: set RESULTS_BASE}"
DATASET_PATH="${DATASET_PATH:-$PROJECT_DIR/evaluation/results/android_control_evaluation_fixed.jsonl}"
COORD_ROUTING="${COORD_ROUTING:-1}"
MAX_WORKERS="${MAX_WORKERS:-128}"

# Derive a short name for the merged model temp dir
CKPT_NAME=$(echo "$COOP_CKPT" | sed 's|/|_|g' | tail -c 60)
MERGED_MODEL="/tmp/merged_lossless_v9_${CKPT_NAME}_${SLURM_JOB_ID}"
MODEL_NAME="v9_eval"

echo "=========================================="
echo "v9 AC AR-Trajectory Eval (vLLM LOSSLESS)"
echo "=========================================="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
echo "COOP_CKPT:    $COOP_CKPT"
echo "MERGED_MODEL: $MERGED_MODEL"
echo "OUT:          $RESULTS_BASE"
echo "DATASET:      $DATASET_PATH"
echo "COORD_ROUTING:$COORD_ROUTING"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
mkdir -p "$RESULTS_BASE" "v9/scripts/logs"

# ── Step 1: Lossless merge ──────────────────────────────────────────
echo "Step 1: Lossless merge..."
python evaluation/merge_cooperative_lossless.py \
    --base_model "$BASE_MODEL" \
    --coop_checkpoint "$COOP_CKPT" \
    --output_dir "$MERGED_MODEL"

MERGE_EXIT=$?
if [ $MERGE_EXIT -ne 0 ]; then
    echo "ERROR: Lossless merge failed with exit code $MERGE_EXIT"
    exit $MERGE_EXIT
fi

if [ ! -f "$MERGED_MODEL/correction_adapters.pt" ]; then
    echo "ERROR: correction_adapters.pt not found after merge"
    exit 1
fi
echo "Lossless merge complete."
ls -lh "$MERGED_MODEL/correction_adapters.pt" "$MERGED_MODEL/correction_config.json"

# ── Step 2: Start corrected vLLM server ─────────────────────────────
echo "Step 2: Starting corrected vLLM server..."

COORD_FLAG=""
if [ "$COORD_ROUTING" = "1" ]; then
    COORD_FLAG="--coord-routing"
fi

python evaluation/serve_corrected_vllm.py \
    --model "$MERGED_MODEL" \
    --served-model-name "$MODEL_NAME" \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image": 2}' \
    $COORD_FLAG \
    &
VLLM_PID=$!

MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Corrected vLLM server ready after ${WAIT_COUNT}s."
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    if [ $((WAIT_COUNT % 60)) -eq 0 ]; then
        echo "Waiting for vLLM server... ($WAIT_COUNT/$MAX_WAIT seconds)"
    fi
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: Corrected vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null
    rm -rf "$MERGED_MODEL"
    exit 1
fi

# ── Step 3: Run AR trajectory eval ──────────────────────────────────
echo "Step 3: Running AR trajectory eval..."
cd "$PROJECT_DIR/evaluation"

python eval_qwenvl.py \
    --model_name "$MODEL_NAME" \
    --jsonl_file "$DATASET_PATH" \
    --output_dir "$RESULTS_BASE" \
    --n_history_image_limit 2 \
    --max_workers "$MAX_WORKERS"

EVAL_EXIT=$?

# ── Cleanup ─────────────────────────────────────────────────────────
kill $VLLM_PID 2>/dev/null || true
pkill -f "serve_corrected_vllm" || true
wait $VLLM_PID 2>/dev/null
rm -rf "$MERGED_MODEL"

echo "=========================================="
echo "Done: $(date)  exit=$EVAL_EXIT"
echo "Results: $RESULTS_BASE"
echo "=========================================="
exit $EVAL_EXIT
