#!/bin/bash
#SBATCH --job-name=exp0_ac
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/exp0_ac_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/exp0_ac_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================================
# Exp 0: CoT vs Decomposition — AndroidControl Trajectory Eval
# ============================================================
# Runs 3 conditions (no RandDesc for trajectory — too slow):
#   C0       — standard single-pass baseline
#   CoT      — two-turn self-CoT
#   Decomp   — separate observer + action model (F4-equivalent)
#
# Override model: MODEL_PATH=/path/to/model sbatch eval_exp0_ac.slurm
# ============================================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="$PROJECT_DIR/evaluation"
LOG_DIR="$PROJECT_DIR/train_GUI_360/logs"

# Model — default to AC Full SFT
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/train_GUI_360/llamafactory/output/ac_full_sft/checkpoint-100}"
MODEL_NAME="$MODEL_PATH"

# Dataset
AC_JSONL="${AC_JSONL:-$PROJECT_DIR/evaluation/dataset/android_control_evaluation_fixed.jsonl}"

# Conditions
CONDITIONS="${CONDITIONS:-C0,CoT,Decomp}"

VLLM_PORT=8000
API_URL="http://localhost:${VLLM_PORT}/v1"
EVAL_WORKERS="${EVAL_WORKERS:-128}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_DIR/evaluation/results/exp0_ac_${TIMESTAMP}"

echo "============================================================"
echo "Exp 0: CoT vs Decomposition — AndroidControl"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Model: $MODEL_PATH"
echo "Dataset: $AC_JSONL"
echo "Conditions: $CONDITIONS"
echo "Workers: $EVAL_WORKERS"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

# Environment
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

export TMPDIR=/tmp/${SLURM_JOB_ID}
mkdir -p $TMPDIR 2>/dev/null || true
# Fallback to scratch if /tmp is full
if ! touch "$TMPDIR/.test_write" 2>/dev/null; then
    echo "WARNING: /tmp is full, using scratch for TMPDIR"
    export TMPDIR=/scratch/a5l/shuqing.a5l/tmp/${SLURM_JOB_ID}
    mkdir -p $TMPDIR
fi
rm -f "$TMPDIR/.test_write" 2>/dev/null
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor
export TORCH_HOME=$TMPDIR/torch
export XDG_CACHE_HOME=$TMPDIR/cache
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TORCH_HOME $XDG_CACHE_HOME

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# Validate
if [ ! -f "$MODEL_PATH/model.safetensors.index.json" ]; then
    echo "ERROR: Model checkpoint incomplete at $MODEL_PATH"
    ls -la "$MODEL_PATH/" 2>/dev/null
    exit 1
fi

# ============================================
# Start vLLM Server
# ============================================
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee "$LOG_DIR/vllm_exp0_ac_${SLURM_JOB_ID}.log" &
VLLM_PID=$!

MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM server ready!"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    echo "Waiting for vLLM... ($WAIT_COUNT/$MAX_WAIT)"
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# ============================================
# Run Conditions
# ============================================
cd "$EVAL_DIR"

IFS=',' read -ra CONDS <<< "$CONDITIONS"
for COND in "${CONDS[@]}"; do
    echo ""
    echo "============================================"
    echo "Running condition: $COND (AndroidControl)"
    echo "Start: $(date)"
    echo "============================================"

    python eval_exp0_cot_vs_decomp.py \
        --model_name "$MODEL_NAME" \
        --condition "$COND" \
        --dataset ac \
        --jsonl_file "$AC_JSONL" \
        --api_url "$API_URL" \
        --output_dir "$OUTPUT_DIR" \
        --max_workers $EVAL_WORKERS \
        --n_history_image_limit 2 \
        --no_thought \
        --save_debug \
        --resume

    echo "Condition $COND completed at $(date)"
done

# ============================================
# Summary
# ============================================
echo ""
echo "============================================================"
echo "Exp 0 — AndroidControl — All Conditions Complete"
echo "============================================================"
for f in "$OUTPUT_DIR"/*_summary.json; do
    if [ -f "$f" ]; then
        echo "--- $(basename "$f") ---"
        python3 -m json.tool "$f"
        echo ""
    fi
done
echo "============================================================"

kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true
echo "Done at $(date)"
