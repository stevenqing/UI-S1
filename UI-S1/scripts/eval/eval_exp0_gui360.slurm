#!/bin/bash
#SBATCH --job-name=exp0_gui360
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/exp0_gui360_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/exp0_gui360_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================================
# Exp 0: CoT vs Decomposition — GUI-360 Step-Level Evaluation
# ============================================================
# Runs 4 conditions sequentially on the same vLLM server:
#   C0       — standard single-pass baseline
#   CoT      — two-turn self-CoT (describe UI elements, then act)
#   Decomp   — separate observer + action model (F4-equivalent)
#   RandDesc — action model with random screenshot's description
#
# Override model: MODEL_PATH=/path/to/model sbatch eval_exp0_gui360.slurm
# Override conditions: CONDITIONS="C0,CoT" sbatch eval_exp0_gui360.slurm
# ============================================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="$PROJECT_DIR/evaluation"
LOG_DIR="$PROJECT_DIR/train_GUI_360/logs"

# Model — default to GUI-360 SFT v2 checkpoint
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/train_GUI_360/llamafactory/output/gui360_full_sft_v2/checkpoint-796}"
MODEL_NAME="$MODEL_PATH"

# Dataset
PARQUET="${PARQUET:-$PROJECT_DIR/train_GUI_360/data/gui360_test_sft_eval_format_with_bbox.parquet}"

# Conditions to run (comma-separated)
CONDITIONS="${CONDITIONS:-C0,CoT,Decomp,RandDesc}"

VLLM_PORT=8000
API_URL="http://localhost:${VLLM_PORT}/v1"
EVAL_WORKERS="${EVAL_WORKERS:-128}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_DIR/evaluation/results/exp0_gui360_${TIMESTAMP}"

echo "============================================================"
echo "Exp 0: CoT vs Decomposition — GUI-360"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Model: $MODEL_PATH"
echo "Dataset: $PARQUET"
echo "Conditions: $CONDITIONS"
echo "Workers: $EVAL_WORKERS"
echo "Max samples: $MAX_SAMPLES (0=all)"
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

# Validate model
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
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
    2>&1 | tee "$LOG_DIR/vllm_exp0_gui360_${SLURM_JOB_ID}.log" &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for vLLM
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
    echo "Running condition: $COND"
    echo "Start: $(date)"
    echo "============================================"

    # RandDesc needs sequential processing, reduce workers
    if [ "$COND" = "RandDesc" ]; then
        COND_WORKERS=1
    else
        COND_WORKERS=$EVAL_WORKERS
    fi

    python eval_exp0_cot_vs_decomp.py \
        --model_name "$MODEL_NAME" \
        --condition "$COND" \
        --dataset gui360 \
        --parquet_file "$PARQUET" \
        --api_url "$API_URL" \
        --output_dir "$OUTPUT_DIR" \
        --max_workers $COND_WORKERS \
        --max_samples $MAX_SAMPLES \
        --save_debug \
        --resume

    echo "Condition $COND completed at $(date)"
done

# ============================================
# Print Comparison Summary
# ============================================
echo ""
echo "============================================================"
echo "Exp 0 — All Conditions Complete"
echo "============================================================"
echo ""

for f in "$OUTPUT_DIR"/*_summary.json; do
    if [ -f "$f" ]; then
        echo "--- $(basename "$f") ---"
        python3 -m json.tool "$f"
        echo ""
    fi
done

echo ""
echo "Interpretation Guide:"
echo "  CoT ≈ Decomp > C0  → H5 (CoT effect), no need for multi-agent/MoE"
echo "  CoT < Decomp       → Decomposition has independent value → MoE justified"
echo "  RandDesc ≈ Decomp  → Format effect, description content doesn't matter"
echo "  RandDesc << Decomp → Description content matters → decomposition is real"
echo "============================================================"

# Cleanup
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true
echo "Done at $(date)"
