#!/bin/bash
#SBATCH --job-name=eval_odyssey
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/gui_odyssey_eval/logs/eval_odyssey_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/gui_odyssey_eval/logs/eval_odyssey_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# GUI-Odyssey AR Trajectory Evaluation
# ============================================
# Evaluates a model on GUI-Odyssey using the
# same pipeline as AndroidControl AR eval.
# ============================================

set -e

# ── Parameters (override via sbatch --export) ──
PROJECT_DIR="${PROJECT_DIR:-/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct}"
MODEL_NAME="${MODEL_NAME:-$MODEL_PATH}"
SPLIT="${SPLIT:-random_split}"
SUBSET="${SUBSET:-test}"
MAX_EPISODES="${MAX_EPISODES:-}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
N_HISTORY_IMAGES="${N_HISTORY_IMAGES:-2}"

ODYSSEY_DIR="$PROJECT_DIR/gui_odyssey_eval"
DATASET_DIR="$PROJECT_DIR/datasets/GUI-Odyssey"
LOG_DIR="$ODYSSEY_DIR/logs"
VLLM_PORT=8000

echo "=========================================="
echo "GUI-Odyssey AR Trajectory Evaluation"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Model: $MODEL_NAME"
echo "Split: $SPLIT / $SUBSET"
echo "=========================================="

# ── Environment ──
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

mkdir -p "$LOG_DIR"

# ── Validate inputs ──
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset not found at $DATASET_DIR"
    exit 1
fi

# ── Step 1: Convert dataset (if not already done) ──
JSONL_FILE="$DATASET_DIR/gui_odyssey_${SPLIT}_${SUBSET}.jsonl"
if [ ! -f "$JSONL_FILE" ]; then
    echo "Converting GUI-Odyssey ${SPLIT}/${SUBSET} to eval format..."
    python "$ODYSSEY_DIR/convert_to_eval_format.py" \
        --data_dir "$DATASET_DIR" \
        --split "$SPLIT" \
        --subset "$SUBSET" \
        --output_dir "$DATASET_DIR"
    echo "Conversion done: $JSONL_FILE"
else
    echo "JSONL already exists: $JSONL_FILE"
fi

# ── Step 2: Start vLLM server ──
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image": 2}' \
    2>&1 | tee "$LOG_DIR/vllm_odyssey_${SLURM_JOB_ID}.log" &

VLLM_PID=$!

MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    echo "Waiting for vLLM... ($WAIT_COUNT/$MAX_WAIT seconds)"
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server failed to start within $MAX_WAIT seconds"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# ── Step 3: Run evaluation ──
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$ODYSSEY_DIR/results/${SPLIT}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Running AR trajectory evaluation..."
echo "Output: $OUTPUT_DIR"

EXTRA_ARGS=""
if [ -n "$MAX_EPISODES" ]; then
    EXTRA_ARGS="--max_episodes $MAX_EPISODES"
fi

cd "$ODYSSEY_DIR"
python eval_ar_trajectory.py \
    --jsonl_file "$JSONL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --split_name "$SPLIT" \
    --n_history_image_limit "$N_HISTORY_IMAGES" \
    --max_workers "$EVAL_WORKERS" \
    $EXTRA_ARGS

echo "Stop-on-error evaluation completed."
cat "$OUTPUT_DIR/summary.json" 2>/dev/null || echo "No summary found"

# ── Step 4: Run no-stop evaluation ──
NOSTOP_DIR="${OUTPUT_DIR}_nostop"
mkdir -p "$NOSTOP_DIR"

echo ""
echo "=========================================="
echo "Running no-stop evaluation (all steps)..."
echo "Output: $NOSTOP_DIR"
echo "=========================================="

python eval_ar_trajectory.py \
    --jsonl_file "$JSONL_FILE" \
    --output_dir "$NOSTOP_DIR" \
    --model_name "$MODEL_NAME" \
    --split_name "$SPLIT" \
    --n_history_image_limit "$N_HISTORY_IMAGES" \
    --max_workers "$EVAL_WORKERS" \
    --no_stop \
    $EXTRA_ARGS

echo "No-stop evaluation completed."
cat "$NOSTOP_DIR/summary.json" 2>/dev/null || echo "No summary found"

# ── Step 5: Run analysis ──
echo ""
echo "=========================================="
echo "Running failure analysis..."
echo "=========================================="

python analyze_odyssey_results.py \
    --stop_dir "$OUTPUT_DIR" \
    --nostop_dir "$NOSTOP_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "Analysis completed."

# ── Cleanup ──
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true

echo "End time: $(date)"
echo "Results: $OUTPUT_DIR"
