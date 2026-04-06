#!/bin/bash
#SBATCH --job-name=eval_smoe_vllm
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp3/logs/eval_vllm_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp3/logs/eval_vllm_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Selective MoE: Merge + vLLM Evaluation
# Step 1: Merge MoE LoRA into base model
# Step 2: Start vLLM server
# Step 3: Run GUI-360 eval (26K test set)
# ============================================

set -e

# --- Configuration ---
CONDITION="${CONDITION:-c2}"
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
CHECKPOINT_DIR="$PROJECT_DIR/train_GUI_360/llamafactory/output/selective_moe_${CONDITION}/final/moe_checkpoint"
MERGED_DIR="$PROJECT_DIR/train_GUI_360/llamafactory/output/selective_moe_${CONDITION}/merged_model"
EVAL_DIR="$PROJECT_DIR/evaluation"
PARQUET_FILE="$PROJECT_DIR/train_GUI_360/data/gui360_test_sft_eval_format_with_bbox.parquet"
OUTPUT_DIR="$PROJECT_DIR/scripts/exp3/results/vllm_eval_${CONDITION}"
LOG_DIR="$PROJECT_DIR/scripts/exp3/logs"

VLLM_PORT=8000
API_URL="http://localhost:${VLLM_PORT}/v1"
EVAL_WORKERS=64

echo "=========================================="
echo "Selective MoE vLLM Evaluation: ${CONDITION}"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Merged model: ${MERGED_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

# Environment setup
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

# Fix TMPDIR/cache
export TMPDIR=/tmp/${SLURM_JOB_ID}
mkdir -p $TMPDIR
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor
export TORCH_HOME=$TMPDIR/torch
export XDG_CACHE_HOME=$TMPDIR/cache
export HF_HOME=$TMPDIR/hf_home
export HF_DATASETS_CACHE=$TMPDIR/hf_datasets
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TORCH_HOME $XDG_CACHE_HOME $HF_HOME $HF_DATASETS_CACHE

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# Validate inputs
if [ ! -f "$CHECKPOINT_DIR/lora_weights.pt" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_DIR"
    exit 1
fi

# ============================================
# Step 1: Merge MoE LoRA into base model
# ============================================
if [ -f "$MERGED_DIR/model.safetensors.index.json" ]; then
    echo "Merged model already exists, skipping merge step."
else
    echo ""
    echo "=== Step 1: Merging MoE LoRA into base model ==="
    echo "Start: $(date)"

    python "$EVAL_DIR/merge_moe_to_full.py" \
        --base_model "$BASE_MODEL" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --output_dir "$MERGED_DIR"

    echo "Merge completed: $(date)"
fi

# Validate merged model
if [ ! -f "$MERGED_DIR/model.safetensors.index.json" ]; then
    echo "ERROR: Merged model incomplete at $MERGED_DIR"
    exit 1
fi
echo "Merged model verified: $MERGED_DIR"

# ============================================
# Step 2: Start vLLM Server
# ============================================
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2

echo ""
echo "=== Step 2: Starting vLLM server ==="
python -m vllm.entrypoints.openai.api_server \
    --model "$MERGED_DIR" \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee "$LOG_DIR/vllm_${CONDITION}_${SLURM_JOB_ID}.log" &
VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Wait for vLLM
MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM server is ready! (${WAIT_COUNT}s)"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    if [ $((WAIT_COUNT % 30)) -eq 0 ]; then
        echo "Waiting for vLLM... ($WAIT_COUNT/$MAX_WAIT seconds)"
    fi
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server failed to start within $MAX_WAIT seconds"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# ============================================
# Step 3: Run GUI-360 Evaluation
# ============================================
echo ""
echo "=== Step 3: GUI-360 Evaluation (baseline mode) ==="
echo "Parquet: $PARQUET_FILE"
echo "Workers: $EVAL_WORKERS"
echo "Start: $(date)"

cd "$EVAL_DIR"

MODEL_NAME="$MERGED_DIR"

python eval_multiview_gui360.py \
    --parquet_file "$PARQUET_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --mode baseline \
    --api_url "$API_URL" \
    --max_workers $EVAL_WORKERS \
    --resume

echo ""
echo "Evaluation completed: $(date)"

# ============================================
# Cleanup
# ============================================
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true

echo ""
echo "=========================================="
echo "Selective MoE vLLM Evaluation Complete: ${CONDITION}"
echo "=========================================="
echo "End: $(date)"
echo "Results: $OUTPUT_DIR"
echo ""
echo "Summary:"
cat "$OUTPUT_DIR"/*_summary.json 2>/dev/null | python3 -m json.tool || echo "No summary found"
echo "=========================================="
