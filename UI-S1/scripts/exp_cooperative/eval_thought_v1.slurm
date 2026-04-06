#!/bin/bash
#SBATCH --job-name=eval_coop
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_thought_v1_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_thought_v1_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Evaluate Cooperative LoRA + Thought v1
# ============================================
# Steps:
#   1. Merge cooperative LoRA into base model
#   2. Serve merged model via vLLM
#   3. Run GUI-360 evaluation (grounding + action_prediction)
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
EVAL_DIR="$TRAIN_DIR/GUI-360-eval"
DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"

BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
COOP_CKPT="$TRAIN_DIR/llamafactory/output/cooperative_thought_v1/final"
MERGED_MODEL="$PROJECT_DIR/checkpoints/cooperative_thought_v1_merged"
RESULTS_DIR="$EVAL_DIR/results/cooperative_thought_v1"

MODEL_TYPE="qwen2.5_vl_7b"
VLLM_PORT=19820
VLLM_HOST="localhost"
EVAL_THREADS=5

echo "=========================================="
echo "Cooperative LoRA + Thought v1 Evaluation"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# ── Step 1: Merge LoRA into base model ──

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

if [ ! -d "$MERGED_MODEL" ] || [ ! -f "$MERGED_MODEL/model.safetensors.index.json" ]; then
    echo "Merging cooperative LoRA into base model..."
    python "$PROJECT_DIR/evaluation/merge_cooperative_lora.py" \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$COOP_CKPT" \
        --output_dir "$MERGED_MODEL"
    echo "Merge complete."
else
    echo "Merged model already exists, skipping merge."
fi

# ── Step 2: Serve via vLLM ──

echo "=========================================="
echo "Starting vLLM Server (port $VLLM_PORT)"
echo "=========================================="

# Switch to eval env for vLLM
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 3

python -m vllm.entrypoints.openai.api_server \
    --model "$MERGED_MODEL" \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee "$PROJECT_DIR/scripts/exp_cooperative/logs/vllm_coop_v1_${SLURM_JOB_ID}.log" &

VLLM_PID=$!
echo "vLLM server started (PID=$VLLM_PID)"

# Wait for server
MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://${VLLM_HOST}:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "vLLM server ready!"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# ── Step 3: Evaluate ──

echo "=========================================="
echo "Running Evaluations"
echo "=========================================="

mkdir -p "$RESULTS_DIR"
cd "$EVAL_DIR"

for EVAL_TYPE in grounding action_prediction; do
    echo "Running $EVAL_TYPE..."
    python evaluation.py \
        --root_dir "$DATASET_ROOT" \
        --type "$EVAL_TYPE" \
        --model_type "$MODEL_TYPE" \
        --model_name "$MERGED_MODEL" \
        --api_url "http://${VLLM_HOST}:${VLLM_PORT}/v1" \
        --threads $EVAL_THREADS \
        --output_dir "${RESULTS_DIR}/${EVAL_TYPE}"
    echo "$EVAL_TYPE done."
done

# Cleanup
pkill -f "vllm.entrypoints.openai.api_server" || true

echo "=========================================="
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="
echo "Summaries:"
cat $RESULTS_DIR/*/evaluation_summary_*.json 2>/dev/null || echo "(no summaries)"
echo ""
echo "Done: $(date)"
echo "=========================================="
