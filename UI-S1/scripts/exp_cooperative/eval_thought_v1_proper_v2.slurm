#!/bin/bash
#SBATCH --job-name=eval_coop_prop
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_proper_v2_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_proper_v2_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Proper routing eval via cooperative API server
# Uses GUI-360 eval framework (correct prompts + parsing)
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
EVAL_DIR="$TRAIN_DIR/GUI-360-eval"
DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"

BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
COOP_CKPT="$TRAIN_DIR/llamafactory/output/cooperative_thought_v1/final"
RESULTS_DIR="$EVAL_DIR/results/cooperative_thought_v1_proper_routing"

MODEL_TYPE="qwen2.5_vl_7b"
SERVER_PORT=19820
SERVER_HOST="localhost"
NUM_GPUS=4
EVAL_THREADS=4  # one thread per GPU

echo "=========================================="
echo "Cooperative LoRA v1 — Proper Routing Eval"
echo "Using GUI-360 eval framework"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

cd "$PROJECT_DIR"

# ── Step 1: Start cooperative model server ──

echo "Starting cooperative model server..."
python evaluation/serve_cooperative.py \
    --base_model "$BASE_MODEL" \
    --coop_checkpoint "$COOP_CKPT" \
    --port $SERVER_PORT \
    --num_gpus $NUM_GPUS \
    2>&1 | tee scripts/exp_cooperative/logs/serve_coop_${SLURM_JOB_ID}.log &

SERVER_PID=$!
echo "Server started (PID=$SERVER_PID)"

# Wait for server
MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://${SERVER_HOST}:${SERVER_PORT}/health > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: Server failed to start"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# ── Step 2: Run GUI-360 evaluation ──

echo "=========================================="
echo "Running Evaluations via GUI-360 framework"
echo "=========================================="

mkdir -p "$RESULTS_DIR"

# Switch to eval env for GUI-360 eval dependencies
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

cd "$EVAL_DIR"

for EVAL_TYPE in action_prediction grounding; do
    echo ""
    echo "=== $EVAL_TYPE ==="
    python evaluation.py \
        --root_dir "$DATASET_ROOT" \
        --type "$EVAL_TYPE" \
        --model_type "$MODEL_TYPE" \
        --model_name "cooperative" \
        --api_url "http://${SERVER_HOST}:${SERVER_PORT}/v1" \
        --threads $EVAL_THREADS \
        --output_dir "${RESULTS_DIR}/${EVAL_TYPE}"
    echo "$EVAL_TYPE done."
done

# Cleanup
kill $SERVER_PID 2>/dev/null || true

echo "=========================================="
echo "Results:"
for f in $RESULTS_DIR/*/evaluation_summary_*.json; do
    echo "--- $(basename $(dirname $f)) ---"
    cat "$f" 2>/dev/null
done
echo ""
echo "Done: $(date)"
echo "=========================================="
