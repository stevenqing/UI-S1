#!/bin/bash
#SBATCH --job-name=sft3_s3_ac
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/sft_v3_step3_ac_summary_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/sft_v3_step3_ac_summary_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# SFT v3: AC Summary Context Evaluation
# Three formats: action_level, semantic, progress
# Tests cross-domain transfer
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
SCRIPT_DIR="$PROJECT_DIR/scripts/exp2"
RESULTS_DIR="$SCRIPT_DIR/results/sft_v3/ac"
LOG_DIR="$SCRIPT_DIR/logs"

MODEL_NAME="$PROJECT_DIR/train_GUI_360/llamafactory/output/gui360_full_sft_v3_grounding"
AC_JSONL="$PROJECT_DIR/evaluation/dataset/android_control_evaluation_std.jsonl"

VLLM_PORT=19806
VLLM_HOST="localhost"

echo "=========================================="
echo "SFT v3: AC Summary Context Evaluation"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Model: $MODEL_NAME"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

mkdir -p $LOG_DIR
mkdir -p $RESULTS_DIR

# Start vLLM Server
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $LOG_DIR/vllm_sft3_s3_ac_${SLURM_JOB_ID}.log &

VLLM_PID=$!

MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://${VLLM_HOST}:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    echo "Waiting... ($WAIT_COUNT/$MAX_WAIT seconds)"
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

API_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
cd $PROJECT_DIR

# Format A: Action-level summary
echo ""
echo "=== Format A: action_level ==="
python $SCRIPT_DIR/step3_ac_summary_eval.py \
    --jsonl_file $AC_JSONL \
    --output_dir $RESULTS_DIR \
    --model_name $MODEL_NAME \
    --summary_format action_level \
    --api_url $API_URL \
    --max_workers 4

# Format B: Semantic-level summary
echo ""
echo "=== Format B: semantic_level ==="
python $SCRIPT_DIR/step3_ac_summary_eval.py \
    --jsonl_file $AC_JSONL \
    --output_dir $RESULTS_DIR \
    --model_name $MODEL_NAME \
    --summary_format semantic_level \
    --api_url $API_URL \
    --max_workers 4

# Format C: Progress-level summary
echo ""
echo "=== Format C: progress_level ==="
python $SCRIPT_DIR/step3_ac_summary_eval.py \
    --jsonl_file $AC_JSONL \
    --output_dir $RESULTS_DIR \
    --model_name $MODEL_NAME \
    --summary_format progress_level \
    --api_url $API_URL \
    --max_workers 4

# Cleanup
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true

echo ""
echo "End: $(date)"
echo "Results: $RESULTS_DIR"
echo ""
echo "=== Action-Level Summary ==="
cat $RESULTS_DIR/ac_summary_action_level_summary.json 2>/dev/null || echo "Not found"
echo ""
echo "=== Semantic-Level Summary ==="
cat $RESULTS_DIR/ac_summary_semantic_level_summary.json 2>/dev/null || echo "Not found"
echo ""
echo "=== Progress-Level Summary ==="
cat $RESULTS_DIR/ac_summary_progress_level_summary.json 2>/dev/null || echo "Not found"
