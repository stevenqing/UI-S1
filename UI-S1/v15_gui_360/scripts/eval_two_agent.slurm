#!/bin/bash
#SBATCH --job-name=ev_2agent
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/eval_two_agent_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/eval_two_agent_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Two-Agent Eval: Reasoner (CoT SFT) + Actor (Base SFT)
#
# Agent 1 (Reasoner): CoT SFT model on GPU 0,1 (port 8000)
#   - Generates reasoning: CURRENT STATE, SUB-GOAL, LOCATE
# Agent 2 (Actor): Base SFT checkpoint-272 on GPU 2,3 (port 8001)
#   - Takes reasoning as guidance, predicts precise action
#
# Test: gui360_test_1000_balanced.jsonl (1000 episodes)
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
REASONER_CKPT="$PROJECT_ROOT/train_GUI_360/llamafactory/output/cot_sft_full"
ACTOR_CKPT="$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_balanced_full_sft/checkpoint-272"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/v15_gui_360/outputs/eval_two_agent"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_ROOT/v15_gui_360/scripts/logs"

echo "=========================================="
echo "Two-Agent Eval: Reasoner + Actor"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "Reasoner: $REASONER_CKPT"
echo "Actor:    $ACTOR_CKPT"
echo "=========================================="

# Start Reasoner vLLM on GPU 0,1 (port 8000)
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model "$REASONER_CKPT" \
    --served-model-name "cot_reasoner" \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image": 2}' \
    &
REASONER_PID=$!

# Start Actor vLLM on GPU 2,3 (port 8001)
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
    --model "$ACTOR_CKPT" \
    --served-model-name "base_actor" \
    --port 8001 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image": 2}' \
    &
ACTOR_PID=$!

# Wait for both servers
echo "Waiting for vLLM servers..."
WAIT_COUNT=0
REASONER_READY=0
ACTOR_READY=0
while [ $WAIT_COUNT -lt 600 ]; do
    if [ $REASONER_READY -eq 0 ] && curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Reasoner ready after ${WAIT_COUNT}s"
        REASONER_READY=1
    fi
    if [ $ACTOR_READY -eq 0 ] && curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo "Actor ready after ${WAIT_COUNT}s"
        ACTOR_READY=1
    fi
    if [ $REASONER_READY -eq 1 ] && [ $ACTOR_READY -eq 1 ]; then
        echo "Both servers ready!"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
done

if [ $REASONER_READY -eq 0 ] || [ $ACTOR_READY -eq 0 ]; then
    echo "ERROR: Server(s) failed to start"
    kill $REASONER_PID $ACTOR_PID 2>/dev/null || true
    exit 1
fi

# Run two-agent eval
python "$PROJECT_ROOT/v15_gui_360/eval_two_agent.py" \
    --test_data "$TEST_DATA" \
    --reasoner_url "http://localhost:8000/v1" \
    --reasoner_model "cot_reasoner" \
    --actor_url "http://localhost:8001/v1" \
    --actor_model "base_actor" \
    --output_dir "$OUTPUT_DIR" \
    --threads 64

# Shutdown servers
echo "Shutting down vLLM servers..."
kill $REASONER_PID $ACTOR_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
wait $REASONER_PID 2>/dev/null || true
wait $ACTOR_PID 2>/dev/null || true

# Print summary
SUMMARY=$(ls "$OUTPUT_DIR"/eval_summary_*.json 2>/dev/null | tail -1)
if [ -n "$SUMMARY" ]; then
    echo ""
    echo "=========================================="
    echo "Result:"
    cat "$SUMMARY"
    echo ""
    echo "=========================================="
fi

echo "Done: $(date)"
