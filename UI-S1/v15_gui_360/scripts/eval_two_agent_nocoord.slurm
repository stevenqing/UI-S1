#!/bin/bash
#SBATCH --job-name=ev_2a_noc
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/eval_two_agent_nocoord_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/eval_two_agent_nocoord_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Two-Agent Eval: Strip coordinates from reasoning
#
# Hypothesis: CoT reasoner outputs [x,y] coords → actor copies
# blindly → loses its own grounding ability (70% → 59%).
#
# Fix: strip coords from reasoning, keep semantic info only.
# Actor must look at the image to find elements itself.
#
# Experiments:
#   1. CoT SFT reasoner + Base actor (strip coords)
#   2. Base SFT as both reasoner + actor (strip coords)
#      (= single model two-pass, like cot_step but fair comparison)
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
COT_SFT_CKPT="$PROJECT_ROOT/train_GUI_360/llamafactory/output/cot_sft_full"
BASE_SFT_CKPT="$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_balanced_full_sft/checkpoint-272"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p "$PROJECT_ROOT/v15_gui_360/scripts/logs"

echo "=========================================="
echo "Two-Agent Eval: No-Coord Reasoning"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

# ─── Experiment 1: CoT SFT reasoner (no coords) + Base actor ───
OUTPUT_DIR_1="$PROJECT_ROOT/v15_gui_360/outputs/eval_two_agent_nocoord"

if ls "$OUTPUT_DIR_1"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[Exp 1] Already done, skipping."
else
    echo ""
    echo "=== Exp 1: CoT SFT reasoner (strip coords) + Base actor ==="
    mkdir -p "$OUTPUT_DIR_1"

    # Start CoT SFT reasoner on GPU 0,1
    CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
        --model "$COT_SFT_CKPT" \
        --served-model-name "cot_reasoner" \
        --port 8000 \
        --tensor-parallel-size 2 \
        --max-model-len 32768 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --limit-mm-per-prompt '{"image": 2}' \
        &
    REASONER_PID=$!

    # Start Base actor on GPU 2,3
    CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
        --model "$BASE_SFT_CKPT" \
        --served-model-name "base_actor" \
        --port 8001 \
        --tensor-parallel-size 2 \
        --max-model-len 32768 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --limit-mm-per-prompt '{"image": 2}' \
        &
    ACTOR_PID=$!

    # Wait for servers
    WAIT_COUNT=0
    R_READY=0; A_READY=0
    while [ $WAIT_COUNT -lt 600 ]; do
        [ $R_READY -eq 0 ] && curl -s http://localhost:8000/health > /dev/null 2>&1 && R_READY=1 && echo "Reasoner ready (${WAIT_COUNT}s)"
        [ $A_READY -eq 0 ] && curl -s http://localhost:8001/health > /dev/null 2>&1 && A_READY=1 && echo "Actor ready (${WAIT_COUNT}s)"
        [ $R_READY -eq 1 ] && [ $A_READY -eq 1 ] && break
        sleep 5; WAIT_COUNT=$((WAIT_COUNT + 5))
    done

    if [ $R_READY -eq 0 ] || [ $A_READY -eq 0 ]; then
        echo "ERROR: Server(s) failed"
        kill $REASONER_PID $ACTOR_PID 2>/dev/null || true
    else
        python "$PROJECT_ROOT/v15_gui_360/eval_two_agent.py" \
            --test_data "$TEST_DATA" \
            --reasoner_url "http://localhost:8000/v1" \
            --reasoner_model "cot_reasoner" \
            --actor_url "http://localhost:8001/v1" \
            --actor_model "base_actor" \
            --output_dir "$OUTPUT_DIR_1" \
            --threads 64 \
            --strip_coords
    fi

    kill $REASONER_PID $ACTOR_PID 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    wait $REASONER_PID 2>/dev/null || true
    wait $ACTOR_PID 2>/dev/null || true
    sleep 10
fi

# ─── Experiment 2: Base SFT as both reasoner + actor (strip coords) ───
OUTPUT_DIR_2="$PROJECT_ROOT/v15_gui_360/outputs/eval_two_agent_base_nocoord"

if ls "$OUTPUT_DIR_2"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[Exp 2] Already done, skipping."
else
    echo ""
    echo "=== Exp 2: Base SFT as both reasoner + actor (strip coords) ==="
    mkdir -p "$OUTPUT_DIR_2"

    # Start Base SFT on GPU 0,1 as reasoner
    CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
        --model "$BASE_SFT_CKPT" \
        --served-model-name "base_reasoner" \
        --port 8000 \
        --tensor-parallel-size 2 \
        --max-model-len 32768 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --limit-mm-per-prompt '{"image": 2}' \
        &
    REASONER_PID=$!

    # Start Base SFT on GPU 2,3 as actor
    CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
        --model "$BASE_SFT_CKPT" \
        --served-model-name "base_actor" \
        --port 8001 \
        --tensor-parallel-size 2 \
        --max-model-len 32768 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --limit-mm-per-prompt '{"image": 2}' \
        &
    ACTOR_PID=$!

    # Wait for servers
    WAIT_COUNT=0
    R_READY=0; A_READY=0
    while [ $WAIT_COUNT -lt 600 ]; do
        [ $R_READY -eq 0 ] && curl -s http://localhost:8000/health > /dev/null 2>&1 && R_READY=1 && echo "Reasoner ready (${WAIT_COUNT}s)"
        [ $A_READY -eq 0 ] && curl -s http://localhost:8001/health > /dev/null 2>&1 && A_READY=1 && echo "Actor ready (${WAIT_COUNT}s)"
        [ $R_READY -eq 1 ] && [ $A_READY -eq 1 ] && break
        sleep 5; WAIT_COUNT=$((WAIT_COUNT + 5))
    done

    if [ $R_READY -eq 0 ] || [ $A_READY -eq 0 ]; then
        echo "ERROR: Server(s) failed"
        kill $REASONER_PID $ACTOR_PID 2>/dev/null || true
    else
        python "$PROJECT_ROOT/v15_gui_360/eval_two_agent.py" \
            --test_data "$TEST_DATA" \
            --reasoner_url "http://localhost:8000/v1" \
            --reasoner_model "base_reasoner" \
            --actor_url "http://localhost:8001/v1" \
            --actor_model "base_actor" \
            --output_dir "$OUTPUT_DIR_2" \
            --threads 64 \
            --strip_coords
    fi

    kill $REASONER_PID $ACTOR_PID 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    wait $REASONER_PID 2>/dev/null || true
    wait $ACTOR_PID 2>/dev/null || true
fi

# Print summaries
echo ""
echo "=========================================="
echo "Results Summary"
echo "=========================================="
for d in "$OUTPUT_DIR_1" "$OUTPUT_DIR_2"; do
    S=$(ls "$d"/eval_summary_*.json 2>/dev/null | tail -1)
    if [ -n "$S" ]; then
        echo "$(basename $(dirname $S)):"
        cat "$S"
        echo ""
    fi
done
echo "Done: $(date)"
