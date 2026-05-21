#!/bin/bash
#SBATCH --job-name=v16_ta
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v16_task_vector_merge/scripts/logs/eval_task_arith_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v16_task_vector_merge/scripts/logs/eval_task_arith_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
MODEL_A="$PROJECT_ROOT/checkpoints/gui_grounding"
MODEL_B="$PROJECT_ROOT/checkpoints/gui_action"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
MERGE_SCRIPT="$PROJECT_ROOT/v16_task_vector_merge/merge_task_vectors.py"
EVAL_SCRIPT="$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py"
CKPT_PREFIX="$PROJECT_ROOT/checkpoints/v16"
EVAL_PREFIX="$PROJECT_ROOT/v16_task_vector_merge/outputs"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=========================================="
echo "V16 Eval: Task Arithmetic (4 configs)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start:  $(date)"
echo "=========================================="

eval_merged_model() {
    local MODEL_DIR=$1
    local MODEL_NAME=$2
    local EVAL_DIR=$3

    mkdir -p "$EVAL_DIR"

    if ls "$EVAL_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
        echo "  Eval already done for $MODEL_NAME, skipping."
        return 0
    fi

    echo "  Serving $MODEL_NAME from $MODEL_DIR"
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_DIR" \
        --served-model-name "$MODEL_NAME" \
        --port 8000 \
        --tensor-parallel-size 4 \
        --max-model-len 32768 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --limit-mm-per-prompt '{"image": 2}' \
        &
    local VLLM_PID=$!

    local WAIT_COUNT=0
    while [ $WAIT_COUNT -lt 600 ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "  vLLM ready after ${WAIT_COUNT}s"
            break
        fi
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
    done

    if [ $WAIT_COUNT -ge 600 ]; then
        echo "  ERROR: vLLM server failed to start for $MODEL_NAME"
        kill $VLLM_PID 2>/dev/null || true
        return 1
    fi

    python "$EVAL_SCRIPT" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$EVAL_DIR" \
        --threads 128

    echo "  Results for $MODEL_NAME:"
    cat "$EVAL_DIR"/eval_summary_*.json 2>/dev/null || true

    echo "  Shutting down vLLM..."
    kill $VLLM_PID 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    sleep 10
}

run_config() {
    local SUFFIX=$1
    shift
    local MERGED_DIR="${CKPT_PREFIX}_${SUFFIX}"
    local EVAL_DIR="${EVAL_PREFIX}/eval_${SUFFIX}"
    local MODEL_NAME="v16_${SUFFIX}"

    echo ""
    echo "=========================================="
    echo "Config: $SUFFIX"
    echo "Time:   $(date)"
    echo "=========================================="

    if [ -f "$MERGED_DIR/config.json" ]; then
        echo "  Merge already exists: $MERGED_DIR"
    else
        echo "  Merging..."
        python "$MERGE_SCRIPT" \
            --base_model "$BASE_MODEL" \
            --model_a "$MODEL_A" \
            --model_b "$MODEL_B" \
            --output_dir "$MERGED_DIR" \
            "$@"
    fi

    eval_merged_model "$MERGED_DIR" "$MODEL_NAME" "$EVAL_DIR"
}

# ── Task Arithmetic: 4 configs ──

run_config "ta_03_07" --method task_arith --alpha 0.3 --beta 0.7
run_config "ta_05_05" --method task_arith --alpha 0.5 --beta 0.5
run_config "ta_07_03" --method task_arith --alpha 0.7 --beta 0.3
run_config "ta_10_10" --method task_arith --alpha 1.0 --beta 1.0

echo ""
echo "=========================================="
echo "Task Arithmetic: ALL 4 CONFIGS COMPLETE"
echo "Done: $(date)"
echo "=========================================="
