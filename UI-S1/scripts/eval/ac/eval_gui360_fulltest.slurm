#!/bin/bash
#SBATCH --job-name=g360_eval
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/eval_gui360_fulltest_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/eval_gui360_fulltest_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:4

set -e
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
MODEL_BASE="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
MODEL_SFT="$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_full_sft_v2"
EVAL_DATA="$PROJECT_ROOT/train/p_emergent/gui360_eval_data_fulltest.pkl"
RESULTS_DIR="$PROJECT_ROOT/outputs/gui360_eval_results_fulltest"

cd "$PROJECT_ROOT"
mkdir -p "$RESULTS_DIR"
mkdir -p scripts/eval/ac/logs

# Phase 0: Build full test eval data (no GPU needed)
echo "===== Phase 0: Build full test eval pickle ====="
python scripts/eval/ac/build_gui360_full_eval_data.py \
    --input "$PROJECT_ROOT/datasets/GUI-360/rl_data/gui360_test.jsonl" \
    --output "$EVAL_DATA"

# Phase 1: Start vLLM for base model + evaluate
echo "===== Phase 1: Evaluate baseline (base Qwen2.5-VL-7B) ====="
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_BASE" \
    --served-model-name baseline \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for server to be ready
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM ready after ${i}s"
        break
    fi
    sleep 5
done

python train/p_emergent/eval_gui360_spwa.py \
    --step eval \
    --condition baseline \
    --model_name baseline \
    --eval_data "$EVAL_DATA" \
    --results_dir "$RESULTS_DIR" \
    --endpoint http://localhost:8000/v1 \
    --max_workers 32

kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
sleep 10

# Phase 2: Start vLLM for SFT model + evaluate
echo "===== Phase 2: Evaluate SFT (gui360_full_sft_v2) ====="
if [ -d "$MODEL_SFT" ]; then
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_SFT" \
        --served-model-name sft \
        --port 8000 \
        --tensor-parallel-size 4 \
        --max-model-len 32768 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"

    for i in $(seq 1 60); do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "vLLM ready after ${i}s"
            break
        fi
        sleep 5
    done

    python train/p_emergent/eval_gui360_spwa.py \
        --step eval \
        --condition sft \
        --model_name sft \
        --eval_data "$EVAL_DATA" \
        --results_dir "$RESULTS_DIR" \
        --endpoint http://localhost:8000/v1 \
        --max_workers 32

    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
else
    echo "SFT model not found at $MODEL_SFT, skipping"
fi

echo "===== All evaluations complete ====="
echo "Results in: $RESULTS_DIR"
ls -la "$RESULTS_DIR"
