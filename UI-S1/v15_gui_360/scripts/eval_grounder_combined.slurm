#!/bin/bash
#SBATCH --job-name=grounder_comb
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/grounder_combined_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/grounder_combined_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Grounder + SFT Model Combined Evaluation
#
# Phase 1: Deploy gui_grounding model → predict coordinates
# Phase 3: Combine SFT action type + grounder coordinates → evaluate
#
# Tests whether a dedicated grounder can improve our SFT model's
# visual grounding accuracy (which is binary: 0.5px or 333px)
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
GROUNDER_PATH="$PROJECT_ROOT/checkpoints/gui_grounding"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/v15_gui_360/outputs/preliminary_tests/grounder_combined"
SFT_RESULTS="$PROJECT_ROOT/v15_gui_360/outputs/history_ablation/gt_history_full/eval_results_*.json"
GROUNDER_PREDS="$OUTPUT_DIR/grounder_predictions.json"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Grounder + SFT Combined Evaluation"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

# ── Phase 1: Deploy grounder and generate predictions ──
echo ""
echo "=========================================="
echo "Phase 1: Grounder Predictions"
echo "Model: gui_grounding"
echo "Time: $(date)"
echo "=========================================="

# Start vLLM server with grounder model
python -m vllm.entrypoints.openai.api_server \
    --model "$GROUNDER_PATH" \
    --served-model-name gui_grounding \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image": 2}' \
    &
VLLM_PID=$!

WAIT_COUNT=0
while [ $WAIT_COUNT -lt 600 ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM ready after ${WAIT_COUNT}s"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
done

if [ $WAIT_COUNT -ge 600 ]; then
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Run Phase 1: generate grounder predictions
python "$PROJECT_ROOT/v15_gui_360/eval_grounder_combined.py" phase1 \
    --test_data "$TEST_DATA" \
    --api_url "http://localhost:8000/v1" \
    --model_name gui_grounding \
    --output_file "$GROUNDER_PREDS" \
    --threads 128

echo "Phase 1 done: $(date)"

# Shutdown grounder vLLM
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
sleep 5

# ── Phase 3: Combine and evaluate ──
echo ""
echo "=========================================="
echo "Phase 3: Combine Grounder + SFT"
echo "Time: $(date)"
echo "=========================================="

python "$PROJECT_ROOT/v15_gui_360/eval_grounder_combined.py" phase3 \
    --grounder_preds "$GROUNDER_PREDS" \
    --sft_results "$SFT_RESULTS" \
    --test_data "$TEST_DATA" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
