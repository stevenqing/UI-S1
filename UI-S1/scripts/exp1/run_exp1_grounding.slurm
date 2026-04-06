#!/bin/bash
#SBATCH --job-name=exp1_gnd
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/exp1_gnd_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/exp1_gnd_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================================
# Phase 1 Grounding Experiments: Exp 1.1 + 1.2
#
# Runs multi-sample grounding eval for both SFT v3 and SFT v2
# Uses a single vLLM server (switched between models)
#
# Expected runtime: ~12-18 hours total
# ============================================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
LOG_DIR="$PROJECT_DIR/train_GUI_360/logs"
mkdir -p $LOG_DIR

# Model paths
SFT_V3_MODEL="$PROJECT_DIR/train_GUI_360/llamafactory/output/gui360_full_sft_v3_grounding"
SFT_V2_MODEL="$PROJECT_DIR/train_GUI_360/llamafactory/output/gui360_full_sft_v2"

VLLM_PORT=19815
VLLM_HOST="localhost"

echo "=========================================="
echo "Phase 1 Grounding: Exp 1.1 + 1.2"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

# Verify models exist
for MODEL in "$SFT_V3_MODEL" "$SFT_V2_MODEL"; do
    if [ ! -d "$MODEL" ]; then
        echo "ERROR: Model not found at $MODEL"
        exit 1
    fi
done

start_vllm() {
    local MODEL_PATH=$1
    local MODEL_NAME=$(basename $MODEL_PATH)
    echo "Starting vLLM with $MODEL_NAME on port $VLLM_PORT ..."

    pkill -f "vllm.entrypoints.openai.api_server" || true
    sleep 5

    python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH \
        --port $VLLM_PORT \
        --tensor-parallel-size 4 \
        --max-model-len 32768 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        2>&1 | tee $LOG_DIR/vllm_exp1_${MODEL_NAME}_${SLURM_JOB_ID}.log &

    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"

    MAX_WAIT=600
    WAIT_COUNT=0
    while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
        if curl -s http://${VLLM_HOST}:${VLLM_PORT}/health > /dev/null 2>&1; then
            echo "vLLM ready!"
            return 0
        fi
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
        echo "Waiting... ($WAIT_COUNT/$MAX_WAIT)"
    done

    echo "ERROR: vLLM failed to start"
    kill $VLLM_PID 2>/dev/null || true
    return 1
}

stop_vllm() {
    echo "Stopping vLLM..."
    pkill -f "vllm.entrypoints.openai.api_server" || true
    sleep 10
}

ENDPOINT="http://${VLLM_HOST}:${VLLM_PORT}/v1"
cd $PROJECT_DIR

# ============================================================
# Experiment 1.1: SFT v3 Multi-Sample Grounding
# ============================================================
echo ""
echo "=========================================="
echo "Exp 1.1: SFT v3 Multi-Sample Grounding"
echo "=========================================="

start_vllm "$SFT_V3_MODEL"

# K=1 (greedy baseline for comparison)
echo "--- K=1 (greedy) ---"
python scripts/exp1/exp1_1_sft_v3_multisample.py \
    --model_name "$SFT_V3_MODEL" \
    --endpoint "$ENDPOINT" \
    --K 1 --temperature 0.0 \
    --output_dir outputs/exp1_1

# K=5
echo "--- K=5 ---"
python scripts/exp1/exp1_1_sft_v3_multisample.py \
    --model_name "$SFT_V3_MODEL" \
    --endpoint "$ENDPOINT" \
    --K 5 --temperature 0.7 \
    --output_dir outputs/exp1_1

# K=10
echo "--- K=10 ---"
python scripts/exp1/exp1_1_sft_v3_multisample.py \
    --model_name "$SFT_V3_MODEL" \
    --endpoint "$ENDPOINT" \
    --K 10 --temperature 0.7 \
    --output_dir outputs/exp1_1

# Analyze all K values
python scripts/exp1/exp1_1_sft_v3_multisample.py --analyze_only --results_dir outputs/exp1_1

stop_vllm

# ============================================================
# Experiment 1.2: SFT v2 Multi-Sample Grounding
# ============================================================
echo ""
echo "=========================================="
echo "Exp 1.2: SFT v2 Multi-Sample Grounding"
echo "=========================================="

start_vllm "$SFT_V2_MODEL"

# K=1
echo "--- K=1 (greedy) ---"
python scripts/exp1/exp1_2_sft_v2_multisample.py \
    --model_name "$SFT_V2_MODEL" \
    --endpoint "$ENDPOINT" \
    --K 1 --temperature 0.0 \
    --output_dir outputs/exp1_2

# K=5
echo "--- K=5 ---"
python scripts/exp1/exp1_2_sft_v2_multisample.py \
    --model_name "$SFT_V2_MODEL" \
    --endpoint "$ENDPOINT" \
    --K 5 --temperature 0.7 \
    --output_dir outputs/exp1_2

# K=10
echo "--- K=10 ---"
python scripts/exp1/exp1_2_sft_v2_multisample.py \
    --model_name "$SFT_V2_MODEL" \
    --endpoint "$ENDPOINT" \
    --K 10 --temperature 0.7 \
    --output_dir outputs/exp1_2

# Compare V3 vs V2
python scripts/exp1/exp1_2_sft_v2_multisample.py \
    --analyze_only --results_dir outputs/exp1_2 --exp1_1_dir outputs/exp1_1

stop_vllm

# ============================================================
# Experiment 1.6: Error Diversity (CPU only, from K=1 results)
# ============================================================
echo ""
echo "=========================================="
echo "Exp 1.6: Error Diversity Analysis"
echo "=========================================="

python scripts/exp1/exp1_6_error_diversity.py \
    --v3_exp1_results outputs/exp1_1/results_K1.jsonl \
    --v2_exp1_results outputs/exp1_2/results_K1.jsonl \
    --output_dir outputs/exp1_6

echo ""
echo "=========================================="
echo "All Grounding Experiments Complete!"
echo "=========================================="
echo "End time: $(date)"
echo "Results:"
echo "  Exp 1.1: outputs/exp1_1/"
echo "  Exp 1.2: outputs/exp1_2/"
echo "  Exp 1.6: outputs/exp1_6/"
echo "=========================================="
