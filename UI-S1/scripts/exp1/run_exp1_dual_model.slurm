#!/bin/bash
#SBATCH --job-name=exp1_dual
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/exp1_dual_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/exp1_dual_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================================
# Phase 1 Dual-Model Experiments: Exp 1.3 + 1.4 + 1.5
#
# Runs two vLLM servers (SFT v2 on node 0, SFT v3 on node 1)
# Then runs dual-model experiments from node 0
#
# Requires 2 nodes with 4 GPUs each
# Expected runtime: ~12-18 hours
# ============================================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
LOG_DIR="$PROJECT_DIR/train_GUI_360/logs"
mkdir -p $LOG_DIR

SFT_V3_MODEL="$PROJECT_DIR/train_GUI_360/llamafactory/output/gui360_full_sft_v3_grounding"
SFT_V2_MODEL="$PROJECT_DIR/train_GUI_360/llamafactory/output/gui360_full_sft_v2"

V2_PORT=19816
V3_PORT=19815

echo "=========================================="
echo "Phase 1 Dual-Model: Exp 1.3 + 1.4 + 1.5"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

# Get node list
NODELIST=$(scontrol show hostname $SLURM_NODELIST)
NODE0=$(echo "$NODELIST" | head -1)
NODE1=$(echo "$NODELIST" | tail -1)

echo "Node 0 (SFT v2): $NODE0"
echo "Node 1 (SFT v3): $NODE1"

# ============================================================
# Start both vLLM servers
# ============================================================

# Start SFT v2 on Node 0 (current node)
echo "Starting SFT v2 vLLM on $NODE0:$V2_PORT ..."
python -m vllm.entrypoints.openai.api_server \
    --model $SFT_V2_MODEL \
    --port $V2_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $LOG_DIR/vllm_exp1_v2_${SLURM_JOB_ID}.log &
V2_PID=$!

# Start SFT v3 on Node 1
echo "Starting SFT v3 vLLM on $NODE1:$V3_PORT ..."
srun --nodes=1 --ntasks=1 -w $NODE1 \
    python -m vllm.entrypoints.openai.api_server \
    --model $SFT_V3_MODEL \
    --port $V3_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $LOG_DIR/vllm_exp1_v3_${SLURM_JOB_ID}.log &
V3_PID=$!

# Wait for both servers
V2_ENDPOINT="http://${NODE0}:${V2_PORT}/v1"
V3_ENDPOINT="http://${NODE1}:${V3_PORT}/v1"

echo "Waiting for both vLLM servers..."
MAX_WAIT=600
for endpoint_label in "V2:${NODE0}:${V2_PORT}" "V3:${NODE1}:${V3_PORT}"; do
    IFS=':' read -r label host port <<< "$endpoint_label"
    WAIT_COUNT=0
    while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
        if curl -s http://${host}:${port}/health > /dev/null 2>&1; then
            echo "$label server ready on ${host}:${port}"
            break
        fi
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
    done
    if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
        echo "ERROR: $label vLLM server failed to start"
        kill $V2_PID $V3_PID 2>/dev/null || true
        exit 1
    fi
done

echo "Both vLLM servers ready!"

cd $PROJECT_DIR

# ============================================================
# Experiment 1.4: SKIPPED (already completed)
# ============================================================
echo ""
echo "Exp 1.4: SKIPPED (already completed - see outputs/exp1_4/)"

# ============================================================
# Experiment 1.3: Oracle Coordinate Replacement (Go/No-Go)
# ============================================================
echo ""
echo "=========================================="
echo "Exp 1.3: Oracle Coordinate Replacement"
echo "=========================================="

python scripts/exp1/exp1_3_oracle_coord_replacement.py \
    --sft_v2_model "$SFT_V2_MODEL" \
    --sft_v3_model "$SFT_V3_MODEL" \
    --sft_v2_endpoint "$V2_ENDPOINT" \
    --sft_v3_endpoint "$V3_ENDPOINT" \
    --output_dir outputs/exp1_3

# ============================================================
# Experiment 1.5: Dual-Model Combination
# ============================================================
echo ""
echo "=========================================="
echo "Exp 1.5: Dual-Model Combination"
echo "=========================================="

python scripts/exp1/exp1_5_dual_model_eval.py \
    --sft_v2_model "$SFT_V2_MODEL" \
    --sft_v3_model "$SFT_V3_MODEL" \
    --sft_v2_endpoint "$V2_ENDPOINT" \
    --sft_v3_endpoint "$V3_ENDPOINT" \
    --K 5 --temperature 0.7 \
    --output_dir outputs/exp1_5

# ============================================================
# Cleanup
# ============================================================
echo ""
echo "Stopping vLLM servers..."
kill $V2_PID $V3_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true

echo ""
echo "=========================================="
echo "All Dual-Model Experiments Complete!"
echo "=========================================="
echo "End time: $(date)"
echo "Results:"
echo "  Exp 1.3: outputs/exp1_3/ (Go/No-Go Gate)"
echo "  Exp 1.4: outputs/exp1_4/"
echo "  Exp 1.5: outputs/exp1_5/"
echo "=========================================="
