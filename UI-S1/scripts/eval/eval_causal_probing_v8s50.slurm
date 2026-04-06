#!/bin/bash
#SBATCH --job-name=causal_probe
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/causal_probe_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/causal_probe_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ============================================
# Causal Probing: What does V8 extract from s_{t+1}?
# Direction C from visual_idm_exploration.md §13
#
# 5 ablation conditions on hindsight mode:
#   blur         — Gaussian blur σ=20 (preserves layout, loses detail)
#   lowres       — 32×32 → resize back (only color distribution)
#   random       — Screenshot from different episode (wrong s_{t+1})
#   copy_st      — s_t duplicated as s_{t+1} (zero transition info)
#   same_ep_rand — Random screenshot from same episode (same app, wrong step)
#
# Baseline (full s_{t+1}) already done: 66.0% type_match (V8 s70)
# ============================================

set -e

source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="${PROJECT_DIR}/evaluation"
SCRIPT_DIR="${PROJECT_DIR}/scripts/eval"

MODEL_DIR="${PROJECT_DIR}/checkpoints/sp_gigpo_ac/sp_gigpo_spwa_k8_v8/merged_step_70"
MODEL_NAME="v8_step70"
DATASET_PATH="${EVAL_DIR}/dataset/android_control_evaluation_fixed.jsonl"
OUTPUT_DIR="${EVAL_DIR}/results/visual_idm"
MAX_SAMPLES=500

mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "Causal Probing: V8 Step 50 Hindsight Ablation"
echo "=========================================="
echo "Model: $MODEL_DIR"
echo "Start time: $(date)"
echo "=========================================="

# Fix dataset paths if needed
if [ ! -f "$DATASET_PATH" ]; then
    sed 's|/datasets/|/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/|g' \
        "${EVAL_DIR}/dataset/android_control_evaluation_std.jsonl" > "$DATASET_PATH"
fi

# Start vLLM server (reused across all ablation conditions)
vllm serve "$MODEL_DIR" \
    --served-model-name $MODEL_NAME \
    --tensor-parallel-size 4 \
    --limit-mm-per-prompt image=3 \
    --port 8000 \
    --trust-remote-code \
    &

VLLM_PID=$!

MAX_RETRIES=60
RETRY_COUNT=0
while ! curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 10
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: vLLM server failed to start"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
done
echo "vLLM server ready!"

cd $PROJECT_DIR

# Run 5 ablation conditions sequentially (all hindsight mode)
for ABLATION in blur lowres random copy_st same_ep_rand; do
    echo ""
    echo "================================================"
    echo "Ablation: $ABLATION ($(date))"
    echo "================================================"
    python ${SCRIPT_DIR}/eval_visual_idm.py \
        --model_name $MODEL_NAME \
        --jsonl_file $DATASET_PATH \
        --output_dir $OUTPUT_DIR \
        --mode hindsight \
        --max_samples $MAX_SAMPLES \
        --max_workers 128 \
        --ablation $ABLATION
done

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo ""
echo "=========================================="
echo "All causal probing experiments completed at $(date)"
echo "=========================================="
echo "Results in: $OUTPUT_DIR/vidm_hindsight_v8_step70_*.json"
ls -la $OUTPUT_DIR/vidm_hindsight_v8_step70_*
