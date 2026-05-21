#!/bin/bash
#SBATCH --job-name=eval_coop_g360
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_coop_gui360_vllm_lossless_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_coop_gui360_vllm_lossless_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Lossless eval of Cooperative v6.5 on GUI-360 AR trajectory
# 1) Lossless merge (fold LoRA_A into base + correction adapters)
# 2) Start corrected vLLM server
# 3) Run GUI-360 AR trajectory eval (stop + nostop)
#
# Usage: COOP_EPOCH=4 sbatch this_script.slurm
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
COOP_EPOCH="${COOP_EPOCH:-4}"
COOP_CKPT="$PROJECT_ROOT/train_GUI_360/llamafactory/output/cooperative_v6_5_comm_thought/epoch-${COOP_EPOCH}"
MERGED_MODEL="/tmp/coop_v65_gui360_lossless_ep${COOP_EPOCH}_${SLURM_JOB_ID}"
GUI360_DIR="$PROJECT_ROOT/train_GUI_360/GUI-360-eval"
GUI360_DATA="$PROJECT_ROOT/datasets/GUI-360/test"
OUTPUT_BASE="$PROJECT_ROOT/scripts/eval/results/gui360"
EVAL_WORKERS="${EVAL_WORKERS:-128}"

export TMPDIR=/tmp/${SLURM_JOB_ID}
mkdir -p $TMPDIR
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor
export TORCH_HOME=$TMPDIR/torch
export XDG_CACHE_HOME=$TMPDIR/cache
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TORCH_HOME $XDG_CACHE_HOME

echo "=========================================="
echo "Eval: Cooperative v6.5 Lossless on GUI-360 (epoch ${COOP_EPOCH})"
echo "Checkpoint: $COOP_CKPT"
echo "Merged:     $MERGED_MODEL"
echo "Workers:    $EVAL_WORKERS"
echo "Start:      $(date)"
echo "=========================================="

# Step 1: Lossless merge
echo "Running lossless merge..."
cd "$PROJECT_ROOT"
python evaluation/merge_cooperative_lossless.py \
    --base_model "$BASE_MODEL" \
    --coop_checkpoint "$COOP_CKPT" \
    --output_dir "$MERGED_MODEL"

MERGE_EXIT=$?
if [ $MERGE_EXIT -ne 0 ]; then
    echo "ERROR: Lossless merge failed with exit code $MERGE_EXIT"
    exit $MERGE_EXIT
fi
echo "Lossless merge complete."

# Step 2: Start corrected vLLM server
echo "Starting corrected vLLM server..."
python evaluation/serve_corrected_vllm.py \
    --model "$MERGED_MODEL" \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image": 2}' \
    &
VLLM_PID=$!

MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM server ready."
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    echo "Waiting for vLLM server... ($WAIT_COUNT/$MAX_WAIT seconds)"
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null
    rm -rf "$MERGED_MODEL"
    exit 1
fi

cd "$GUI360_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
API_URL="http://localhost:8000/v1"

# Step 3a: Stop-on-Error AR eval
STOP_OUTPUT="$OUTPUT_BASE/coop_v65_ep${COOP_EPOCH}_lossless_stop_${TIMESTAMP}"
echo ""
echo "=== Stop-on-Error AR Evaluation ==="
echo "Output: $STOP_OUTPUT"

python evaluation.py \
    --root_dir "$GUI360_DATA" \
    --model_name "$MERGED_MODEL" \
    --model_type "qwen2.5_vl_7b" \
    --type "action_prediction_ar" \
    --threads "$EVAL_WORKERS" \
    --output_dir "$STOP_OUTPUT" \
    --api_url "$API_URL" \
    --ar_stop_on_error

echo "Stop-on-error completed."

# Step 3b: No-Stop AR eval
NOSTOP_OUTPUT="$OUTPUT_BASE/coop_v65_ep${COOP_EPOCH}_lossless_nostop_${TIMESTAMP}"
echo ""
echo "=== No-Stop AR Evaluation ==="
echo "Output: $NOSTOP_OUTPUT"

python evaluation.py \
    --root_dir "$GUI360_DATA" \
    --model_name "$MERGED_MODEL" \
    --model_type "qwen2.5_vl_7b" \
    --type "action_prediction_ar" \
    --threads "$EVAL_WORKERS" \
    --output_dir "$NOSTOP_OUTPUT" \
    --api_url "$API_URL" \
    --ar_no_stop_on_error

echo "No-stop completed."

# Cleanup
kill $VLLM_PID 2>/dev/null || true
pkill -f "serve_corrected_vllm" || true
rm -rf "$MERGED_MODEL"

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "Stop results: $STOP_OUTPUT"
echo "NoStop results: $NOSTOP_OUTPUT"
echo "=========================================="
