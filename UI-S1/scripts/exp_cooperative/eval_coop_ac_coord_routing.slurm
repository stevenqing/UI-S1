#!/bin/bash
#SBATCH --job-name=eval_ac_cr
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_coop_ac_coord_routing_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_coop_ac_coord_routing_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Eval cooperative LoRA v6.5 + coord_routing + cooperative reasoning on AC
#
# Uses --coord-routing and --coop-reasoning-alpha 0.3 so that:
# - Image + coord/bbox tokens get full correction (1.0)
# - Assistant text tokens get α=0.3 correction weight
#
# Supports INFERENCE_MODE for CA ablation:
#   hard:   img+coord→V(1.0), assistant→α·V+(1-α)·A  (default)
#   v_only: all tokens→V
#   t_only: all tokens→A
#   merge:  0.5*(V+A)
#
# Usage:
#   COOP_EPOCH=4 sbatch eval_coop_ac_coord_routing.slurm
#   COOP_EPOCH=4 INFERENCE_MODE=v_only sbatch eval_coop_ac_coord_routing.slurm
# ============================================

set -e

COOP_EPOCH="${COOP_EPOCH:-4}"
INFERENCE_MODE="${INFERENCE_MODE:-hard}"

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
COOP_CKPT="$PROJECT_ROOT/train_GUI_360/llamafactory/output/cooperative_v6_5_ac_coord_routing/epoch-${COOP_EPOCH}"
MERGED_MODEL="/tmp/merged_lossless_ac_cr_ep${COOP_EPOCH}_${INFERENCE_MODE}_${SLURM_JOB_ID}"
MODEL_NAME="coop_ac_cr_ep${COOP_EPOCH}_${INFERENCE_MODE}"
DATASET_PATH="$PROJECT_ROOT/evaluation/results/android_control_evaluation_fixed.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/outputs/eval_original/ac_coord_routing_ep${COOP_EPOCH}_${INFERENCE_MODE}"

echo "=========================================="
echo "Eval: Cooperative v6.5 + coord_routing AC ep${COOP_EPOCH} mode=${INFERENCE_MODE}"
echo "=========================================="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "COOP_CKPT: $COOP_CKPT"
echo "Output:    $OUTPUT_DIR"
echo "Start:     $(date)"
echo "=========================================="

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p "$OUTPUT_DIR"

# Step 1: Lossless merge
echo "Step 1: Lossless merge..."
python evaluation/merge_cooperative_lossless.py \
    --base_model "$BASE_MODEL" \
    --coop_checkpoint "$COOP_CKPT" \
    --output_dir "$MERGED_MODEL"

MERGE_EXIT=$?
if [ $MERGE_EXIT -ne 0 ]; then
    echo "ERROR: Lossless merge failed with exit code $MERGE_EXIT"
    exit $MERGE_EXIT
fi
echo "Lossless merge complete: $MERGED_MODEL"

# Step 2: Start corrected vLLM server with coord routing
echo "Step 2: Starting vLLM server (mode=$INFERENCE_MODE, coord_routing=True, alpha=0.3)..."
python evaluation/serve_corrected_vllm.py \
    --model "$MERGED_MODEL" \
    --served-model-name "$MODEL_NAME" \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image": 2}' \
    --inference-mode "$INFERENCE_MODE" \
    --coord-routing \
    --coop-reasoning-alpha 0.3 \
    &
VLLM_PID=$!

MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM server ready after ${WAIT_COUNT}s."
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    if [ $((WAIT_COUNT % 60)) -eq 0 ]; then
        echo "Waiting for vLLM server... ($WAIT_COUNT/$MAX_WAIT seconds)"
    fi
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null
    rm -rf "$MERGED_MODEL"
    exit 1
fi

# Step 3: Run trajectory eval
echo "Step 3: Running AR trajectory eval..."
cd "$PROJECT_ROOT/evaluation"

python eval_qwenvl.py \
    --model_name "$MODEL_NAME" \
    --jsonl_file "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --n_history_image_limit 2 \
    --max_workers 128

EVAL_EXIT=$?

# Cleanup
kill $VLLM_PID 2>/dev/null || true
pkill -f "serve_corrected_vllm" || true
wait $VLLM_PID 2>/dev/null
rm -rf "$MERGED_MODEL"

echo "=========================================="
echo "Done: $(date)  exit=$EVAL_EXIT  mode=$INFERENCE_MODE"
echo "Results: $OUTPUT_DIR"
echo "=========================================="
exit $EVAL_EXIT
