#!/bin/bash
#SBATCH --job-name=eval_ody_ll
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_coop_odyssey_vllm_lossless_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_coop_odyssey_vllm_lossless_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Eval cooperative LoRA v6.5 on GUI-Odyssey via vLLM (LOSSLESS)
#
# 1) Lossless merge: base weights + correction adapters
# 2) Start corrected vLLM server (per-token correction hooks)
# 3) Run AR trajectory eval (stop + no-stop)
#
# Usage:
#   COOP_EPOCH=2 sbatch eval_coop_odyssey_vllm_lossless.slurm
#   COOP_EPOCH=3 sbatch eval_coop_odyssey_vllm_lossless.slurm
# ============================================

set -e

COOP_EPOCH="${COOP_EPOCH:-2}"

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
COOP_CKPT="$PROJECT_ROOT/train_GUI_360/llamafactory/output/cooperative_v6_5_odyssey/epoch-${COOP_EPOCH}"
MERGED_MODEL="/tmp/merged_lossless_ody_ep${COOP_EPOCH}_${SLURM_JOB_ID}"
MODEL_NAME="coop_ody_ep${COOP_EPOCH}"
SPLIT="${SPLIT:-random_split}"
SUBSET="${SUBSET:-test}"
DATASET_DIR="$PROJECT_ROOT/datasets/GUI-Odyssey"
JSONL_FILE="$DATASET_DIR/gui_odyssey_${SPLIT}_${SUBSET}.jsonl"
ODYSSEY_DIR="$PROJECT_ROOT/gui_odyssey_eval"
OUTPUT_DIR="$PROJECT_ROOT/outputs/eval_cooperative_v6_5_odyssey/epoch${COOP_EPOCH}_lossless_vllm"

echo "=========================================="
echo "Eval: Cooperative v6.5 Odyssey epoch-${COOP_EPOCH} (vLLM LOSSLESS)"
echo "=========================================="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Base:      $BASE_MODEL"
echo "COOP_CKPT: $COOP_CKPT"
echo "Merged:    $MERGED_MODEL"
echo "JSONL:     $JSONL_FILE"
echo "Output:    $OUTPUT_DIR"
echo "Start:     $(date)"
echo "=========================================="

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p "$OUTPUT_DIR"

# Step 0: Convert dataset if needed
if [ ! -f "$JSONL_FILE" ]; then
    echo "Converting GUI-Odyssey ${SPLIT}/${SUBSET} to eval format..."
    python "$ODYSSEY_DIR/convert_to_eval_format.py" \
        --data_dir "$DATASET_DIR" \
        --split "$SPLIT" \
        --subset "$SUBSET" \
        --output_dir "$DATASET_DIR"
fi

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

if [ ! -f "$MERGED_MODEL/correction_adapters.pt" ]; then
    echo "ERROR: correction_adapters.pt not found after merge"
    exit 1
fi
echo "Lossless merge complete: $MERGED_MODEL"
ls -lh "$MERGED_MODEL/correction_adapters.pt" "$MERGED_MODEL/correction_config.json"

# Step 2: Start corrected vLLM server
echo "Step 2: Starting corrected vLLM server..."
python evaluation/serve_corrected_vllm.py \
    --model "$MERGED_MODEL" \
    --served-model-name "$MODEL_NAME" \
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
        echo "Corrected vLLM server ready after ${WAIT_COUNT}s."
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    if [ $((WAIT_COUNT % 60)) -eq 0 ]; then
        echo "Waiting for vLLM server... ($WAIT_COUNT/$MAX_WAIT seconds)"
    fi
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: Corrected vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null
    rm -rf "$MERGED_MODEL"
    exit 1
fi

# Step 3: Run trajectory eval (stop-on-error)
cd "$ODYSSEY_DIR"

echo "Step 3: Running stop-on-error evaluation..."
python eval_ar_trajectory.py \
    --jsonl_file "$JSONL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --split_name "$SPLIT" \
    --n_history_image_limit 2 \
    --max_workers 128

echo "Stop-on-error evaluation completed."
cat "$OUTPUT_DIR/summary.json" 2>/dev/null || echo "No summary found"

# Step 4: Run no-stop evaluation
NOSTOP_DIR="${OUTPUT_DIR}_nostop"
mkdir -p "$NOSTOP_DIR"

echo ""
echo "=========================================="
echo "Running no-stop evaluation (all steps)..."
echo "=========================================="

python eval_ar_trajectory.py \
    --jsonl_file "$JSONL_FILE" \
    --output_dir "$NOSTOP_DIR" \
    --model_name "$MODEL_NAME" \
    --split_name "$SPLIT" \
    --n_history_image_limit 2 \
    --max_workers 128 \
    --no_stop

echo "No-stop evaluation completed."
cat "$NOSTOP_DIR/summary.json" 2>/dev/null || echo "No summary found"

# Cleanup
kill $VLLM_PID 2>/dev/null || true
pkill -f "serve_corrected_vllm" || true
wait $VLLM_PID 2>/dev/null
rm -rf "$MERGED_MODEL"

echo "=========================================="
echo "Done: $(date)"
echo "Results: $OUTPUT_DIR"
echo "=========================================="
