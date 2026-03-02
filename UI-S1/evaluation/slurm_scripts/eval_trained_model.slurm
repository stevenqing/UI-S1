#!/bin/bash
#SBATCH --job-name=eval_trained
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/results/eval_trained_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/results/eval_trained_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# Activate environment
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

# Set CUDA library path
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

# Paths
BASE_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CHECKPOINT_DIR="${BASE_DIR}/checkpoints/gui_traj_grpo/qwenvl_uis1_DAPO_False_0.3_patch_2_gamma_0.5/global_step_93"
BASE_MODEL="${BASE_DIR}/checkpoints/Qwen2.5-VL-7B-Instruct"
OUTPUT_MODEL="${BASE_DIR}/checkpoints/UI-S1-GRPO-Trained"
MODEL_NAME="UI-S1-GRPO-Trained"
EVAL_DIR="${BASE_DIR}/evaluation"
DATASET_PATH="${EVAL_DIR}/dataset/android_control_evaluation_fixed.jsonl"
OUTPUT_DIR="${EVAL_DIR}/results"

echo "=========================================="
echo "Starting Trained Model Evaluation Pipeline"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Base Model: $BASE_MODEL"
echo "Output Model: $OUTPUT_MODEL"
echo "Start time: $(date)"
echo "=========================================="

# Step 1: Convert FSDP checkpoint to HuggingFace format
echo ""
echo "Step 1: Converting FSDP checkpoint to HuggingFace format..."
echo "=========================================="

if [ -d "$OUTPUT_MODEL" ] && [ -f "$OUTPUT_MODEL/model.safetensors" ]; then
    echo "HuggingFace checkpoint already exists at $OUTPUT_MODEL"
    echo "Skipping conversion..."
else
    cd $BASE_DIR
    python scripts/convert_fsdp_checkpoint.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --base_model "$BASE_MODEL" \
        --output_dir "$OUTPUT_MODEL" \
        --world_size 16

    CONVERT_EXIT_CODE=$?
    if [ $CONVERT_EXIT_CODE -ne 0 ]; then
        echo "ERROR: Checkpoint conversion failed with exit code $CONVERT_EXIT_CODE"
        exit $CONVERT_EXIT_CODE
    fi
    echo "Checkpoint conversion completed successfully!"
fi

# Step 2: Start vLLM server
echo ""
echo "Step 2: Starting vLLM server..."
echo "=========================================="

cd $EVAL_DIR

vllm serve $OUTPUT_MODEL \
    --served-model-name $MODEL_NAME \
    --tensor-parallel-size 4 \
    --limit-mm-per-prompt image=2 \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 16384 \
    &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Wait for vLLM server to be ready
echo "Waiting for vLLM server to start..."
MAX_RETRIES=60
RETRY_COUNT=0
while ! curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 10
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Waiting for server... ($RETRY_COUNT/$MAX_RETRIES)"
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: vLLM server failed to start within timeout"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
done

echo "vLLM server is ready!"

# Step 3: Run evaluation
echo ""
echo "Step 3: Running evaluation..."
echo "=========================================="

python $EVAL_DIR/eval_qwenvl.py \
    --model_name $MODEL_NAME \
    --jsonl_file $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --n_history_image_limit 2 \
    --max_workers 4

EVAL_EXIT_CODE=$?

# Cleanup - kill vLLM server
echo ""
echo "Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "=========================================="
echo "Evaluation completed at $(date)"
echo "Exit code: $EVAL_EXIT_CODE"
echo "Results saved to: $OUTPUT_DIR/${MODEL_NAME}.jsonl"
echo "=========================================="

# Analyze results
if [ -f "$OUTPUT_DIR/${MODEL_NAME}.jsonl" ]; then
    echo ""
    echo "Results Summary:"
    echo "=========================================="
    python -c "
import json
results = []
with open('$OUTPUT_DIR/${MODEL_NAME}.jsonl', 'r') as f:
    for line in f:
        results.append(json.loads(line))
success = sum(1 for r in results if r.get('task_success', False))
total = len(results)
print(f'Total tasks: {total}')
print(f'Successful tasks: {success}')
print(f'Success rate: {success/total*100:.2f}%' if total > 0 else 'N/A')
"
fi

exit $EVAL_EXIT_CODE
