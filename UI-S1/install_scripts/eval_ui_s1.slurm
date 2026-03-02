#!/bin/bash
#SBATCH --job-name=eval_ui-s1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --gres=gpu:4

# =============================================================================
# UI-S1-7B Evaluation Script
# Downloads evaluation images and runs evaluation
# =============================================================================

set -e

# Load modules
module load gcc-native/12.3 cuda/12.6

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate ui-s1

# Set paths
MODEL_PATH=/scratch/a5l/shuqing.a5l/models/UI-S1-7B
EVAL_DIR=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation
OUTPUT_DIR=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/results
IMAGES_DIR=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/AndroidControl/images
DATASET_PATH=$EVAL_DIR/dataset/android_control_evaluation_std.jsonl

# Set CUDA paths
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"

# Fix GLIBCXX
if [ -f /opt/cray/pe/gcc-native/12/lib64/libstdc++.so.6 ]; then
    export LD_LIBRARY_PATH="/opt/cray/pe/gcc-native/12/lib64:${LD_LIBRARY_PATH:-}"
fi

# vLLM environment variables
export VLLM_USE_V1=0
export VLLM_TORCH_COMPILE=0
export VLLM_USE_TRITON_FLASH_ATTN=0

mkdir -p $OUTPUT_DIR

echo "=== Starting UI-S1-7B Evaluation ==="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
date

# Fix image paths in evaluation dataset (convert /datasets/... to our actual path)
echo "=== Fixing image paths in evaluation dataset ==="
FIXED_DATASET=$OUTPUT_DIR/android_control_evaluation_fixed.jsonl
python -c "
import json
with open('$DATASET_PATH') as f, open('$FIXED_DATASET', 'w') as out:
    for line in f:
        d = json.loads(line)
        for step in d['steps']:
            # Replace /datasets/AndroidControl/images/ with our actual path
            step['screenshot'] = step['screenshot'].replace(
                '/datasets/AndroidControl/images/',
                '$IMAGES_DIR/'
            )
        out.write(json.dumps(d) + '\n')
print('Fixed dataset saved to: $FIXED_DATASET')
"
DATASET_PATH=$FIXED_DATASET

# Start vLLM server in background
echo "=== Starting vLLM server ==="
vllm serve $MODEL_PATH \
    --served-model-name UI-S1-7B \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --limit-mm-per-prompt image=2 \
    --port 8000 \
    --host 0.0.0.0 \
    --dtype auto \
    --max-model-len 8192 \
    &

VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

# Wait for server to be ready
echo "Waiting for vLLM server to start..."
MAX_WAIT=300
WAITED=0
while ! curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM server failed to start within $MAX_WAIT seconds"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    echo "  Still waiting... ($WAITED seconds)"
done

echo "vLLM server is ready!"

# Run evaluation
echo "=== Running evaluation ==="
cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1

python evaluation/eval_qwenvl.py \
    --model_name UI-S1-7B \
    --jsonl_file $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --n_history_image_limit 2 \
    --max_workers 4

echo "=== Evaluation complete ==="

# Cleanup - stop vLLM server
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo "=== Results ==="
if [ -f "$OUTPUT_DIR/UI-S1-7B.jsonl" ]; then
    echo "Results saved to: $OUTPUT_DIR/UI-S1-7B.jsonl"
    echo "Total evaluations: $(wc -l < $OUTPUT_DIR/UI-S1-7B.jsonl)"
    echo "Successes: $(grep -c '"task_success": true' $OUTPUT_DIR/UI-S1-7B.jsonl || echo 0)"
fi

date
echo "=== Done ==="
