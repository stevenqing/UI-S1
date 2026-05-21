#!/bin/bash
#SBATCH --job-name=ev_v15_lora
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/eval_lora_r256_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/eval_lora_r256_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V15: Eval Standard LoRA r=256 SFT (final checkpoint)
#
# 1. Merge LoRA adapter into base model
# 2. Serve via vLLM
# 3. Run GUI-360 eval (action_prediction_ar)
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
LORA_ADAPTER="$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_balanced_lora_r256"
MERGED_MODEL="$PROJECT_ROOT/checkpoints/v15_balanced_lora_r256_merged"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
MODEL_NAME="v15_lora_r256"
OUTPUT_DIR="$PROJECT_ROOT/v15_gui_360/outputs/eval_lora_r256"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_ROOT/v15_gui_360/scripts/logs"

echo "=========================================="
echo "V15 Eval: LoRA r=256 SFT (balanced data)"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

# Step 1: Merge LoRA into base model (if not already done)
if [ ! -f "$MERGED_MODEL/config.json" ]; then
    echo "Merging LoRA adapter into base model..."
    python -c "
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import torch

print('Loading base model...')
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    '$BASE_MODEL',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
print('Loading LoRA adapter...')
model = PeftModel.from_pretrained(model, '$LORA_ADAPTER')
print('Merging...')
model = model.merge_and_unload()
print('Saving merged model...')
model.save_pretrained('$MERGED_MODEL')

# Copy processor/tokenizer files
proc = AutoProcessor.from_pretrained('$BASE_MODEL')
proc.save_pretrained('$MERGED_MODEL')
print('Done merging.')
"
else
    echo "Merged model already exists, skipping merge."
fi

# Step 2: Start vLLM server
echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MERGED_MODEL" \
    --served-model-name "$MODEL_NAME" \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image": 2}' \
    &
VLLM_PID=$!

# Wait for server
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

# Step 3: Run eval
echo "Running GUI-360 eval..."
python "$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py" \
    --test_data "$TEST_DATA" \
    --api_url "http://localhost:8000/v1" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --threads 128

# Cleanup
echo "Shutting down vLLM..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo "=========================================="
echo "Done: $(date)"
echo "Results: $OUTPUT_DIR"
echo "=========================================="
