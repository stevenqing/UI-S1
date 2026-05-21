#!/bin/bash
#SBATCH --job-name=eval_ac_mt
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_sft_multiturn_ac_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_sft_multiturn_ac_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Eval SFT multi-turn baseline on AC
# 1) Merge PEFT LoRA adapter into base model
# 2) Start vLLM server on merged model
# 3) Run trajectory eval
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH="${ADAPTER_PATH:-$PROJECT_ROOT/train_GUI_360/llamafactory/output/v6_5_ac_sft_multiturn}"
MERGED_MODEL="/tmp/merged_ac_sft_multiturn_${SLURM_JOB_ID}"
MODEL_NAME="ac_sft_multiturn"
DATASET_PATH="$PROJECT_ROOT/datasets/android_control_evaluation_std.jsonl"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/eval_cooperative/ac_sft_multiturn}"

echo "=========================================="
echo "Eval: AC SFT Multi-Turn Baseline"
echo "Base:    $BASE_MODEL"
echo "Adapter: $ADAPTER_PATH"
echo "Merged:  $MERGED_MODEL"
echo "Output:  $OUTPUT_DIR"
echo "Start:   $(date)"
echo "=========================================="

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p "$OUTPUT_DIR"

# Step 1: Merge LoRA into base model
echo "Merging LoRA adapter..."
python -c "
import torch
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

base_path = '$BASE_MODEL'
adapter_path = '$ADAPTER_PATH'
output_path = '$MERGED_MODEL'

print(f'Loading base model from {base_path}...')
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_path, torch_dtype=torch.bfloat16, device_map='cpu')

print(f'Loading adapter from {adapter_path}...')
model = PeftModel.from_pretrained(model, adapter_path)

print('Merging and unloading...')
model = model.merge_and_unload()

print(f'Saving merged model to {output_path}...')
model.save_pretrained(output_path)

# Copy tokenizer/processor
processor = AutoProcessor.from_pretrained(base_path)
processor.save_pretrained(output_path)

print('Merge complete.')
"
MERGE_EXIT=$?
if [ $MERGE_EXIT -ne 0 ]; then
    echo "ERROR: Merge failed with exit code $MERGE_EXIT"
    exit $MERGE_EXIT
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

# Step 3: Run trajectory eval (using original eval_qwenvl.py)
cd "$PROJECT_ROOT/evaluation"

python eval_qwenvl.py \
    --model_name "$MODEL_NAME" \
    --jsonl_file "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --n_history_image_limit 2 \
    --max_workers 128

EVAL_EXIT=$?

kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true
wait $VLLM_PID 2>/dev/null
rm -rf "$MERGED_MODEL"

echo "=========================================="
echo "Done: $(date)  exit=$EVAL_EXIT"
echo "Results: $OUTPUT_DIR"
echo "=========================================="
exit $EVAL_EXIT
