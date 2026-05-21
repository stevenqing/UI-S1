#!/bin/bash
#SBATCH --job-name=eval_r512_ss
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_sft_r512_gui360_ss_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_sft_r512_gui360_ss_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Single-step action_prediction eval for SFT r=512 on GUI-360
# Comparable to cooperative v6.5's 50.06% single-step accuracy
#
# Usage: ADAPTER_PATH=... EVAL_TAG=ep4 sbatch this_script.slurm
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH="${ADAPTER_PATH:-$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_lora_r512_thought_baseline}"
EVAL_TAG="${EVAL_TAG:-ep4}"
MERGED_MODEL="/tmp/merged_sft_r512_ss_${EVAL_TAG}_${SLURM_JOB_ID}"
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
echo "Eval: SFT r=512 Single-Step on GUI-360 ($EVAL_TAG)"
echo "Adapter: $ADAPTER_PATH"
echo "Merged:  $MERGED_MODEL"
echo "Workers: $EVAL_WORKERS"
echo "Start:   $(date)"
echo "=========================================="

# Step 1: Merge LoRA
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

processor = AutoProcessor.from_pretrained(base_path)
processor.save_pretrained(output_path)

print('Merge complete.')
"
MERGE_EXIT=$?
if [ $MERGE_EXIT -ne 0 ]; then
    echo "ERROR: Merge failed"; exit $MERGE_EXIT
fi

# Step 2: Start vLLM server
echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MERGED_MODEL" \
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
        echo "vLLM server ready."
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null; rm -rf "$MERGED_MODEL"; exit 1
fi

cd "$GUI360_DIR"
API_URL="http://localhost:8000/v1"

# Step 3: Single-step action_prediction
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SS_OUTPUT="$OUTPUT_BASE/sft_r512_${EVAL_TAG}_singlestep_${TIMESTAMP}"
echo ""
echo "=== Single-Step Action Prediction ==="
echo "Output: $SS_OUTPUT"

python evaluation.py \
    --root_dir "$GUI360_DATA" \
    --model_name "$MERGED_MODEL" \
    --model_type "qwen2.5_vl_7b" \
    --type "action_prediction" \
    --threads "$EVAL_WORKERS" \
    --output_dir "$SS_OUTPUT" \
    --api_url "$API_URL"

echo "Single-step eval completed."

# Cleanup
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true
rm -rf "$MERGED_MODEL"

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "Results: $SS_OUTPUT"
echo "=========================================="
