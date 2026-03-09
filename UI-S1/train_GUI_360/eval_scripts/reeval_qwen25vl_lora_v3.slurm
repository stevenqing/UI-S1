#!/bin/bash
#SBATCH --job-name=reeval_lora_v3
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/reeval_lora_v3_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/reeval_lora_v3_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# Re-evaluate LoRA SFT v3 with fixed pipeline
set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
EVAL_DIR="$TRAIN_DIR/GUI-360-eval"
LOG_DIR="$TRAIN_DIR/logs"
RESULTS_DIR="$EVAL_DIR/results/qwen25vl_lora_v3_fixed"

BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
LORA_ADAPTER="$TRAIN_DIR/llamafactory/output/gui360_lora_sft_v3"
MERGED_MODEL="$TRAIN_DIR/llamafactory/output/gui360_lora_sft_v3_merged"
MODEL_TYPE="qwen2.5_vl_7b"
DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"

VLLM_PORT=19808
VLLM_HOST="localhost"
EVAL_THREADS=20

echo "=========================================="
echo "Re-eval: LoRA SFT v3 (fixed pipeline)"
echo "Job ID: ${SLURM_JOB_ID} | Node: $(hostname) | Start: $(date)"
echo "=========================================="

module load cuda/12.6
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

mkdir -p $LOG_DIR $RESULTS_DIR

[ ! -d "$DATASET_ROOT" ] && echo "ERROR: Dataset not found" && exit 1

# Merge LoRA if not already done
if [ -f "$MERGED_MODEL/model.safetensors.index.json" ]; then
    echo "Merged model exists, skipping merge."
else
    echo "Merging LoRA adapter..."
    llamafactory-cli export \
        --model_name_or_path $BASE_MODEL \
        --adapter_name_or_path $LORA_ADAPTER \
        --template qwen2_vl \
        --finetuning_type lora \
        --export_dir $MERGED_MODEL \
        --export_size 5 \
        --export_legacy_format False
    echo "Merge completed."
fi

pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model $MERGED_MODEL \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $LOG_DIR/vllm_reeval_lora_v3_${SLURM_JOB_ID}.log &

VLLM_PID=$!

MAX_WAIT=600; WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    curl -s http://${VLLM_HOST}:${VLLM_PORT}/health > /dev/null 2>&1 && echo "vLLM ready!" && break
    sleep 5; WAIT_COUNT=$((WAIT_COUNT + 5))
    echo "Waiting... ($WAIT_COUNT/$MAX_WAIT s)"
done
[ $WAIT_COUNT -ge $MAX_WAIT ] && echo "ERROR: vLLM timeout" && kill $VLLM_PID 2>/dev/null && exit 1

cd $EVAL_DIR
API_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

run_eval() {
    local EVAL_TYPE=$1
    local OUTPUT_DIR="$RESULTS_DIR/${EVAL_TYPE}_${TIMESTAMP}"
    mkdir -p $OUTPUT_DIR
    echo "Running $EVAL_TYPE..."
    python evaluation.py \
        --root_dir $DATASET_ROOT --type $EVAL_TYPE \
        --model_type $MODEL_TYPE --model_name $MERGED_MODEL \
        --api_url $API_URL --threads $EVAL_THREADS --output_dir $OUTPUT_DIR
    echo "$EVAL_TYPE done. Results: $OUTPUT_DIR"
}

run_eval "grounding"
run_eval "action_prediction"

kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true
echo "Completed at $(date). Results: $RESULTS_DIR"
ls -la $RESULTS_DIR/
