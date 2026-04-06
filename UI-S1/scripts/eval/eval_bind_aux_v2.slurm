#!/bin/bash
#SBATCH --job-name=eval_ba_v2
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_ba_v2_%j.out
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_ba_v2_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================================
# Evaluate bind_aux_v2_r32: merge LoRA → GUI-360 eval → error decomposition
# ============================================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
LORA_PATH="$PROJECT_DIR/train_GUI_360/llamafactory/output/bind_aux_v2_r32/final"
MERGED_MODEL="$PROJECT_DIR/train_GUI_360/llamafactory/output/bind_aux_v2_r32/final_merged"
EVAL_DIR="$PROJECT_DIR/train_GUI_360/GUI-360-eval"
DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"
PARQUET="$PROJECT_DIR/train_GUI_360/data/gui360_test_sft_eval_format_with_bbox.parquet"
MODEL_TYPE="qwen2.5_vl_7b"
VLLM_PORT=19815
EVAL_THREADS=5

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "Eval Pipeline: bind_aux_v2_r32"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "LoRA: $LORA_PATH"
echo "============================================================"

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export PYTHONUNBUFFERED=1

if [ -f "$PROJECT_DIR/train/env_config.sh" ]; then
    source "$PROJECT_DIR/train/env_config.sh"
fi

mkdir -p "$PROJECT_DIR/scripts/eval/logs"

# ── Step 1: Merge LoRA ──
echo ""
echo "=== Step 1: Merge LoRA ==="
if [ -f "$MERGED_MODEL/model.safetensors.index.json" ]; then
    echo "Merged model already exists, skipping merge"
else
    echo "Merging LoRA adapter..."
    cd "$PROJECT_DIR/train_GUI_360/LLaMA-Factory"
    llamafactory-cli export \
        --model_name_or_path "$BASE_MODEL" \
        --adapter_name_or_path "$LORA_PATH" \
        --template qwen2_vl \
        --finetuning_type lora \
        --export_dir "$MERGED_MODEL" \
        --export_size 4 \
        --export_legacy_format false
    echo "Merge complete at $(date)"
fi

# ── Step 2: GUI-360 Eval ──
echo ""
echo "=== Step 2: GUI-360 Eval ==="
RESULTS_DIR="$EVAL_DIR/results/bind_aux_v2_r32_final_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model $MERGED_MODEL \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $PROJECT_DIR/scripts/eval/logs/vllm_ba_v2_${SLURM_JOB_ID}.log &

VLLM_PID=$!
MAX_WAIT=600; WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    curl -s http://localhost:${VLLM_PORT}/health > /dev/null 2>&1 && echo "vLLM ready!" && break
    sleep 5; WAIT_COUNT=$((WAIT_COUNT + 5))
done
[ $WAIT_COUNT -ge $MAX_WAIT ] && echo "ERROR: vLLM timeout" && kill $VLLM_PID 2>/dev/null && exit 1

cd $EVAL_DIR
API_URL="http://localhost:${VLLM_PORT}/v1"

for EVAL_TYPE in grounding action_prediction action_prediction_a11y; do
    OUTPUT="$RESULTS_DIR/${EVAL_TYPE}_${TIMESTAMP}"
    mkdir -p $OUTPUT
    echo "Running $EVAL_TYPE..."
    python evaluation.py --root_dir $DATASET_ROOT --type $EVAL_TYPE --model_type $MODEL_TYPE --model_name $MERGED_MODEL --api_url $API_URL --threads $EVAL_THREADS --output_dir $OUTPUT
done

kill $VLLM_PID 2>/dev/null || true
sleep 5

echo ""
echo "=== GUI-360 Results ==="
cat $RESULTS_DIR/*/evaluation_summary_*.json 2>/dev/null || true

# ── Step 3: Error Decomposition ──
echo ""
echo "=== Step 3: Error Decomposition ==="
ED_OUTPUT="$PROJECT_DIR/evaluation/results/error_decomp_bind_aux_v2_r32_${TIMESTAMP}"
mkdir -p "$ED_OUTPUT"
cd "$PROJECT_DIR"

python -u evaluation/error_decomposition.py collect \
    --model_path "$BASE_MODEL" \
    --lora_path "$LORA_PATH" \
    --parquet_file "$PARQUET" \
    --n_samples 500 \
    --output_dir "$ED_OUTPUT"

python -u evaluation/error_decomposition.py analyze \
    --predictions "$ED_OUTPUT/predictions.json" \
    --output_dir "$ED_OUTPUT"

echo ""
echo "============================================================"
echo "All evaluations complete at $(date)"
echo "GUI-360: $RESULTS_DIR"
echo "Error decomp: $ED_OUTPUT"
echo "============================================================"
