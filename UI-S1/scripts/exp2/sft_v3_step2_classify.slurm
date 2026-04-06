#!/bin/bash
#SBATCH --job-name=sft3_s2_cls
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/sft_v3_step2_classify_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/sft_v3_step2_classify_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# SFT v3: LLM Error Classification
# Depends on SFT v3 Step 1 results
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
SCRIPT_DIR="$PROJECT_DIR/scripts/exp2"
RESULTS_DIR="$SCRIPT_DIR/results/sft_v3/classification"
LOG_DIR="$SCRIPT_DIR/logs"

# Use base model for classification (classifier should be consistent)
MODEL_NAME="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"

VLLM_PORT=19806
VLLM_HOST="localhost"

echo "=========================================="
echo "SFT v3: LLM Error Classification"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

mkdir -p $LOG_DIR
mkdir -p $RESULTS_DIR

# Start vLLM Server (use base model for classification)
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $LOG_DIR/vllm_sft3_s2_${SLURM_JOB_ID}.log &

VLLM_PID=$!

MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://${VLLM_HOST}:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    echo "Waiting... ($WAIT_COUNT/$MAX_WAIT seconds)"
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

API_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
cd $PROJECT_DIR

# Classify AC errors from SFT v3 results
echo "=== Classifying AC errors (SFT v3) ==="
python $SCRIPT_DIR/step2_classify_errors.py \
    --ac_results_dir $SCRIPT_DIR/results/sft_v3/ac \
    --gui360_results_dir $SCRIPT_DIR/results/sft_v3/gui360 \
    --output_dir $RESULTS_DIR \
    --api_url $API_URL \
    --model_name $MODEL_NAME \
    --max_workers 4 \
    --max_trajectories 500

# Convert GUI-360 results and classify
echo ""
echo "=== Converting GUI-360 results ==="
SFT_GUI360_DIR="$SCRIPT_DIR/results/sft_v3/gui360"
LATEST_NOSTOP=$(readlink -f $SFT_GUI360_DIR/latest_nostop 2>/dev/null || echo "")

if [ -n "$LATEST_NOSTOP" ] && [ -d "$LATEST_NOSTOP" ]; then
    # Find the results JSON file
    GUI360_JSON=$(ls $LATEST_NOSTOP/ar_evaluation_results_*.json 2>/dev/null | head -1)
    if [ -n "$GUI360_JSON" ]; then
        echo "Converting $GUI360_JSON to unified JSONL"
        python $SCRIPT_DIR/convert_gui360_to_unified.py \
            --input_json "$GUI360_JSON" \
            --output_jsonl "$SFT_GUI360_DIR/gui360_nostop_results.jsonl"

        echo "=== Classifying GUI-360 errors (SFT v3) ==="
        python $SCRIPT_DIR/step2_gui360_classify_and_merge.py \
            --gui360_jsonl "$SFT_GUI360_DIR/gui360_nostop_results.jsonl" \
            --existing_summary "$RESULTS_DIR/classification_summary.json" \
            --output_dir $RESULTS_DIR \
            --api_url $API_URL \
            --model_name $MODEL_NAME \
            --max_workers 4 \
            --max_trajectories 500
    else
        echo "WARNING: No GUI-360 nostop results JSON found in $LATEST_NOSTOP"
    fi
else
    echo "WARNING: No GUI-360 latest_nostop symlink found"
fi

# Run cascade analysis on SFT v3 results
echo ""
echo "=== Running Cascade Analysis (SFT v3) ==="
ANALYSIS_DIR="$SCRIPT_DIR/results/sft_v3/analysis"
mkdir -p $ANALYSIS_DIR
python $SCRIPT_DIR/step1_analyze_cascade.py \
    --ac_results_dir $SCRIPT_DIR/results/sft_v3/ac \
    --gui360_results_dir $SCRIPT_DIR/results/sft_v3/gui360 \
    --output_dir $ANALYSIS_DIR

# Cleanup
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true

echo ""
echo "End: $(date)"
echo "Classification results: $RESULTS_DIR"
cat $RESULTS_DIR/classification_summary.json 2>/dev/null || echo "Not found"
echo ""
echo "Analysis results: $ANALYSIS_DIR"
cat $ANALYSIS_DIR/cross_dataset_cascade_report.md 2>/dev/null || echo "Not found"
