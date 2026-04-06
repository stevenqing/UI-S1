#!/bin/bash
#SBATCH --job-name=eval_v9_all
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_v9_all_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_v9_all_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ============================================================
# Evaluate V9 and V9b best checkpoints
# Candidates: V9 step 40, V9 step 50, V9b step 40
# Modes: standard + hindsight (visual IDM eval)
# workers=128
# ============================================================

set -e
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="${PROJECT_DIR}/evaluation"
SCRIPT_DIR="${PROJECT_DIR}/scripts/eval"
DATASET_PATH="${EVAL_DIR}/dataset/android_control_evaluation_fixed.jsonl"
OUTPUT_DIR="${EVAL_DIR}/results/v9_eval"
BASE_MODEL="${PROJECT_DIR}/checkpoints/Qwen2.5-VL-7B-Instruct"
MAX_SAMPLES=500

mkdir -p $OUTPUT_DIR ${SCRIPT_DIR}/logs
cd $PROJECT_DIR

# Define candidates
declare -A CKPTS
CKPTS[v9_step40]="${PROJECT_DIR}/checkpoints/sp_gigpo_ac/sp_gigpo_spwa_k8_v9/global_step_40"
CKPTS[v9_step50]="${PROJECT_DIR}/checkpoints/sp_gigpo_ac/sp_gigpo_spwa_k8_v9/global_step_50"
CKPTS[v9b_step40]="${PROJECT_DIR}/checkpoints/sp_gigpo_ac/sp_gigpo_spwa_k8_v9b/global_step_40"

for NAME in v9_step40 v9_step50 v9b_step40; do
    CKPT_DIR="${CKPTS[$NAME]}"
    MERGED_DIR="${CKPT_DIR%/*}/merged_${NAME#*_}"

    echo "=========================================="
    echo "Processing: $NAME"
    echo "Checkpoint: $CKPT_DIR"
    echo "Merged: $MERGED_DIR"
    echo "=========================================="

    # Check checkpoint exists
    if [ ! -d "$CKPT_DIR" ]; then
        echo "SKIP: $CKPT_DIR does not exist"
        continue
    fi

    # Step 1: Merge if needed
    if [ ! -d "$MERGED_DIR" ] || [ ! -f "$MERGED_DIR/config.json" ]; then
        echo "Merging FSDP checkpoint..."
        python ${SCRIPT_DIR}/merge_fsdp_checkpoint.py \
            --ckpt_dir "$CKPT_DIR" \
            --output_dir "$MERGED_DIR" \
            --base_model "$BASE_MODEL"
    else
        echo "Merged checkpoint already exists, skipping merge"
    fi

    # Step 2: Start vLLM server
    echo "Starting vLLM server for $NAME..."
    vllm serve "$MERGED_DIR" \
        --served-model-name $NAME \
        --tensor-parallel-size 4 \
        --limit-mm-per-prompt image=3 \
        --port 8000 \
        --trust-remote-code \
        &
    VLLM_PID=$!

    # Wait for server
    MAX_RETRIES=60
    RETRY_COUNT=0
    while ! curl -s http://localhost:8000/health > /dev/null 2>&1; do
        sleep 10
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
            echo "ERROR: vLLM server failed to start for $NAME"
            kill $VLLM_PID 2>/dev/null
            continue 2
        fi
    done
    echo "vLLM server ready for $NAME!"

    # Step 3: Evaluate standard + hindsight
    for MODE in standard hindsight; do
        echo "Running mode: $MODE for $NAME ($(date))"
        python ${SCRIPT_DIR}/eval_visual_idm.py \
            --model_name $NAME \
            --jsonl_file $DATASET_PATH \
            --output_dir $OUTPUT_DIR \
            --mode $MODE \
            --max_samples $MAX_SAMPLES \
            --max_workers 128
    done

    # Step 4: Kill vLLM
    echo "Stopping vLLM server for $NAME..."
    kill $VLLM_PID 2>/dev/null
    wait $VLLM_PID 2>/dev/null
    sleep 5

    echo "$NAME evaluation complete at $(date)"
    echo ""
done

echo "=========================================="
echo "All evaluations complete at $(date)"
echo "Results in: $OUTPUT_DIR"
echo "=========================================="

# Print summary
echo ""
echo "=== RESULTS SUMMARY ==="
for f in ${OUTPUT_DIR}/*summary*.json; do
    echo "--- $(basename $f) ---"
    python -c "import json; d=json.load(open('$f')); print(f'  type_match: {d.get(\"type_match\",\"N/A\")}, extract_match: {d.get(\"extract_match\",\"N/A\")}, n={d.get(\"n\",\"N/A\")}')"
done
