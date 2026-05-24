#!/bin/bash
#SBATCH --job-name=res_ablat
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/resolution_ablation_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/resolution_ablation_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Resolution Ablation Experiment
#
# Model: gui360_balanced_full_sft checkpoint-272
# All conditions: GT history (teacher-forced)
#
# Conditions:
#   (a) 602112 px — match training resolution
#   (b) default  — full resolution (~765K, current baseline)
#   (c) 1503200 px — ~1.5M pixels, higher detail
#
# Screenshots are 1040×736 = 765,440 px
# Training used image_max_pixels=602112 (downscaled to ~776×776)
# vLLM default max_pixels=12.8M (no downscaling)
#
# Key question: is visual grounding limited by resolution?
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CKPT_PATH="$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_balanced_full_sft/checkpoint-272"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
OUTPUT_BASE="$PROJECT_ROOT/v15_gui_360/outputs/resolution_ablation"
MODEL_NAME="balanced_sft_step272"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=========================================="
echo "Resolution Ablation Experiment"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "$CKPT_PATH" \
    --served-model-name "$MODEL_NAME" \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image": 2}' \
    &
VLLM_PID=$!

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

# Run conditions
# Note: "default" = no --image_max_pixels flag = send full-res image
for COND in "602112:train_match" "default:full_res"; do
    MAX_PX=$(echo "$COND" | cut -d: -f1)
    LABEL=$(echo "$COND" | cut -d: -f2)
    OUTPUT_DIR="$OUTPUT_BASE/gt_history_${LABEL}"

    if ls "$OUTPUT_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
        echo "[${LABEL}] Already evaluated, skipping."
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Condition: ${LABEL} (max_pixels=${MAX_PX})"
    echo "Time: $(date)"
    echo "=========================================="

    mkdir -p "$OUTPUT_DIR"

    if [ "$MAX_PX" = "default" ]; then
        python "$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py" \
            --test_data "$TEST_DATA" \
            --api_url "http://localhost:8000/v1" \
            --model_name "$MODEL_NAME" \
            --output_dir "$OUTPUT_DIR" \
            --threads 128 \
            --gt_history \
            --match_threshold 0.5
    else
        python "$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py" \
            --test_data "$TEST_DATA" \
            --api_url "http://localhost:8000/v1" \
            --model_name "$MODEL_NAME" \
            --output_dir "$OUTPUT_DIR" \
            --threads 128 \
            --gt_history \
            --match_threshold 0.5 \
            --image_max_pixels "$MAX_PX"
    fi

    echo "[${LABEL}] Done."
done

# Shutdown
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

# Print comparison
echo ""
echo "=========================================="
echo "Resolution Ablation Results"
echo "=========================================="
python3 -c "
import json, glob

# Include history ablation baseline
print(f'{\"Condition\":<25} {\"MaxPx\":<12} {\"TSR\":<8} {\"StepSR\":<8} {\"Progress\":<10}')
print('-' * 65)

# Baseline from history ablation (full res, no --image_max_pixels)
bl_dir = '$PROJECT_ROOT/v15_gui_360/outputs/history_ablation/gt_history_full'
bl_files = sorted(glob.glob(f'{bl_dir}/eval_summary_*.json'))
if bl_files:
    d = json.load(open(bl_files[-1]))
    print(f'{\"baseline (hist ablat)\":<25} {\"~765K\":<12} {d[\"tsr\"]:<8.3f} {d[\"step_sr\"]:<8.3f} {d[\"avg_progress\"]:<10.3f}')

for label in ['train_match', 'full_res']:
    edir = '$OUTPUT_BASE/gt_history_' + label
    summary_files = sorted(glob.glob(f'{edir}/eval_summary_*.json'))
    if summary_files:
        d = json.load(open(summary_files[-1]))
        px = '602K' if 'train' in label else '~765K'
        print(f'{label:<25} {px:<12} {d[\"tsr\"]:<8.3f} {d[\"step_sr\"]:<8.3f} {d[\"avg_progress\"]:<10.3f}')
    else:
        print(f'{label:<25} {\"N/A\":<12}')
"

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
