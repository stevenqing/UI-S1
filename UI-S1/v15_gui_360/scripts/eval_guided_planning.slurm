#!/bin/bash
#SBATCH --job-name=guided_plan
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/guided_planning_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/guided_planning_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Guided Planning Experiment (Variant B: per-step guidance)
#
# Phase 1: Deploy base Qwen2.5-VL-7B-Instruct (not SFT'd)
#   → Generate per-step guidance for all 1000 episodes
#   → Save to JSON
#
# Phase 2: Deploy SFT model (balanced_full_sft step-272)
#   → Run eval with pre-generated guidance
#   → Compare Step SR with baseline
#
# Hypothesis: base model retains general reasoning/planning
# ability that SFT model lost during action-focused fine-tuning
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
SFT_MODEL="$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_balanced_full_sft/checkpoint-272"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
OUTPUT_BASE="$PROJECT_ROOT/v15_gui_360/outputs/preliminary_tests"
GUIDANCE_FILE="$OUTPUT_BASE/step_guidance_base_model.json"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=========================================="
echo "Guided Planning Experiment"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

# ============================================
# Phase 1: Generate guidance with base model
# ============================================
if [ -f "$GUIDANCE_FILE" ]; then
    echo "[Phase 1] Guidance file already exists, skipping."
else
    echo ""
    echo "=========================================="
    echo "Phase 1: Generate per-step guidance (base model)"
    echo "Time: $(date)"
    echo "=========================================="

    python -m vllm.entrypoints.openai.api_server \
        --model "$BASE_MODEL" \
        --served-model-name "Qwen2.5-VL-7B-Instruct" \
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
        echo "ERROR: vLLM server failed to start (Phase 1)"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi

    python "$PROJECT_ROOT/v15_gui_360/generate_step_guidance.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "Qwen2.5-VL-7B-Instruct" \
        --output_file "$GUIDANCE_FILE" \
        --threads 64

    echo "Shutting down base model..."
    kill $VLLM_PID 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    sleep 15
fi

# ============================================
# Phase 2: Eval SFT model with guidance
# ============================================
GUIDED_DIR="$OUTPUT_BASE/exp_d_guided_base_model"
if ls "$GUIDED_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[Phase 2] Already evaluated, skipping."
else
    echo ""
    echo "=========================================="
    echo "Phase 2: Eval SFT model with base model guidance"
    echo "Time: $(date)"
    echo "=========================================="

    python -m vllm.entrypoints.openai.api_server \
        --model "$SFT_MODEL" \
        --served-model-name "balanced_sft_step272" \
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
        echo "ERROR: vLLM server failed to start (Phase 2)"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi

    mkdir -p "$GUIDED_DIR"

    python "$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "balanced_sft_step272" \
        --output_dir "$GUIDED_DIR" \
        --threads 128 \
        --gt_history \
        --guidance_file "$GUIDANCE_FILE" \
        --match_threshold 0.5

    echo "Shutting down SFT model..."
    kill $VLLM_PID 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
fi

# Print comparison
echo ""
echo "=========================================="
echo "Results Comparison"
echo "=========================================="
python3 -c "
import json, glob

print(f'{\"Experiment\":<40} {\"TSR\":<8} {\"StepSR\":<8} {\"Progress\":<10}')
print('-' * 70)

# Baseline
bl_dir = '$PROJECT_ROOT/v15_gui_360/outputs/history_ablation/gt_history_full'
bl_files = sorted(glob.glob(f'{bl_dir}/eval_summary_*.json'))
if bl_files:
    d = json.load(open(bl_files[-1]))
    print(f'{\"Baseline (GT hist, no guidance)\":<40} {d[\"tsr\"]:<8.3f} {d[\"step_sr\"]:<8.3f} {d[\"avg_progress\"]:<10.3f}')

# Exp A (self-plan)
ea_dir = '$OUTPUT_BASE/exp_a_planning'
ea_files = sorted(glob.glob(f'{ea_dir}/eval_summary_*.json'))
if ea_files:
    d = json.load(open(ea_files[-1]))
    print(f'{\"Exp A: Self-generated plan\":<40} {d[\"tsr\"]:<8.3f} {d[\"step_sr\"]:<8.3f} {d[\"avg_progress\"]:<10.3f}')

# Exp D (base model guidance)
ed_dir = '$OUTPUT_BASE/exp_d_guided_base_model'
ed_files = sorted(glob.glob(f'{ed_dir}/eval_summary_*.json'))
if ed_files:
    d = json.load(open(ed_files[-1]))
    print(f'{\"Exp D: Base model per-step guidance\":<40} {d[\"tsr\"]:<8.3f} {d[\"step_sr\"]:<8.3f} {d[\"avg_progress\"]:<10.3f}')
"

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
