#!/bin/bash
#SBATCH --job-name=acc_boost
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/accuracy_boost_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/accuracy_boost_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Per-Step Accuracy Boosting Experiments
#
# Three approaches to improve effective per-step accuracy:
#   (1) CoT per-step reasoning (explicit element identification)
#   (2) Self-consistency N=3 (majority voting + oracle)
#   (3) Self-consistency N=5 (majority voting + oracle)
#
# All with GT history (teacher-forced) for clean measurement.
# Model: gui360_balanced_full_sft checkpoint-272
#
# Estimated time: CoT ~40min, SC-N3 ~90min, SC-N5 ~150min = ~5h total
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CKPT_PATH="$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_balanced_full_sft/checkpoint-272"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
OUTPUT_BASE="$PROJECT_ROOT/v15_gui_360/outputs/preliminary_tests"
MODEL_NAME="balanced_sft_step272"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=========================================="
echo "Per-Step Accuracy Boosting Experiments"
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

# ── Experiment 1: CoT per-step reasoning ──
echo ""
echo "=========================================="
echo "Exp 1: Chain-of-Thought per step"
echo "Time: $(date)"
echo "=========================================="

COT_DIR="$OUTPUT_BASE/cot_step"
if ls "$COT_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "CoT already evaluated, skipping."
else
    mkdir -p "$COT_DIR"
    python "$PROJECT_ROOT/v15_gui_360/eval_cot_step.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$COT_DIR" \
        --threads 128 \
        --gt_history \
        --match_threshold 0.5
    echo "CoT done: $(date)"
fi

# ── Experiment 2: Self-consistency N=3 ──
echo ""
echo "=========================================="
echo "Exp 2: Self-consistency N=3"
echo "Time: $(date)"
echo "=========================================="

SC3_DIR="$OUTPUT_BASE/self_consistency_n3"
if ls "$SC3_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "SC N=3 already evaluated, skipping."
else
    mkdir -p "$SC3_DIR"
    python "$PROJECT_ROOT/v15_gui_360/eval_self_consistency.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$SC3_DIR" \
        --threads 64 \
        --gt_history \
        --num_samples 3 \
        --temperature 0.7 \
        --match_threshold 0.5
    echo "SC N=3 done: $(date)"
fi

# ── Experiment 3: Self-consistency N=5 ──
echo ""
echo "=========================================="
echo "Exp 3: Self-consistency N=5"
echo "Time: $(date)"
echo "=========================================="

SC5_DIR="$OUTPUT_BASE/self_consistency_n5"
if ls "$SC5_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "SC N=5 already evaluated, skipping."
else
    mkdir -p "$SC5_DIR"
    python "$PROJECT_ROOT/v15_gui_360/eval_self_consistency.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$SC5_DIR" \
        --threads 64 \
        --gt_history \
        --num_samples 5 \
        --temperature 0.7 \
        --match_threshold 0.5
    echo "SC N=5 done: $(date)"
fi

# Shutdown
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

# ── Print comparison ──
echo ""
echo "=========================================="
echo "Accuracy Boosting Results Summary"
echo "=========================================="
python3 -c "
import json, glob

print(f'{\"Method\":<30} {\"Step SR\":<10} {\"Oracle\":<10} {\"TSR\":<8}')
print('-' * 60)

# Baseline
bl_dir = '$OUTPUT_BASE/../history_ablation/gt_history_full'
bl_files = sorted(glob.glob(f'{bl_dir}/eval_summary_*.json'))
if bl_files:
    d = json.load(open(bl_files[-1]))
    print(f'{\"Baseline (greedy T=0)\":<30} {d[\"step_sr\"]*100:<10.1f} {\"—\":<10} {d[\"tsr\"]*100:<8.1f}')

# CoT
cot_files = sorted(glob.glob('$COT_DIR/eval_summary_*.json'))
if cot_files:
    d = json.load(open(cot_files[-1]))
    print(f'{\"CoT per-step\":<30} {d[\"step_sr\"]*100:<10.1f} {\"—\":<10} {d[\"tsr\"]*100:<8.1f}')

# SC N=3
sc3_files = sorted(glob.glob('$SC3_DIR/eval_summary_*.json'))
if sc3_files:
    d = json.load(open(sc3_files[-1]))
    print(f'{\"Self-consistency N=3\":<30} {d[\"majority_vote_step_sr\"]*100:<10.1f} {d[\"oracle_best_of_n_step_sr\"]*100:<10.1f} {\"—\":<8}')

# SC N=5
sc5_files = sorted(glob.glob('$SC5_DIR/eval_summary_*.json'))
if sc5_files:
    d = json.load(open(sc5_files[-1]))
    print(f'{\"Self-consistency N=5\":<30} {d[\"majority_vote_step_sr\"]*100:<10.1f} {d[\"oracle_best_of_n_step_sr\"]*100:<10.1f} {\"—\":<8}')
"

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
