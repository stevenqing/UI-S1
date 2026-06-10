#!/bin/bash
#SBATCH --job-name=ev_v22abl
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v22_memory_agent/scripts/logs/eval_ablation_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v22_memory_agent/scripts/logs/eval_ablation_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V22 Ablation: Isolate memory content vs template effect
#
# Experiments:
#   1. ablation_random_memory  — GUIDED template + random examples (template control)
#   2. memory_goal_fixed       — GUIDED template + goal memory (GT leak fixed)
#   3. memory_visual_fixed     — GUIDED template + visual memory (GT leak fixed)
#   4. memory_type_fixed       — GUIDED template + type memory (GT leak fixed)
#
# Comparison logic:
#   memory_goal_fixed vs ablation_random_memory → memory content effect
#   ablation_random_memory vs angle_standard (21.9%) → template effect
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_SFT="$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_balanced_full_sft/checkpoint-272"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
INDEX_DIR="$PROJECT_ROOT/v22_memory_agent/indices"
OUTPUT_BASE="$PROJECT_ROOT/v22_memory_agent/outputs"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p "$PROJECT_ROOT/v22_memory_agent/scripts/logs"

echo "=========================================="
echo "V22 Ablation: Memory Content vs Template"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

# Verify index exists
if [ ! -f "$INDEX_DIR/step_records.json" ]; then
    echo "ERROR: Memory index not built. Run build_index.slurm first."
    exit 1
fi

# ── Start vLLM server ──
python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_SFT" \
    --served-model-name "base_sft" \
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
    echo "ERROR: vLLM failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

COMMON_ARGS="--test_data $TEST_DATA --api_url http://localhost:8000/v1 --model_name base_sft --index_dir $INDEX_DIR --threads 32"

run_experiment() {
    local EXP="$1"
    local OUTPUT="$OUTPUT_BASE/$EXP"
    if ls "$OUTPUT"/summary_*.json 1>/dev/null 2>&1; then
        echo "[SKIP] $EXP — already done"
    else
        echo ""
        echo "=== $EXP ==="
        echo "Start: $(date)"
        mkdir -p "$OUTPUT"
        python "$PROJECT_ROOT/v22_memory_agent/eval_memory_reasoning.py" \
            --experiment "$EXP" \
            $COMMON_ARGS --output_dir "$OUTPUT_BASE"
        echo "Done: $(date)"
    fi
}

# ── Ablation runs ──
run_experiment "ablation_random_memory"
run_experiment "memory_goal_fixed"
run_experiment "memory_visual_fixed"
run_experiment "memory_type_fixed"

# ── Shutdown vLLM ──
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

# ── Print all summaries ──
echo ""
echo "=========================================="
echo "V22 ABLATION RESULTS"
echo "=========================================="
echo "Controls:"
echo "  standard SFT (no memory, STANDARD template): 21.9%"
echo "  memory_goal (original, GUIDED template):     21.5%"
echo "  memory_visual (original, GUIDED template):   22.5%"
echo ""

for EXP in ablation_random_memory memory_goal_fixed memory_visual_fixed memory_type_fixed; do
    OUTPUT="$OUTPUT_BASE/$EXP"
    S=$(ls "$OUTPUT"/summary_*.json 2>/dev/null | tail -1)
    if [ -n "$S" ]; then
        echo "--- ${EXP} ---"
        python3 -c "
import json
with open('$S') as f:
    d = json.load(f)
print(f\"  TSR={d['tsr']*100:.1f}%  StepSR={d['step_sr']*100:.1f}%  Progress={d['avg_progress']*100:.1f}%\")
tl = d.get('task_length_metrics', {})
for k in ['1','2-3','4-5','6+']:
    if k in tl:
        m = tl[k]
        print(f\"  {k:>5s}: TSR={m['tsr']*100:5.1f}% (n={m['num_episodes']})\")
"
        echo ""
    fi
done

echo "Interpretation:"
echo "  random_memory vs standard(21.9%) = template effect"
echo "  memory_goal_fixed vs random_memory = memory content effect"
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
