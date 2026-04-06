#!/bin/bash
#SBATCH --job-name=exp_gamma
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/exp_gamma_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/exp_gamma_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Exp γ: RL Diagnostic Battery on AndroidControl
#
# Evaluates V6-merged, Base, and UI-S1-GRPO on AC step-level
# Tests: per-function accuracy (narrowing), click error decomp
#
# Usage:
#   sbatch scripts/eval/eval_exp_gamma_rl_diagnostic.slurm
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_SCRIPT="$PROJECT_DIR/evaluation/eval_exp_gamma_rl_diagnostic.py"
LOG_DIR="$PROJECT_DIR/scripts/eval/logs"
AC_JSONL="$PROJECT_DIR/datasets/android_control_evaluation_std.jsonl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
N_SAMPLES=1000
WORKERS=16
PORT=8000

# Models to evaluate
declare -A MODELS=(
    ["base"]="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
    ["v6_step120"]="$PROJECT_DIR/checkpoints/sp_gigpo_ac/sp_gigpo_spwa_k8_v6/merged_step_120"
    ["uis1_grpo"]="$PROJECT_DIR/checkpoints/UI-S1-GRPO-Trained"
)

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Exp γ: RL Diagnostic Battery"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "N_SAMPLES: $N_SAMPLES"
echo "Models: ${!MODELS[@]}"
echo "=========================================="

# Environment
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

# CUDA library path (needed for vLLM on this cluster)
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH
# Force vLLM V0 engine for stability
export VLLM_USE_V1=0

# Fix temp dirs
export TMPDIR=/tmp/${SLURM_JOB_ID}
mkdir -p $TMPDIR
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor
export TORCH_HOME=$TMPDIR/torch
export XDG_CACHE_HOME=$TMPDIR/cache
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TORCH_HOME $XDG_CACHE_HOME

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"

# ── Helper function ──
run_eval() {
    local MODEL_KEY=$1
    local MODEL_PATH=$2

    echo ""
    echo "=========================================="
    echo "Evaluating: $MODEL_KEY"
    echo "Model: $MODEL_PATH"
    echo "=========================================="

    if [ ! -d "$MODEL_PATH" ]; then
        echo "ERROR: Model not found at $MODEL_PATH, skipping"
        return 1
    fi

    # Stop any existing vLLM
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 3

    # Start vLLM
    echo "Starting vLLM for $MODEL_KEY..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port $PORT \
        --tensor-parallel-size 4 \
        --max-model-len 16384 \
        --trust-remote-code \
        --gpu-memory-utilization 0.85 \
        2>&1 | tee "$LOG_DIR/vllm_gamma_${MODEL_KEY}_${SLURM_JOB_ID}.log" &
    VLLM_PID=$!

    # Wait for vLLM
    MAX_WAIT=300
    WAIT_COUNT=0
    while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "vLLM ready!"
            break
        fi
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
        echo "  Waiting... ($WAIT_COUNT/$MAX_WAIT)"
    done

    if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM failed to start for $MODEL_KEY"
        kill $VLLM_PID 2>/dev/null || true
        return 1
    fi

    # Run diagnostic
    OUTPUT_DIR="$PROJECT_DIR/evaluation/results/exp_gamma_${MODEL_KEY}_${TIMESTAMP}"
    echo "Output: $OUTPUT_DIR"

    python "$EVAL_SCRIPT" run \
        --model_name "$MODEL_PATH" \
        --jsonl_file "$AC_JSONL" \
        --output_dir "$OUTPUT_DIR" \
        --n_samples $N_SAMPLES \
        --max_workers $WORKERS \
        --port $PORT

    echo ""
    echo "=== $MODEL_KEY Results ==="
    cat "$OUTPUT_DIR/report.txt" 2>/dev/null | head -60
    echo ""

    # Cleanup vLLM
    kill $VLLM_PID 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 5
}

# ── Run all models sequentially ──
for MODEL_KEY in base v6_step120 uis1_grpo; do
    MODEL_PATH="${MODELS[$MODEL_KEY]}"
    run_eval "$MODEL_KEY" "$MODEL_PATH" || echo "WARNING: $MODEL_KEY evaluation failed"
done

echo ""
echo "=========================================="
echo "Exp γ Complete"
echo "End: $(date)"
echo "=========================================="

# ── Cross-model comparison ──
echo ""
echo "=== Cross-Model Comparison ==="
echo ""

python3 -c "
import json, os, glob

results_dir = '$PROJECT_DIR/evaluation/results'
pattern = f'{results_dir}/exp_gamma_*_$TIMESTAMP/summary.json'
files = sorted(glob.glob(pattern))

print(f'Found {len(files)} result files')
print()
print(f'{\"Model\":<20} {\"TypeM\":>7} {\"ExtractM\":>9} {\"ParseF\":>7} {\"Active\":>7}')
print('-' * 55)

for f in files:
    name = os.path.basename(os.path.dirname(f)).replace('exp_gamma_', '').replace('_$TIMESTAMP', '')
    with open(f) as fh:
        s = json.load(fh)
    print(f'{name:<20} {100*s[\"type_match_rate\"]:>6.1f}% {100*s[\"extract_match_rate\"]:>8.1f}% {100*s[\"parse_fail_rate\"]:>6.1f}% {s[\"n_active_functions\"]:>3}/{s[\"n_total_functions\"]}')

    # Per-function detail
    for pf in s.get('per_function', []):
        print(f'  {pf[\"function\"]:<14} n={pf[\"n\"]:>4}  type={pf[\"type_match\"]:>5.1f}%  extract={pf[\"extract_match\"]:>5.1f}%')
    print()
"

echo "=========================================="
echo "All done. Check results in:"
echo "  $PROJECT_DIR/evaluation/results/exp_gamma_*_$TIMESTAMP/"
echo "=========================================="
