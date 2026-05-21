#!/bin/bash
#SBATCH --job-name=eval_v6nt
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_v6_nothought_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_v6_nothought_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Evaluate Cooperative LoRA v6 (nothought) on 2 nodes × 4 GPUs
# ============================================
# Strategy:
#   - Node 0: serves vLLM (TP=4), runs "grounding" eval
#   - Node 1: serves vLLM (TP=4), runs "action_prediction" eval
#   - Both tasks run in parallel -> total time ~= max(grounding, action_pred)
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
EVAL_DIR="$TRAIN_DIR/GUI-360-eval"
DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"

BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
COOP_CKPT="$TRAIN_DIR/llamafactory/output/cooperative_v6_comm_nothought/final"
MERGED_MODEL="$PROJECT_DIR/checkpoints/cooperative_v6_nothought_merged"
RESULTS_DIR="$EVAL_DIR/results/cooperative_v6_nothought"

MODEL_TYPE="qwen2.5_vl_7b"
VLLM_PORT=19822
EVAL_THREADS=128

echo "=========================================="
echo "Cooperative LoRA v6 (nothought) Eval (2-node)"
echo "=========================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Nodes:       ${SLURM_NNODES}"
echo "Node list:   ${SLURM_NODELIST}"
echo "Submit host: $(hostname)"
echo "Start:       $(date)"
echo "=========================================="

# ── Step 1 (head node only): Merge LoRA into base model ──

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

if [ ! -d "$MERGED_MODEL" ] || [ ! -f "$MERGED_MODEL/model.safetensors.index.json" ]; then
    echo "Merging cooperative v6 LoRA (with communication) into base model..."
    python "$PROJECT_DIR/evaluation/merge_cooperative_lora.py" \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$COOP_CKPT" \
        --output_dir "$MERGED_MODEL"
    echo "Merge complete."
else
    echo "Merged model already exists at $MERGED_MODEL, skipping merge."
fi

# ── Step 2: Launch one vLLM + eval worker per node (parallel) ──

# Write a worker script that will run on each node.
WORKER_SCRIPT="$PROJECT_DIR/scripts/exp_cooperative/logs/eval_v6nt_worker_${SLURM_JOB_ID}.sh"
cat > "$WORKER_SCRIPT" << 'WORKEREOF'
#!/bin/bash
set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="$PROJECT_DIR/train_GUI_360/GUI-360-eval"
DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"
MERGED_MODEL="$PROJECT_DIR/checkpoints/cooperative_v6_nothought_merged"
RESULTS_DIR="$EVAL_DIR/results/cooperative_v6_nothought"
MODEL_TYPE="qwen2.5_vl_7b"
VLLM_PORT=19822
EVAL_THREADS=128

# Decide which eval type this node runs, based on SLURM_NODEID (0 or 1)
if [ "${SLURM_NODEID}" = "0" ]; then
    EVAL_TYPE="grounding"
else
    EVAL_TYPE="action_prediction"
fi

echo "[node ${SLURM_NODEID} / $(hostname)] Assigned EVAL_TYPE=${EVAL_TYPE}"

# Use eval env for vLLM
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

VLLM_LOG="$PROJECT_DIR/scripts/exp_cooperative/logs/vllm_v6nt_${SLURM_JOB_ID}_node${SLURM_NODEID}.log"

pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 3

echo "[node ${SLURM_NODEID}] Starting vLLM on $(hostname) port ${VLLM_PORT}, log: ${VLLM_LOG}"
python -m vllm.entrypoints.openai.api_server \
    --model "$MERGED_MODEL" \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "[node ${SLURM_NODEID}] vLLM PID=${VLLM_PID}"

# Wait for local server
MAX_WAIT=900
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "[node ${SLURM_NODEID}] vLLM ready after ${WAIT_COUNT}s"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "[node ${SLURM_NODEID}] ERROR: vLLM failed to start within ${MAX_WAIT}s"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

mkdir -p "${RESULTS_DIR}/${EVAL_TYPE}"
cd "$EVAL_DIR"

echo "[node ${SLURM_NODEID}] === Running ${EVAL_TYPE} ($(date)) ==="
python evaluation.py \
    --root_dir "$DATASET_ROOT" \
    --type "$EVAL_TYPE" \
    --model_type "$MODEL_TYPE" \
    --model_name "$MERGED_MODEL" \
    --api_url "http://localhost:${VLLM_PORT}/v1" \
    --threads $EVAL_THREADS \
    --output_dir "${RESULTS_DIR}/${EVAL_TYPE}"
echo "[node ${SLURM_NODEID}] === ${EVAL_TYPE} done ($(date)) ==="

kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true
WORKEREOF

chmod +x "$WORKER_SCRIPT"

echo "=========================================="
echo "Dispatching worker to each node via srun"
echo "=========================================="

# Run the worker on both nodes in parallel (one task per node, 4 GPUs each)
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=144 \
    bash "$WORKER_SCRIPT"

echo "=========================================="
echo "Both eval tasks completed: $(date)"
echo "Results dir: $RESULTS_DIR"
echo "=========================================="
echo "Summaries:"
find "$RESULTS_DIR" -name "evaluation_summary*.json" -exec echo "--- {} ---" \; -exec cat {} \;
echo ""
echo "Done: $(date)"
echo "=========================================="
