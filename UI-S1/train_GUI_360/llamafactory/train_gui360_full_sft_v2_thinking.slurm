#!/bin/bash
#SBATCH --job-name=gui360_sft_v2_think
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/gui360_sft_v2_thinking_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/gui360_sft_v2_thinking_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# GUI-360 Full-Parameter SFT v2 with Thinking
#
# Same as gui360_full_sft_v2 but with
# "Reasoning: {thought}" added to assistant responses.
# This trains the model to output reasoning before <tool_call>.
#
# Step 1: Prepare with-thinking data (on head node)
# Step 2: Run SFT training (all nodes)
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
LLAMA_FACTORY_DIR="$PROJECT_DIR/train_GUI_360/LLaMA-Factory"
CONFIG_FILE="$PROJECT_DIR/train_GUI_360/llamafactory/qwen25vl_gui360_full_sft_v2_thinking.yaml"
LOG_DIR="$PROJECT_DIR/train_GUI_360/logs"
DATA_DIR="$PROJECT_DIR/train_GUI_360/llamafactory/data"

TRAIN_INPUT="$DATA_DIR/gui360_train.json"
TRAIN_OUTPUT="$DATA_DIR/gui360_train_with_thinking.json"
VAL_INPUT="$DATA_DIR/gui360_val.json"
VAL_OUTPUT="$DATA_DIR/gui360_val_with_thinking.json"
RAW_DATA_DIR="$PROJECT_DIR/datasets/GUI-360/train/data"

echo "=========================================="
echo "GUI-360 Full SFT v2 with Thinking"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Nodes: $SLURM_NNODES"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

if [ -f "$PROJECT_DIR/train/env_config.sh" ]; then
    source "$PROJECT_DIR/train/env_config.sh"
fi

mkdir -p "$LOG_DIR"

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200
export TORCH_GLOO_TIMEOUT_SECONDS=7200

echo "Master node: $MASTER_ADDR"
echo "Config file: $CONFIG_FILE"

# ============================================
# Step 1: Prepare with-thinking data (head node only)
# ============================================
if [ "$SLURM_NODEID" == "0" ] || [ -z "$SLURM_NODEID" ]; then
    echo "=========================================="
    echo "Step 1: Preparing with-thinking data"
    echo "=========================================="

    if [ ! -f "$TRAIN_OUTPUT" ]; then
        echo "Generating training data with thinking..."
        python "$PROJECT_DIR/train_GUI_360/data_preparation/prepare_gui360_sft_with_thinking_llamafactory.py" \
            --input "$TRAIN_INPUT" \
            --raw-data-dir "$RAW_DATA_DIR" \
            --output "$TRAIN_OUTPUT"
        echo "Training data ready: $TRAIN_OUTPUT"
    else
        echo "Training data already exists: $TRAIN_OUTPUT"
    fi

    if [ ! -f "$VAL_OUTPUT" ]; then
        echo "Generating validation data with thinking..."
        python "$PROJECT_DIR/train_GUI_360/data_preparation/prepare_gui360_sft_with_thinking_llamafactory.py" \
            --input "$VAL_INPUT" \
            --raw-data-dir "$RAW_DATA_DIR" \
            --output "$VAL_OUTPUT"
        echo "Validation data ready: $VAL_OUTPUT"
    else
        echo "Validation data already exists: $VAL_OUTPUT"
    fi

    echo "Data preparation complete."
fi

# Wait for data prep to finish
sleep 10

# ============================================
# Step 2: Run Training
# ============================================
echo "=========================================="
echo "Step 2: Starting SFT training"
echo "=========================================="

cd "$PROJECT_DIR"

srun --ntasks-per-node=1 bash -c '
    export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
    export CONDA_DEFAULT_ENV=qwen3-eval
    export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
    export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    if [ -f '"$PROJECT_DIR"'/train/env_config.sh ]; then
        source '"$PROJECT_DIR"'/train/env_config.sh
    fi

    export MASTER_ADDR='"$MASTER_ADDR"'
    export MASTER_PORT='"$MASTER_PORT"'
    export NODE_RANK=$SLURM_NODEID

    export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_${SLURM_JOB_ID}
    export HF_HOME=/tmp/hf_home_${SLURM_JOB_ID}
    mkdir -p $HF_DATASETS_CACHE $HF_HOME

    cd '"$PROJECT_DIR"'

    FORCE_TORCHRUN=1 \
    NNODES=$SLURM_NNODES \
    NODE_RANK=$SLURM_NODEID \
    MASTER_ADDR=$MASTER_ADDR \
    MASTER_PORT=$MASTER_PORT \
    NPROC_PER_NODE=4 \
    llamafactory-cli train '"$CONFIG_FILE"'
'
TRAIN_EXIT_CODE=$?

echo "=========================================="
echo "Training Completed"
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "Output dir: $PROJECT_DIR/train_GUI_360/llamafactory/output/gui360_full_sft_v2_thinking"
echo "=========================================="

exit $TRAIN_EXIT_CODE
