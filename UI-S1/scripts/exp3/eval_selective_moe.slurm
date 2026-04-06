#!/bin/bash
#SBATCH --job-name=smoe_eval
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp3/logs/eval_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp3/logs/eval_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Selective MoE Evaluation
# Single-node HF generate (no vLLM)
# Pass --condition c1b|c2|c3|c4|c5
# ============================================

set -e

# Parse condition from command line: sbatch --export=CONDITION=c2 eval_selective_moe.slurm
CONDITION=${CONDITION:-c2}

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CHECKPOINT_DIR="$PROJECT_DIR/train_GUI_360/llamafactory/output/selective_moe_${CONDITION}/final/moe_checkpoint"
EVAL_DATA="$PROJECT_DIR/train_GUI_360/llamafactory/data/gui360_val.json"
OUTPUT_DIR="$PROJECT_DIR/scripts/exp3/results/selective_moe_${CONDITION}"
LOG_DIR="$PROJECT_DIR/scripts/exp3/logs"

echo "=========================================="
echo "Selective MoE Evaluation: ${CONDITION}"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

if [ -f "$PROJECT_DIR/train/env_config.sh" ]; then
    source "$PROJECT_DIR/train/env_config.sh"
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TMPDIR=/tmp
export HF_HOME=/tmp/hf_home_${SLURM_JOB_ID}
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_${SLURM_JOB_ID}
mkdir -p $HF_HOME $HF_DATASETS_CACHE

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"

python evaluation/selective_moe_eval.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --eval_data "$EVAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --max_new_tokens 512

echo "=========================================="
echo "Evaluation Completed"
echo "End: $(date)"
echo "Results: $OUTPUT_DIR"
echo "=========================================="

# Print summary
cat "$OUTPUT_DIR/eval_summary.json" 2>/dev/null || echo "No summary found"
