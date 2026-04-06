#!/bin/bash
#SBATCH --job-name=probe_c
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp3/logs/probe_c_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp3/logs/probe_c_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

# ============================================
# Run Probe C (Cross-Modal Alignment) on a model checkpoint
#
# Usage:
#   MODEL_PATH=/path/to/model N_SAMPLES=300 sbatch scripts/exp3/run_probe_c.slurm
#   MODEL_PATH=/path/to/model LORA_PATH=/path/to/lora sbatch scripts/exp3/run_probe_c.slurm
# ============================================

set -e

MODEL_PATH=${MODEL_PATH:-/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/llamafactory/output/gui360_full_sft_v2}
LORA_PATH=${LORA_PATH:-}
N_SAMPLES=${N_SAMPLES:-300}
LABEL=${LABEL:-sftv2}

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
OUTPUT_DIR="$PROJECT_DIR/scripts/exp3/results/probe_c_${LABEL}_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "Probe C: Cross-Modal Alignment Diagnostic"
echo "=========================================="
echo "Job: ${SLURM_JOB_ID} | Node: $(hostname)"
echo "Model: ${MODEL_PATH}"
echo "LoRA: ${LORA_PATH:-none}"
echo "Samples: ${N_SAMPLES}"
echo "Output: ${OUTPUT_DIR}"
echo "Start: $(date)"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

if [ -f "$PROJECT_DIR/train/env_config.sh" ]; then
    source "$PROJECT_DIR/train/env_config.sh"
fi

cd "$PROJECT_DIR"

LORA_ARG=""
if [ -n "$LORA_PATH" ]; then
    LORA_ARG="--lora_path $LORA_PATH"
fi

python evaluation/probing_diagnostic.py \
    --model_path "$MODEL_PATH" \
    $LORA_ARG \
    --parquet_file train_GUI_360/data/gui360_test_sft_eval_format_with_bbox.parquet \
    --n_samples $N_SAMPLES \
    --output_dir "$OUTPUT_DIR"

echo "=========================================="
echo "Probe C Complete"
echo "End: $(date)"
echo "Results: ${OUTPUT_DIR}"
echo "=========================================="
