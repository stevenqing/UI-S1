#!/bin/bash
#SBATCH --job-name=gc_sft
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/gc_sft_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/gc_sft_%j.err
#SBATCH --time=02:30:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:1

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH
export TMPDIR=/tmp/${SLURM_JOB_ID}
mkdir -p $TMPDIR
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor
export TORCH_HOME=$TMPDIR/torch
export XDG_CACHE_HOME=$TMPDIR/cache
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TORCH_HOME $XDG_CACHE_HOME

echo "=== Gradient Conflict: SFT v2 ==="
echo "Job: ${SLURM_JOB_ID} | Node: $(hostname) | Start: $(date)"

python "$PROJECT_DIR/evaluation/gradient_conflict_analysis.py" \
    --model_path "$PROJECT_DIR/train_GUI_360/llamafactory/output/gui360_full_sft_v2" \
    --parquet_file "$PROJECT_DIR/train_GUI_360/data/gui360_eval_sft.parquet" \
    --image_base "$PROJECT_DIR" \
    --n_samples 200 --seed 42 \
    --output_dir "$PROJECT_DIR/evaluation/results/gradient_conflict_sft_v2_${TIMESTAMP}"

echo "Done: $(date)"
