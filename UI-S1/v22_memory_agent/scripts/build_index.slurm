#!/bin/bash
#SBATCH --job-name=v22_index
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v22_memory_agent/scripts/logs/build_index_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v22_memory_agent/scripts/logs/build_index_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ============================================
# V22: Build Memory Index (CPU only, ~5 min)
#
# Builds TF-IDF, type, and metadata indices from
# balanced 2000 training episodes.
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p "$PROJECT_ROOT/v22_memory_agent/scripts/logs"
mkdir -p "$PROJECT_ROOT/v22_memory_agent/indices"

echo "=========================================="
echo "V22: Build Memory Index"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

python "$PROJECT_ROOT/v22_memory_agent/build_memory_index.py" \
    --balanced_data "$PROJECT_ROOT/v12_gui_360/data/gui360_train_2000_balanced.jsonl" \
    --rl_data "$PROJECT_ROOT/datasets/GUI-360/rl_data/gui360_train.jsonl" \
    --output_dir "$PROJECT_ROOT/v22_memory_agent/indices" \
    --max_tfidf_features 5000

echo ""
echo "=========================================="
echo "Index build complete: $(date)"
echo "=========================================="
