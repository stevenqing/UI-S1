#!/bin/bash
#SBATCH --job-name=exp0-data
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/outputs/exp0_data_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/outputs/exp0_data_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# =============================================================================
# Experiment 0.3 (Oracle Recovery) + 0.4 (Per-Domain FCV)
# Pure data analysis — no GPU needed.
# =============================================================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
cd "${PROJECT_DIR}"

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

mkdir -p outputs/exp0_3 outputs/exp0_4

echo "=========================================="
echo "Experiment 0.x: Data Analysis"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# Step 0: Build pair index cache (if not exists)
echo ""
echo "--- Building pair index cache ---"
python3 -c "
from scripts.exp0.data_utils import _build_request_index
index = _build_request_index(rebuild=False)
paired = sum(1 for v in index.values() if v.get('success') and v.get('fail'))
print(f'Index ready: {paired} paired requests')
"

# Step 1: Experiment 0.3 — SKIPPED (already completed)
# python3 scripts/exp0/exp0_3_oracle_recovery.py --max_pairs 500 --output_dir outputs/exp0_3

# Step 2: Experiment 0.4 — Per-Domain FCV Curves
echo ""
echo "=========================================="
echo "Experiment 0.4: Per-Domain FCV Curves"
echo "=========================================="
python3 scripts/exp0/exp0_4_domain_fcv.py \
    --max_pairs 2000 \
    --max_steps 30 \
    --output_dir outputs/exp0_4

echo ""
echo "=========================================="
echo "All data analysis experiments complete."
echo "End: $(date)"
echo "Results: outputs/exp0_3/ outputs/exp0_4/"
echo "=========================================="
