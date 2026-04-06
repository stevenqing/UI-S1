#!/bin/bash
#SBATCH --job-name=phase0_steer
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp3/logs/phase0_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp3/logs/phase0_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

# ============================================
# Phase 0: Inference-time Attention Steering
# Run one mode per job (set MODE env var)
# ============================================

set -e

MODE="${MODE:-baseline}"
ALPHA="${ALPHA:-2.0}"
BETA="${BETA:-0.5}"
N_SAMPLES="${N_SAMPLES:-1000}"

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
MODEL_PATH="$PROJECT_DIR/train_GUI_360/llamafactory/output/gui360_full_sft_v2"
EVAL_DIR="$PROJECT_DIR/evaluation"
PARQUET="$PROJECT_DIR/train_GUI_360/data/gui360_test_sft_eval_format_with_bbox.parquet"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_DIR/scripts/exp3/results/phase0_${MODE}_a${ALPHA}_b${BETA}_${TIMESTAMP}"

echo "=========================================="
echo "Phase 0: Attention Steering"
echo "=========================================="
echo "Job: ${SLURM_JOB_ID} | Node: $(hostname)"
echo "Mode: ${MODE} | Alpha: ${ALPHA} | Beta: ${BETA}"
echo "Samples: ${N_SAMPLES}"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Start: $(date)"
echo "=========================================="

# Environment
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

export TMPDIR=/tmp/${SLURM_JOB_ID}
mkdir -p $TMPDIR
export HF_HOME=$TMPDIR/hf_home
export HF_DATASETS_CACHE=$TMPDIR/hf_datasets
mkdir -p $HF_HOME $HF_DATASETS_CACHE

# Validate
if [ ! -f "$MODEL_PATH/model.safetensors.index.json" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

cd "$EVAL_DIR"

python attention_steering_eval.py \
    --model_path "$MODEL_PATH" \
    --parquet_file "$PARQUET" \
    --output_dir "$OUTPUT_DIR" \
    --mode "$MODE" \
    --alpha "$ALPHA" \
    --beta "$BETA" \
    --n_samples "$N_SAMPLES" \
    --bind_layer 24 \
    --top_p 0.2 \
    --layer_start 19 \
    --layer_end 27

echo ""
echo "=========================================="
echo "Phase 0 Complete: ${MODE}"
echo "End: $(date)"
echo "Results: ${OUTPUT_DIR}"
echo "=========================================="
cat "${OUTPUT_DIR}/${MODE}_summary.json" 2>/dev/null | python3 -m json.tool || true
