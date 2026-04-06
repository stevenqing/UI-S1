#!/bin/bash
#SBATCH --job-name=exp_def
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/exp_def_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/exp_def_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="$PROJECT_DIR/evaluation"

MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/train_GUI_360/llamafactory/output/gui360_full_sft_v2/checkpoint-796}"
EXP="${EXP:?Must set EXP (ef or d)}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "Exp $EXP"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Model: $MODEL_PATH"
echo "============================================================"

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export PYTHONUNBUFFERED=1

export TMPDIR=/tmp/${SLURM_JOB_ID}
mkdir -p $TMPDIR 2>/dev/null || true
if ! touch "$TMPDIR/.test_write" 2>/dev/null; then
    export TMPDIR=/scratch/a5l/shuqing.a5l/tmp/${SLURM_JOB_ID}
    mkdir -p $TMPDIR
fi
rm -f "$TMPDIR/.test_write" 2>/dev/null
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor
export TORCH_HOME=$TMPDIR/torch
export XDG_CACHE_HOME=$TMPDIR/cache
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TORCH_HOME $XDG_CACHE_HOME

cd "$EVAL_DIR"

if [ "$EXP" = "ef" ]; then
    # Exp E + F: Binding Analysis
    OUTPUT_DIR="$PROJECT_DIR/evaluation/results/exp_ef_${TIMESTAMP}"
    mkdir -p "$OUTPUT_DIR"
    python -u binding_analysis.py \
        --model_path "$MODEL_PATH" \
        --parquet_file "$PROJECT_DIR/train_GUI_360/data/gui360_test_sft_eval_format_with_bbox.parquet" \
        --n_samples 300 \
        --output_dir "$OUTPUT_DIR"

elif [ "$EXP" = "d" ]; then
    # Exp D: Micro Grounding SFT + Probe C
    OUTPUT_DIR="$PROJECT_DIR/evaluation/results/exp_d_${TIMESTAMP}"
    mkdir -p "$OUTPUT_DIR"
    python -u train_micro_grounding.py \
        --base_model "$MODEL_PATH" \
        --data_path "$PROJECT_DIR/evaluation/dataset/gui360_grounding_200_simple.jsonl" \
        --output_dir "$OUTPUT_DIR" \
        --lr 2e-4 \
        --epochs 1 \
        --lora_rank 16 \
        --then_probe \
        --probe_n 200
fi

echo "Done at $(date)"
