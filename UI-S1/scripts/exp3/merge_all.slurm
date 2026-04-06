#!/bin/bash
#SBATCH --job-name=merge_all_smoe
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp3/logs/merge_all_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp3/logs/merge_all_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Merge all 5 Selective MoE conditions into full models for vLLM

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

echo "=== Merge All Selective MoE Conditions ==="
echo "Job: ${SLURM_JOB_ID} | Node: $(hostname) | Start: $(date)"

for COND in c1b c2 c3 c4 c5; do
    CKPT_DIR="$PROJECT_DIR/train_GUI_360/llamafactory/output/selective_moe_${COND}/final/moe_checkpoint"
    OUT_DIR="$PROJECT_DIR/train_GUI_360/llamafactory/output/selective_moe_${COND}/merged_model"

    if [ -f "$OUT_DIR/model.safetensors.index.json" ]; then
        echo ""
        echo "--- ${COND}: already merged, skipping ---"
        continue
    fi

    echo ""
    echo "--- Merging ${COND} ---"
    echo "Start: $(date)"

    python "$PROJECT_DIR/evaluation/merge_moe_to_full.py" \
        --base_model "$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct" \
        --checkpoint_dir "$CKPT_DIR" \
        --output_dir "$OUT_DIR"

    echo "${COND} done: $(date)"
done

echo ""
echo "=== All merges complete ==="
echo "End: $(date)"

# Verify
for COND in c1b c2 c3 c4 c5; do
    OUT_DIR="$PROJECT_DIR/train_GUI_360/llamafactory/output/selective_moe_${COND}/merged_model"
    if [ -f "$OUT_DIR/model.safetensors.index.json" ]; then
        SIZE=$(du -sh "$OUT_DIR" | cut -f1)
        echo "  ${COND}: OK ($SIZE)"
    else
        echo "  ${COND}: MISSING"
    fi
done
