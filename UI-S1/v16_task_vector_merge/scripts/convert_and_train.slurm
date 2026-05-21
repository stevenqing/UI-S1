#!/bin/bash
#SBATCH --job-name=v16_conv
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v16_task_vector_merge/scripts/logs/convert_and_train_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v16_task_vector_merge/scripts/logs/convert_and_train_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:1

# ============================================
# Step 1: Convert gui_grounding + gui_action → V16 cooperative checkpoint
# Step 2: Submit sequential and parallel RL training jobs
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="${PROJECT_DIR}:$PYTHONPATH"
export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"

mkdir -p v16_task_vector_merge/scripts/logs

echo "=========================================="
echo "V16 Joint SVD Conversion"
echo "  gui_grounding (expert 1) + gui_action (expert 2)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start:  $(date)"
echo "=========================================="

python v16_task_vector_merge/convert_fullsft_to_cooperative.py \
    --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
    --model_a checkpoints/gui_grounding \
    --model_b checkpoints/gui_action \
    --output_dir checkpoints/v16_coop_r128_attn \
    --rank 128 \
    --alpha 256 \
    --num_comm_rounds 2 \
    --target_modules q_proj,k_proj,v_proj,o_proj

echo ""
echo "=========================================="
echo "Conversion done. Submitting training jobs..."
echo "=========================================="

# Submit sequential RL
JOB_SEQ=$(sbatch --parsable v16_task_vector_merge/scripts/train_sequential_rl.slurm)
echo "Submitted sequential RL: job ${JOB_SEQ}"

# Submit parallel RL
JOB_PAR=$(sbatch --parsable v16_task_vector_merge/scripts/train_parallel_rl.slurm)
echo "Submitted parallel RL: job ${JOB_PAR}"

echo ""
echo "=========================================="
echo "All jobs submitted."
echo "  Sequential RL: ${JOB_SEQ}"
echo "  Parallel RL:   ${JOB_PAR}"
echo "Done: $(date)"
echo "=========================================="
