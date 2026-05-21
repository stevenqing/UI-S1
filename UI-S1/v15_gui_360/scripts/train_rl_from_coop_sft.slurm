#!/bin/bash
#SBATCH --job-name=v15_rl_wsft
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/train_rl_from_coop_sft_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/train_rl_from_coop_sft_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4
# ============================================
# V15 GUI-360: Cooperative RL from Cooperative SFT Warmstart
#
# Same config as V13 cold-start RL (train_rl_balanced.slurm) but
# initialises cooperative LoRA weights from the best SFT checkpoint.
#
# Best epoch determined from V15 cooperative SFT eval:
#   epoch-0: TSR=3.8%, epoch-1: 6.6%, epoch-2: 10.5% (best)
#
#   nodes=4  ngpu/node=4  grad_accum=4  ->  eff_bs = 4*4*4 = 64 episodes
#   lora_lr=1e-5, route_lr=1e-3 (also used for comm params)
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_OUTPUT_ROOT="$PROJECT_DIR/v15_gui_360/outputs"
CKPT_ROOT="$PROJECT_DIR/checkpoints/v15_balanced_cooperative_sft"

# ---- Auto-select best SFT epoch by TSR ----
select_best_epoch() {
    local best_epoch=""
    local best_tsr=-1

    for ep in epoch-0 epoch-1 epoch-2 epoch-3; do
        local summary_dir="$EVAL_OUTPUT_ROOT/eval_coop_sft_${ep}"
        local summary_file=$(ls "$summary_dir"/eval_summary_*.json 2>/dev/null | tail -1)
        if [ -z "$summary_file" ]; then
            echo "WARNING: No eval summary for $ep, skipping." >&2
            continue
        fi
        local tsr=$(python3 -c "import json; d=json.load(open('$summary_file')); print(d['tsr'])")
        echo "  $ep: TSR=$tsr" >&2
        local is_better=$(python3 -c "print(1 if $tsr > $best_tsr else 0)")
        if [ "$is_better" = "1" ]; then
            best_tsr=$tsr
            best_epoch=$ep
        fi
    done

    if [ -z "$best_epoch" ]; then
        echo "ERROR: No valid eval summaries found!" >&2
        exit 1
    fi

    echo "  => Best: $best_epoch (TSR=$best_tsr)" >&2
    echo "$best_epoch"
}

echo "=========================================="
echo "V15 RL from Cooperative SFT Warmstart"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Start:   $(date)"
echo "=========================================="

echo ""
echo "Using pre-selected best SFT epoch (epoch-2, TSR=10.5%)..."
BEST_EPOCH="epoch-2"
SFT_CKPT="$CKPT_ROOT/${BEST_EPOCH}/cooperative"

echo "Using SFT checkpoint: $SFT_CKPT"
echo ""

# Verify checkpoint files exist
if [ ! -f "$SFT_CKPT/lora_weights.pt" ]; then
    echo "ERROR: lora_weights.pt not found at $SFT_CKPT"
    exit 1
fi

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29502
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200

OUTPUT_DIR="${PROJECT_DIR}/checkpoints/v15_rl_from_coop_sft"

srun --ntasks-per-node=1 bash -c '
    export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
    export CONDA_DEFAULT_ENV=ui-s1
    export PYTHONPATH="'"$PROJECT_DIR"':$PYTHONPATH"
    export PYTHONUNBUFFERED=1
    export WANDB_MODE=disabled

    cd "'"$PROJECT_DIR"'"

    torchrun \
        --nproc_per_node=4 \
        --nnodes=$SLURM_NNODES \
        --node_rank=$SLURM_NODEID \
        --master_addr='"$MASTER_ADDR"' \
        --master_port='"$MASTER_PORT"' \
        v13_gui_360/train_trajectory_rl.py \
        --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
        --sft_checkpoint '"$SFT_CKPT"' \
        --train_data v12_gui_360/data/gui360_train_2000_balanced.jsonl \
        --val_data v12_gui_360/data/gui360_val_50.jsonl \
        --output_dir '"$OUTPUT_DIR"' \
        --lora_r 128 \
        --lora_alpha 256 \
        --lora_dropout 0.05 \
        --target_modules q_proj k_proj v_proj o_proj \
        --num_comm_rounds 2 \
        --balance_weight 0.01 \
        --routing_noise_scale 0.5 \
        --image_max_pixels 602112 \
        --num_samples 8 \
        --temperature 1.0 \
        --top_p 0.95 \
        --clip_range 0.2 \
        --kl_coef 0.001 \
        --max_new_tokens 256 \
        --match_threshold 0.5 \
        --spwa_decay 0.5 \
        --dapo_threshold 0.0 \
        --step_adv_weight 0.5 \
        --w_format 0.1 \
        --w_type 0.2 \
        --w_content 0.7 \
        --lora_lr 1e-5 \
        --route_lr 1e-3 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --num_epochs 3 \
        --gradient_accumulation_steps 4 \
        --logging_steps 1 \
        --save_steps 25 \
        --val_steps 25
'

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
