#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29531}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_NET=${NCCL_NET:-Socket}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-LOC}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-0}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export WANDB_MODE=${WANDB_MODE:-disabled}

OUT_DIR=${OUT_DIR:-checkpoints/v23_offline_grpo_full_sft}
LOG_DIR=${LOG_DIR:-outputs/v23_visual_transition/grpo_full_sft}
TRAIN_DATA=${TRAIN_DATA:-datasets/gui360-balanced/gui360_train_from_parquet.jsonl}
MODEL_PATH=${MODEL_PATH:-checkpoints/gui360-fullparam-sft-step250}
NPROC=${NPROC:-8}
EXTRA_ARGS=${EXTRA_ARGS:-}

mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "=========================================="
echo "V23 Offline GRPO from full SFT"
echo "Start:      $(date -Is)"
echo "Model:      $MODEL_PATH"
echo "Train data: $TRAIN_DATA"
echo "Output:     $OUT_DIR"
echo "Log dir:    $LOG_DIR"
echo "NPROC:      $NPROC"
echo "CUDA:       $CUDA_VISIBLE_DEVICES"
echo "Extra args: $EXTRA_ARGS"
echo "=========================================="

.venv-qwen3-vllm/bin/torchrun \
  --nproc_per_node="$NPROC" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  v15_gui_360/train_trajectory_gspo.py \
  --model_path "$MODEL_PATH" \
  --train_data "$TRAIN_DATA" \
  --output_dir "$OUT_DIR" \
  --lora_r 128 \
  --lora_alpha 256 \
  --lora_dropout 0.05 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --num_comm_rounds 2 \
  --balance_weight 0.01 \
  --image_max_pixels 602112 \
  --num_trajectories 8 \
  --temperature 0.7 \
  --top_p 0.95 \
  --clip_range 0.2 \
  --max_new_tokens 256 \
  --match_threshold 0.5 \
  --w_format 0.1 \
  --w_type 0.2 \
  --w_content 0.7 \
  --lora_lr 1e-5 \
  --route_lr 1e-3 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --num_epochs 1 \
  --gradient_accumulation_steps 4 \
  --logging_steps 1 \
  --save_steps 25 \
  --val_steps 0 \
  $EXTRA_ARGS

echo "=========================================="
echo "Done: $(date -Is)"
echo "=========================================="