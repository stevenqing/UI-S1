#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29561}
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

OUT_DIR=${OUT_DIR:-checkpoints/v23_offline_candidate_grpo_full_sft}
LOG_DIR=${LOG_DIR:-outputs/v23_visual_transition/offline_candidate_grpo_full_sft}
MODEL_PATH=${MODEL_PATH:-checkpoints/gui360-fullparam-sft-step250}
EPISODE_DATA=${EPISODE_DATA:-datasets/gui360-balanced/gui360_train_from_parquet.jsonl}
CANDIDATE_DATA=${CANDIDATE_DATA:-outputs/v23_visual_transition/train_full_sft_k8_candidates/matcher_candidates.jsonl}
NPROC=${NPROC:-8}
EXTRA_ARGS=${EXTRA_ARGS:-}

mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "=========================================="
echo "V23 Offline Candidate GRPO from full SFT"
echo "Start:          $(date -Is)"
echo "Model:          $MODEL_PATH"
echo "Episode data:   $EPISODE_DATA"
echo "Candidate data: $CANDIDATE_DATA"
echo "Output:         $OUT_DIR"
echo "Log dir:        $LOG_DIR"
echo "NPROC:          $NPROC"
echo "CUDA:           $CUDA_VISIBLE_DEVICES"
echo "Extra args:     $EXTRA_ARGS"
echo "=========================================="

.venv-qwen3-vllm/bin/torchrun \
  --nproc_per_node="$NPROC" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  v23_visual_transition/train_offline_grpo.py \
  --model_path "$MODEL_PATH" \
  --episode_data "$EPISODE_DATA" \
  --candidate_data "$CANDIDATE_DATA" \
  --output_dir "$OUT_DIR" \
  --lora_r 128 \
  --lora_alpha 256 \
  --lora_dropout 0.05 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --num_comm_rounds 2 \
  --balance_weight 0.01 \
  --image_max_pixels 602112 \
  --include_gt_candidate \
  --gt_reward 1.0 \
  --sft_anchor_weight 0.25 \
  --advantage_clip 5.0 \
  --weight_clip 5.0 \
  --lora_lr 1e-5 \
  --route_lr 1e-3 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --num_epochs 1 \
  --gradient_accumulation_steps 4 \
  --logging_steps 1 \
  --save_steps 25 \
  $EXTRA_ARGS

echo "=========================================="
echo "Done: $(date -Is)"
echo "=========================================="