#!/usr/bin/env bash
# Train the text-only Verifier Agent on class-balanced candidate-packet SFT data.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}
MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/checkpoints/Qwen3.5-9B}
TRAIN_PARQUET=${TRAIN_PARQUET:-$PROJECT_DIR/datasets/verifier_agent_gui_odyssey_sft_balanced/train_balanced.parquet}
VAL_PARQUET=${VAL_PARQUET:-$PROJECT_DIR/datasets/verifier_agent_gui_odyssey_sft_balanced/dev.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/outputs/verifier_agent_sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-verifier_agent_qwen35_lora}
N_GPUS=${N_GPUS:-1}
MASTER_PORT=${MASTER_PORT:-29611}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
MAX_LENGTH=${MAX_LENGTH:-2048}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
LR=${LR:-1e-5}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LOGGER=${LOGGER:-console}
MODEL_DTYPE=${MODEL_DTYPE:-fp32}
TRANSFORMER_LAYER_CLS=${TRANSFORMER_LAYER_CLS:-Qwen3_5DecoderLayer}
MODEL_ATTN_IMPLEMENTATION=${MODEL_ATTN_IMPLEMENTATION:-eager}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:-100}

case "$OUTPUT_DIR" in
  /*) ;;
  *) OUTPUT_DIR="$PROJECT_DIR/$OUTPUT_DIR" ;;
esac

CONFIG_DIR="$OUTPUT_DIR/config"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
LOG_DIR="$OUTPUT_DIR/logs"
CONFIG_NAME="verifier_agent_sft"

mkdir -p "$CONFIG_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

if [[ ! -f "$TRAIN_PARQUET" ]]; then
  echo "Missing TRAIN_PARQUET=$TRAIN_PARQUET" >&2
  echo "Run scripts/prepare_verifier_agent_sft_data.py first." >&2
  exit 1
fi
if [[ ! -f "$VAL_PARQUET" ]]; then
  echo "Missing VAL_PARQUET=$VAL_PARQUET" >&2
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Missing MODEL_PATH=$MODEL_PATH" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing executable PYTHON_BIN=$PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
from packaging import version
import transformers
minimum = version.parse("5.12.1")
current = version.parse(transformers.__version__)
if current < minimum:
    raise SystemExit(f"transformers>={minimum} is required for Qwen3.5, found {transformers.__version__}. Run: /home/aiscuser/.local/bin/uv pip install --python .venv/bin/python 'transformers==5.12.1'")
print(f"Transformers: {transformers.__version__}")
PY

cat > "$CONFIG_DIR/$CONFIG_NAME.yaml" <<CONFIGEOF
hydra:
  searchpath:
    - file://$PROJECT_DIR/verl/trainer/config

defaults:
  - sft_trainer
  - _self_

data:
  train_files: $TRAIN_PARQUET
  val_files: $VAL_PARQUET
  train_batch_size: $TRAIN_BATCH_SIZE
  micro_batch_size_per_gpu: $MICRO_BATCH_SIZE
  max_length: $MAX_LENGTH
  truncation: error
  balance_dp_token: False
  multiturn:
    enable: true
    messages_key: messages
    tools_key: tools
    enable_thinking_key: enable_thinking
  custom_cls:
    path: $PROJECT_DIR/verl/utils/dataset/gui_multiturn_sft_dataset.py
    name: GUIMultiTurnSFTDataset
  trust_remote_code: true

model:
  partial_pretrain: $MODEL_PATH
  strategy: fsdp2
  fsdp_config:
    model_dtype: $MODEL_DTYPE
    wrap_policy:
      transformer_layer_cls_to_wrap: [$TRANSFORMER_LAYER_CLS]
      min_num_params: 0
    cpu_offload: False
    offload_params: False
  enable_gradient_checkpointing: True
  trust_remote_code: True
  attn_implementation: $MODEL_ATTN_IMPLEMENTATION
  lora_rank: $LORA_RANK
  lora_alpha: $LORA_ALPHA
  target_modules: all-linear
  use_liger: False

optim:
  lr: $LR
  betas: [0.9, 0.95]
  weight_decay: 0.01
  warmup_steps_ratio: 0.05
  clip_grad: 1.0
  lr_scheduler: cosine

trainer:
  default_local_dir: $CHECKPOINT_DIR
  default_hdfs_dir: null
  resume_path: null
  project_name: verifier-agent-sft
  experiment_name: $EXPERIMENT_NAME
  total_epochs: $TOTAL_EPOCHS
  total_training_steps: null
  logger: ['$LOGGER']
  seed: 17
  save_freq: $SAVE_FREQ
  test_freq: $TEST_FREQ
  nnodes: 1
  n_gpus_per_node: $N_GPUS
  max_ckpt_to_keep: 3

ulysses_sequence_parallel_size: 1
use_remove_padding: False
CONFIGEOF

echo "Verifier Agent SFT"
echo "Project:      $PROJECT_DIR"
echo "Model:        $MODEL_PATH"
echo "Train:        $TRAIN_PARQUET"
echo "Val:          $VAL_PARQUET"
echo "Output:       $OUTPUT_DIR"
echo "Config:       $CONFIG_DIR/$CONFIG_NAME.yaml"
echo "GPUs:         $N_GPUS"
echo "LoRA rank:    $LORA_RANK"
echo "Max length:   $MAX_LENGTH"
echo "Model dtype:   $MODEL_DTYPE"
echo "FSDP layer:    $TRANSFORMER_LAYER_CLS"
echo "Attention:     $MODEL_ATTN_IMPLEMENTATION"
echo "Save freq:     $SAVE_FREQ"
echo "Test freq:     $TEST_FREQ"
echo "Log file:      $LOG_DIR/${EXPERIMENT_NAME}.log"

cd "$PROJECT_DIR"

"$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="$N_GPUS" --master_port="$MASTER_PORT" \
  -m verl.trainer.fsdp_sft_trainer \
  --config-path="$CONFIG_DIR" \
  --config-name="$CONFIG_NAME" \
  > "$LOG_DIR/${EXPERIMENT_NAME}.log" 2>&1
