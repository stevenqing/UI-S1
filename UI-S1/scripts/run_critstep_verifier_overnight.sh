#!/usr/bin/env bash
# Overnight full-action verifier LoRA training.
#
# This script is intentionally resumable and data-source explicit:
# - downloads/syncs the portable verifier data package from Hugging Face
# - creates a local training-view checkpoint config for Qwen2.5-VL rope compatibility
# - launches LLaMA-Factory LoRA SFT with venv torchrun first on PATH

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$PROJECT_DIR"

PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv-qwen3-vllm/bin/python}
HF_BIN=${HF_BIN:-$PROJECT_DIR/.venv-qwen3-vllm/bin/hf}
LLAMAFACTORY_CLI=${LLAMAFACTORY_CLI:-$PROJECT_DIR/.venv-qwen3-vllm/bin/llamafactory-cli}

DATA_REPO_ID=${DATA_REPO_ID:-Stevenshuqing/gui360-critstep-full-action-verifier-data}
DATA_DIR=${DATA_DIR:-$PROJECT_DIR/outputs/critstep_verifier_hf_dataset}
BASE_MODEL=${BASE_MODEL:-$PROJECT_DIR/checkpoints/gui360-fullparam-sft-step250}
TRAINVIEW_DIR=${TRAINVIEW_DIR:-$PROJECT_DIR/outputs/critstep_verifier/gui360_fullparam_sft_step250_trainview}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/outputs/critstep_verifier/verifier_lora_qwen25vl_overnight}
CONFIG_PATH=${CONFIG_PATH:-$PROJECT_DIR/outputs/critstep_verifier/train_full_action_verifier_lora_overnight.yaml}
LOG_DIR=${LOG_DIR:-$PROJECT_DIR/outputs/critstep_verifier/logs}
LOG_FILE=${LOG_FILE:-$LOG_DIR/train_full_action_verifier_lora_overnight.log}

N_GPUS=${N_GPUS:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29643}

LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
LR=${LR:-5.0e-5}
EPOCHS=${EPOCHS:-1.0}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-2}
GRAD_ACCUM=${GRAD_ACCUM:-4}
SAVE_STEPS=${SAVE_STEPS:-100}
EVAL_STEPS=${EVAL_STEPS:-100}
CUTOFF_LEN=${CUTOFF_LEN:-4096}

mkdir -p "$DATA_DIR" "$TRAINVIEW_DIR" "$OUTPUT_DIR" "$LOG_DIR" "$(dirname "$CONFIG_PATH")"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing PYTHON_BIN=$PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -x "$HF_BIN" ]]; then
  echo "Missing HF_BIN=$HF_BIN" >&2
  exit 1
fi
if [[ ! -x "$LLAMAFACTORY_CLI" ]]; then
  echo "Missing LLAMAFACTORY_CLI=$LLAMAFACTORY_CLI" >&2
  exit 1
fi
if [[ ! -d "$BASE_MODEL" ]]; then
  echo "Missing BASE_MODEL=$BASE_MODEL" >&2
  exit 1
fi

echo "[$(date)] Syncing verifier data from HF: $DATA_REPO_ID -> $DATA_DIR"
"$HF_BIN" download "$DATA_REPO_ID" --repo-type dataset --local-dir "$DATA_DIR"

echo "[$(date)] Creating patched training-view checkpoint: $TRAINVIEW_DIR"
rm -rf "$TRAINVIEW_DIR"
mkdir -p "$TRAINVIEW_DIR"
shopt -s nullglob
for item in "$BASE_MODEL"/* "$BASE_MODEL"/.[!.]*; do
  name=$(basename "$item")
  if [[ "$name" == "config.json" || "$name" == ".cache" ]]; then
    continue
  fi
  [[ -e "$TRAINVIEW_DIR/$name" || -L "$TRAINVIEW_DIR/$name" ]] && continue
  ln -s "$(realpath "$item")" "$TRAINVIEW_DIR/$name"
done
shopt -u nullglob
cp "$BASE_MODEL/config.json" "$TRAINVIEW_DIR/config.json"
"$PYTHON_BIN" - <<PY
import json
from pathlib import Path
path = Path("$TRAINVIEW_DIR/config.json")
data = json.loads(path.read_text())
for container in (data, data.get("text_config") or {}):
    rope = container.get("rope_scaling")
    if isinstance(rope, dict) and rope.get("rope_type") == "mrope":
        rope["rope_type"] = "default"
path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
print("patched rope_scaling", data.get("rope_scaling"), (data.get("text_config") or {}).get("rope_scaling"))
PY

cat > "$CONFIG_PATH" <<YAML
### model
model_name_or_path: $TRAINVIEW_DIR
image_max_pixels: 1003520
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: $LORA_RANK
lora_alpha: $LORA_ALPHA
lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
freeze_vision_tower: true
freeze_multi_modal_projector: true

### dataset
dataset: critstep_full_action_verifier_train
eval_dataset: critstep_full_action_verifier_val
dataset_dir: $DATA_DIR
template: qwen2_vl
cutoff_len: $CUTOFF_LEN
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: $OUTPUT_DIR
logging_steps: 10
save_steps: $SAVE_STEPS
save_total_limit: 3
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none
run_name: critstep_full_action_verifier_lora_overnight

### train
per_device_train_batch_size: $PER_DEVICE_BATCH
gradient_accumulation_steps: $GRAD_ACCUM
learning_rate: $LR
num_train_epochs: $EPOCHS
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.01
bf16: true
ddp_timeout: 180000000

### eval
per_device_eval_batch_size: $PER_DEVICE_BATCH
eval_strategy: steps
eval_steps: $EVAL_STEPS
YAML

echo "[$(date)] Overnight verifier training"
echo "Data:      $DATA_DIR"
echo "Model:     $TRAINVIEW_DIR"
echo "Output:    $OUTPUT_DIR"
echo "Config:    $CONFIG_PATH"
echo "Log:       $LOG_FILE"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"

export PATH="$PROJECT_DIR/.venv-qwen3-vllm/bin:$PATH"
export CUDA_VISIBLE_DEVICES
export FORCE_TORCHRUN=1
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE="$N_GPUS"
export MASTER_ADDR
export MASTER_PORT

"$LLAMAFACTORY_CLI" train "$CONFIG_PATH" 2>&1 | tee "$LOG_FILE"

echo "[$(date)] Training complete: $OUTPUT_DIR"
