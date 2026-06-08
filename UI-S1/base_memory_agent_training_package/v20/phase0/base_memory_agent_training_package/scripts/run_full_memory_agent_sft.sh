#!/usr/bin/env bash
set -eo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"

ROOT_DIR="${ROOT_DIR:-/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1}"
PACKAGE_DIR="${PACKAGE_DIR:-$ROOT_DIR/v20/phase0/base_memory_agent_training_package}"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-$ROOT_DIR/train_GUI_360/LLaMA-Factory}"
RUN_KEY="${RUN_KEY:-type_recovery}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29551}"
CONFIG_FILE="$PACKAGE_DIR/configs/qwen25vl_full_proposal_${RUN_KEY}.yaml"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing config: $CONFIG_FILE" >&2
  exit 2
fi

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "llamafactory-cli is not on PATH. Activate the LLaMA-Factory training environment first." >&2
  exit 127
fi

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:$LLAMA_FACTORY_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets_cache_full_proposal_${USER:-user}}"
export HF_HOME="${HF_HOME:-/tmp/hf_home_full_proposal_${USER:-user}}"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME"

echo "[$(date --iso-8601=seconds)] full memory-agent SFT start run_key=$RUN_KEY config=$CONFIG_FILE nodes=$NNODES nproc=$NPROC_PER_NODE"
FORCE_TORCHRUN=1 \
NNODES="$NNODES" \
NODE_RANK="$NODE_RANK" \
MASTER_ADDR="$MASTER_ADDR" \
MASTER_PORT="$MASTER_PORT" \
NPROC_PER_NODE="$NPROC_PER_NODE" \
llamafactory-cli train "$CONFIG_FILE"
echo "[$(date --iso-8601=seconds)] full memory-agent SFT done run_key=$RUN_KEY"
