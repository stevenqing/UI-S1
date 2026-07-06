#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source .venv-qwen3-vllm/bin/activate
export PYTHONNOUSERSITE=1
export PATH="${ROOT}/.venv-qwen3-vllm/bin:${PATH}"

OUT="${OUT:-outputs/critstep_verifier_v2/strict}"
LOG_DIR="${OUT}/logs"
mkdir -p "${LOG_DIR}"
MAIN_LOG="${LOG_DIR}/overnight.log"
exec > >(tee -a "${MAIN_LOG}") 2>&1

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] strict verifier overnight start"
echo "OUT=${OUT}"

run_shards() {
  local label="$1"
  local script="$2"
  local output_dir="$3"
  shift 3
  mkdir -p "${output_dir}/logs"
  local pids=()
  for i in 0 1 2 3 4 5 6 7; do
    local log="${output_dir}/logs/${label}_shard_${i}.log"
    echo "[$(date -u '+%H:%M:%S')] start ${label} shard ${i}: ${log}"
    CUDA_VISIBLE_DEVICES=${i} python "${script}" "$@" --output-dir "${output_dir}" --num-shards 8 --shard-index "${i}" --device cuda:0 --resume > "${log}" 2>&1 &
    pids+=("$!")
  done
  local status=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  if [[ "${status}" != 0 ]]; then
    echo "${label} failed; log tails follow" >&2
    for i in 0 1 2 3 4 5 6 7; do
      echo "--- ${label} shard ${i} ---" >&2
      tail -80 "${output_dir}/logs/${label}_shard_${i}.log" >&2 || true
    done
    exit 1
  fi
}

if [[ ! -f "${OUT}/pools/train_per_step.jsonl" || ! -f "${OUT}/pools/test_per_step.jsonl" ]]; then
  echo "[$(date -u '+%H:%M:%S')] building strict pools"
  python scripts/build_strict_verifier_pools.py --output-dir "${OUT}/pools"
else
  echo "[$(date -u '+%H:%M:%S')] strict pools already exist"
fi

if [[ ! -f "${OUT}/stage1_train/stage1_summary.json" ]]; then
  echo "[$(date -u '+%H:%M:%S')] scoring Stage1 TRAIN"
  run_shards stage1_train scripts/score_critstep_verifier_v2_cot_voting.py "${OUT}/stage1_train" \
    --per-step "${OUT}/pools/train_per_step.jsonl" --vote-ks 8 --score-mode verdict_vote --batch-size 1 --vote-chunk 8 --max-new-tokens 160
  python scripts/score_critstep_verifier_v2_cot_voting.py --per-step "${OUT}/pools/train_per_step.jsonl" --output-dir "${OUT}/stage1_train" --vote-ks 8 --score-mode verdict_vote --merge-shards --num-shards 8
else
  echo "[$(date -u '+%H:%M:%S')] Stage1 TRAIN already scored"
fi

if [[ ! -f "${OUT}/stage1_test/stage1_summary.json" ]]; then
  echo "[$(date -u '+%H:%M:%S')] scoring Stage1 TEST"
  run_shards stage1_test scripts/score_critstep_verifier_v2_cot_voting.py "${OUT}/stage1_test" \
    --per-step "${OUT}/pools/test_per_step.jsonl" --vote-ks 8 --score-mode verdict_vote --batch-size 1 --vote-chunk 8 --max-new-tokens 160
  python scripts/score_critstep_verifier_v2_cot_voting.py --per-step "${OUT}/pools/test_per_step.jsonl" --output-dir "${OUT}/stage1_test" --vote-ks 8 --score-mode verdict_vote --merge-shards --num-shards 8
else
  echo "[$(date -u '+%H:%M:%S')] Stage1 TEST already scored"
fi

if [[ ! -f "${OUT}/stage2_train/stage2_summary.json" ]]; then
  echo "[$(date -u '+%H:%M:%S')] scoring Stage2 TRAIN"
  run_shards stage2_train scripts/score_critstep_verifier_stage2_comparative.py "${OUT}/stage2_train" \
    --per-step "${OUT}/stage1_train/stage1_per_step.jsonl" --stage1-summary "${OUT}/stage1_train/stage1_summary.json" --max-new-tokens 96
  python scripts/score_critstep_verifier_stage2_comparative.py --per-step "${OUT}/stage1_train/stage1_per_step.jsonl" --stage1-summary "${OUT}/stage1_train/stage1_summary.json" --output-dir "${OUT}/stage2_train" --merge-shards --num-shards 8
else
  echo "[$(date -u '+%H:%M:%S')] Stage2 TRAIN already scored"
fi

if [[ ! -f "${OUT}/stage2_test/stage2_summary.json" ]]; then
  echo "[$(date -u '+%H:%M:%S')] scoring Stage2 TEST"
  run_shards stage2_test scripts/score_critstep_verifier_stage2_comparative.py "${OUT}/stage2_test" \
    --per-step "${OUT}/stage1_test/stage1_per_step.jsonl" --stage1-summary "${OUT}/stage1_test/stage1_summary.json" --max-new-tokens 96
  python scripts/score_critstep_verifier_stage2_comparative.py --per-step "${OUT}/stage1_test/stage1_per_step.jsonl" --stage1-summary "${OUT}/stage1_test/stage1_summary.json" --output-dir "${OUT}/stage2_test" --merge-shards --num-shards 8
else
  echo "[$(date -u '+%H:%M:%S')] Stage2 TEST already scored"
fi

echo "[$(date -u '+%H:%M:%S')] fitting strict train/test combination"
python scripts/combine_critstep_verifier_v2_strict.py \
  --train-stage1-per-step "${OUT}/stage1_train/stage1_per_step.jsonl" \
  --train-stage2-per-step "${OUT}/stage2_train/stage2_per_step.jsonl" \
  --test-stage1-per-step "${OUT}/stage1_test/stage1_per_step.jsonl" \
  --test-stage2-per-step "${OUT}/stage2_test/stage2_per_step.jsonl" \
  --output-dir "${OUT}/combine"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] strict verifier overnight complete"