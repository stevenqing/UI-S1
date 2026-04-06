#!/bin/bash
#SBATCH --job-name=mv_g360_E
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/multiview_gui360_E_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/multiview_gui360_E_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

set -e
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
MODEL_PATH="${PROJECT_DIR}/checkpoints/Qwen2.5-VL-7B-Instruct"
VLLM_PORT=8000

echo "=== GUI-360 Variant E | Job ${SLURM_JOB_ID} | $(hostname) | $(date) ==="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:${PATH}"
export CONDA_DEFAULT_ENV=qwen3-eval
export TMPDIR=/tmp/${SLURM_JOB_ID}
mkdir -p ${TMPDIR}
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR}/torchinductor
export TORCH_HOME=${TMPDIR}/torch
export XDG_CACHE_HOME=${TMPDIR}/cache
mkdir -p ${TORCHINDUCTOR_CACHE_DIR} ${TORCH_HOME} ${XDG_CACHE_HOME}
mkdir -p "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/results/multiview_gui360_variants_20260326_073551"

pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" --port ${VLLM_PORT} \
    --tensor-parallel-size 4 --max-model-len 32768 \
    --trust-remote-code --gpu-memory-utilization 0.9 \
    2>&1 | tee "${PROJECT_DIR}/scripts/eval/logs/vllm_g360_E_${SLURM_JOB_ID}.log" &
VLLM_PID=$!

MAX_WAIT=600; WAIT_COUNT=0
while [ ${WAIT_COUNT} -lt ${MAX_WAIT} ]; do
    curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1 && echo "vLLM ready!" && break
    sleep 5; WAIT_COUNT=$((WAIT_COUNT + 5))
done
[ ${WAIT_COUNT} -ge ${MAX_WAIT} ] && echo "ERROR: vLLM failed" && kill ${VLLM_PID} 2>/dev/null && exit 1

cd "${PROJECT_DIR}/evaluation"
python eval_multiview_gui360.py \
    --parquet_file "${PROJECT_DIR}/train_GUI_360/data/gui360_test_sft_eval_format_with_bbox.parquet" \
    --output_dir "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/results/multiview_gui360_variants_20260326_073551" \
    --model_name "${MODEL_PATH}" \
    --mode multiview \
    --variant E \
    --api_url "http://localhost:${VLLM_PORT}/v1" \
    --max_workers 256 \
    --save_debug

kill ${VLLM_PID} 2>/dev/null || true
echo "=== Variant E Complete | $(date) ==="
