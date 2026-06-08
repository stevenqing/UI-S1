#!/bin/bash
#SBATCH --job-name=dagger
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/dagger_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/dagger_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# DAgger (Dataset Aggregation) Pipeline
#
# Key insight: Standard SFT trains on (GT_history, GT_action), but at test
# time the model sees its own imperfect history -> distribution mismatch.
#
# DAgger fixes this by:
#   Phase 1: Roll out model on training data with pred history
#   Phase 2: SFT on (pred_history, GT_action) pairs
#   Phase 3: Merge LoRA
#   Phase 4: Evaluate
#
# This directly addresses the #1 failure mode: error accumulation
# on long tasks (TSR drops from 72.3% at 1-step to 6.6% at 6+ steps).
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_SFT="$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_balanced_full_sft/checkpoint-272"
VERIFIER="$PROJECT_ROOT/v19_step_aware/output/verifier_hard_merged"
TRAIN_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_train_2000_balanced.jsonl"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"

DAGGER_DATA="$PROJECT_ROOT/v19_step_aware/data/dagger_rollouts.jsonl"
DAGGER_OUTPUT="$PROJECT_ROOT/v19_step_aware/output/dagger_sft"
DAGGER_MERGED="$PROJECT_ROOT/v19_step_aware/output/dagger_sft_merged"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p "$PROJECT_ROOT/v19_step_aware/scripts/logs"
mkdir -p "$PROJECT_ROOT/v19_step_aware/data"

export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_${SLURM_JOB_ID}
export HF_HOME=/tmp/hf_home_${SLURM_JOB_ID}
mkdir -p $HF_DATASETS_CACHE $HF_HOME

echo "=========================================="
echo "DAgger Pipeline"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

# ═══════════════════════════════════════════════
# Phase 1: Rollout base model on training data
# ═══════════════════════════════════════════════

if [ -f "$DAGGER_DATA" ]; then
    echo "[SKIP] Phase 1 — DAgger rollout data already exists"
    wc -l "$DAGGER_DATA"
else
    echo ""
    echo "=========================================="
    echo "Phase 1: Model Rollout (pred history)"
    echo "Start: $(date)"
    echo "=========================================="

    # Serve base SFT model for rollout (use all 4 GPUs for speed)
    python -m vllm.entrypoints.openai.api_server \
        --model "$BASE_SFT" \
        --served-model-name "base_sft" \
        --port 8000 \
        --tensor-parallel-size 4 \
        --max-model-len 32768 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --limit-mm-per-prompt '{"image": 2}' \
        &
    VLLM_PID=$!

    # Wait for server
    WAIT_COUNT=0
    while [ $WAIT_COUNT -lt 600 ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "vLLM ready after ${WAIT_COUNT}s"
            break
        fi
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
    done

    # Rollout on training data
    # 2000 episodes × 8.6 avg steps = ~17,200 API calls
    # With 64 threads, ~15-20 min
    python "$PROJECT_ROOT/v19_step_aware/prepare_dagger_data.py" \
        --train_data "$TRAIN_DATA" \
        --api_url http://localhost:8000/v1 \
        --model_name base_sft \
        --output "$DAGGER_DATA" \
        --threads 64

    # Kill vLLM server
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true

    echo "Phase 1 done: $(date)"
fi

# ═══════════════════════════════════════════════
# Phase 2: DAgger SFT Training
# ═══════════════════════════════════════════════

if [ -d "$DAGGER_OUTPUT/checkpoint-epoch2" ]; then
    echo "[SKIP] Phase 2 — DAgger SFT checkpoint already exists"
else
    echo ""
    echo "=========================================="
    echo "Phase 2: DAgger SFT Training"
    echo "Start: $(date)"
    echo "=========================================="

    torchrun --nproc_per_node=4 \
        "$PROJECT_ROOT/v19_step_aware/train_dagger_sft.py" \
        --model_path "$BASE_SFT" \
        --dagger_data "$DAGGER_DATA" \
        --output_dir "$DAGGER_OUTPUT" \
        --data_mode all \
        --include_gt_history \
        --lora_rank 64 \
        --lr 2e-6 \
        --epochs 2 \
        --grad_accum 16 \
        --max_grad_norm 1.0 \
        --gradient_checkpointing

    echo "Phase 2 done: $(date)"
fi

# ═══════════════════════════════════════════════
# Phase 3: Merge LoRA
# ═══════════════════════════════════════════════

if [ -d "$DAGGER_MERGED" ]; then
    echo "[SKIP] Phase 3 — Merged model already exists"
else
    echo ""
    echo "=========================================="
    echo "Phase 3: Merge LoRA"
    echo "Start: $(date)"
    echo "=========================================="

    # Use epoch 1 checkpoint (conservative, less overfitting risk)
    CKPT="$DAGGER_OUTPUT/checkpoint-epoch1"

    python -c "
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch

print('Loading base model...')
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    '$BASE_SFT', torch_dtype=torch.bfloat16, trust_remote_code=True)

print('Loading LoRA from $CKPT...')
model = PeftModel.from_pretrained(base, '$CKPT')

print('Merging...')
merged = model.merge_and_unload()

print('Saving to $DAGGER_MERGED...')
merged.save_pretrained('$DAGGER_MERGED')
processor = AutoProcessor.from_pretrained('$BASE_SFT', trust_remote_code=True)
processor.save_pretrained('$DAGGER_MERGED')
print('Done!')
"

    echo "Phase 3 done: $(date)"
fi

# ═══════════════════════════════════════════════
# Phase 4: Evaluation
# ═══════════════════════════════════════════════

echo ""
echo "=========================================="
echo "Phase 4: Evaluation"
echo "Start: $(date)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model "$DAGGER_MERGED" \
    --served-model-name "dagger_sft" \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image": 2}' \
    &
PRED_PID=$!

CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
    --model "$VERIFIER" \
    --served-model-name "verifier" \
    --port 8001 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image": 3}' \
    &
VERIFY_PID=$!

for PORT in 8000 8001; do
    WAIT_COUNT=0
    while [ $WAIT_COUNT -lt 600 ]; do
        if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
            echo "Port ${PORT} ready after ${WAIT_COUNT}s"
            break
        fi
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
    done
done

# Exp 1: Greedy eval (pred history)
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/dagger_greedy_pred"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] dagger_greedy_pred"
else
    echo "=== Greedy eval (pred history) ==="
    mkdir -p "$OUTPUT"
    python "$PROJECT_ROOT/v19_step_aware/eval_step_aware.py" \
        --prompt_variant type_focused \
        --test_data "$TEST_DATA" \
        --api_url http://localhost:8000/v1 \
        --model_name dagger_sft \
        --output_dir "$OUTPUT" \
        --threads 64
fi

# Exp 2: Best-of-5 + verifier (pred history)
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/dagger_bon5_pred"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] dagger_bon5_pred"
else
    echo "=== Best-of-5 + verifier (pred history) ==="
    mkdir -p "$OUTPUT"
    python "$PROJECT_ROOT/v19_step_aware/eval_best_of_n.py" \
        --test_data "$TEST_DATA" \
        --predict_url http://localhost:8000/v1 \
        --predict_model dagger_sft \
        --verify_url http://localhost:8001/v1 \
        --verify_model verifier \
        --output_dir "$OUTPUT" \
        --n_candidates 5 \
        --sample_temperature 0.7 \
        --selection_strategy logprob_rank \
        --threads 24
fi

# Cleanup
kill $PRED_PID $VERIFY_PID 2>/dev/null || true
wait $PRED_PID $VERIFY_PID 2>/dev/null || true

# ═══════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════
echo ""
echo "=========================================="
echo "DAGGER SFT RESULTS"
echo "=========================================="
echo "Baseline: TSR=23.3% (base_sft greedy), 26.6% (base best-of-5 logprob)"
echo "Step-level RL: All failed (GRPO, RFT, AS-GRPO -> no improvement)"
echo ""

for exp in dagger_greedy_pred dagger_bon5_pred; do
    S=$(ls "$PROJECT_ROOT/v19_step_aware/outputs/${exp}"/eval_summary_*.json 2>/dev/null | tail -1)
    if [ -n "$S" ]; then
        echo "--- ${exp} ---"
        python3 -c "
import json
with open('$S') as f:
    d = json.load(f)
print(f'  TSR={d[\"tsr\"]*100:.1f}%  StepSR={d[\"step_sr\"]*100:.1f}%  Progress={d[\"avg_progress\"]*100:.1f}%')
tlm = d.get('task_length_metrics', {})
for b in ['1','2-3','4-5','6+']:
    if b in tlm:
        m = tlm[b]
        print(f'  {b:>5s} steps: TSR={m[\"tsr\"]*100:5.1f}%  StepSR={m[\"step_sr\"]*100:5.1f}%  (n={m[\"num_episodes\"]})')
sm = d.get('selection_metrics')
if sm:
    print(f'  Greedy acc: {sm[\"greedy_accuracy\"]*100:.1f}%  Oracle: {sm[\"oracle_accuracy\"]*100:.1f}%  Selected: {sm[\"selected_accuracy\"]*100:.1f}%')
    print(f'  Net gain: {sm[\"net_gain\"]} (helped={sm[\"retry_helped\"]}, hurt={sm[\"retry_hurt\"]})')
"
        echo ""
    fi
done

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
