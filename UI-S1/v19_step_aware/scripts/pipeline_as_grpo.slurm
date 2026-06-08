#!/bin/bash
#SBATCH --job-name=as_grpo
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/as_grpo_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/as_grpo_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Action-Space GRPO (AS-GRPO)
#
# Key insight: Standard GRPO fails because token-space sampling
# doesn't create action-space diversity. AS-GRPO perturbs actions
# in the ACTION PARAMETER SPACE:
#   - Click coordinates: Gaussian perturbation (σ=40px)
#   - Reward: Gaussian distance decay (σ=0.05 normalized)
#   - Loss: action tokens only (--action_loss_only)
#
# Phase 1: Create AS-GRPO data (no vLLM needed, fast)
# Phase 2: Train with GRPO + action loss masking
# Phase 3: Merge LoRA
# Phase 4: Evaluate
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_SFT="$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_balanced_full_sft/checkpoint-272"
VERIFIER="$PROJECT_ROOT/v19_step_aware/output/verifier_hard_merged"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"

# Input: existing candidates from R1; Output: action-space perturbed data
INPUT_DATA="$PROJECT_ROOT/v19_step_aware/data/grpo_candidates_n5.jsonl"
AS_GRPO_DATA="$PROJECT_ROOT/v19_step_aware/data/as_grpo_candidates_n8.jsonl"
AS_GRPO_OUTPUT="$PROJECT_ROOT/v19_step_aware/output/as_grpo"
AS_GRPO_MERGED="$PROJECT_ROOT/v19_step_aware/output/as_grpo_merged"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p "$PROJECT_ROOT/v19_step_aware/scripts/logs"
mkdir -p "$PROJECT_ROOT/v19_step_aware/data"

export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_${SLURM_JOB_ID}
export HF_HOME=/tmp/hf_home_${SLURM_JOB_ID}
mkdir -p $HF_DATASETS_CACHE $HF_HOME

echo "=========================================="
echo "Action-Space GRPO (AS-GRPO)"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

# ═══════════════════════════════════════════════
# Phase 1: Create AS-GRPO data (CPU only, fast)
# ═══════════════════════════════════════════════

if [ -f "$AS_GRPO_DATA" ]; then
    echo "[SKIP] Phase 1 — AS-GRPO data already exists"
    wc -l "$AS_GRPO_DATA"
else
    echo ""
    echo "=========================================="
    echo "Phase 1: Action-Space Perturbation"
    echo "Start: $(date)"
    echo "=========================================="

    python "$PROJECT_ROOT/v19_step_aware/prepare_as_grpo_data.py" \
        --input "$INPUT_DATA" \
        --output "$AS_GRPO_DATA" \
        --n_candidates 8 \
        --coord_sigma 40 \
        --reward_sigma 0.05 \
        --seed 42

    echo "Phase 1 done: $(date)"
fi

# ═══════════════════════════════════════════════
# Phase 2: AS-GRPO Training
# ═══════════════════════════════════════════════

if [ -d "$AS_GRPO_OUTPUT/checkpoint-epoch2" ]; then
    echo "[SKIP] Phase 2 — AS-GRPO checkpoint already exists"
else
    echo ""
    echo "=========================================="
    echo "Phase 2: AS-GRPO Training"
    echo "Start: $(date)"
    echo "=========================================="

    torchrun --nproc_per_node=4 \
        "$PROJECT_ROOT/v19_step_aware/train_grpo.py" \
        --model_path "$BASE_SFT" \
        --grpo_data "$AS_GRPO_DATA" \
        --output_dir "$AS_GRPO_OUTPUT" \
        --type_filtered \
        --lora_rank 64 \
        --lr 5e-7 \
        --epochs 1 \
        --beta_kl 0.0 \
        --grad_accum 16 \
        --max_grad_norm 0.5 \
        --gradient_checkpointing

    echo "Phase 2 done: $(date)"
fi

# ═══════════════════════════════════════════════
# Phase 3: Merge LoRA
# ═══════════════════════════════════════════════

if [ -d "$AS_GRPO_MERGED" ]; then
    echo "[SKIP] Phase 3 — Merged model already exists"
else
    echo ""
    echo "=========================================="
    echo "Phase 3: Merge LoRA"
    echo "Start: $(date)"
    echo "=========================================="

    CKPT="$AS_GRPO_OUTPUT/checkpoint-epoch1"

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

print('Saving to $AS_GRPO_MERGED...')
merged.save_pretrained('$AS_GRPO_MERGED')
processor = AutoProcessor.from_pretrained('$BASE_SFT', trust_remote_code=True)
processor.save_pretrained('$AS_GRPO_MERGED')
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
    --model "$AS_GRPO_MERGED" \
    --served-model-name "as_grpo" \
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
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/as_grpo_greedy_pred"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] as_grpo_greedy_pred"
else
    echo "=== Greedy eval (pred history) ==="
    mkdir -p "$OUTPUT"
    python "$PROJECT_ROOT/v19_step_aware/eval_step_aware.py" \
        --prompt_variant type_focused \
        --test_data "$TEST_DATA" \
        --api_url http://localhost:8000/v1 \
        --model_name as_grpo \
        --output_dir "$OUTPUT" \
        --threads 64
fi

# Exp 2: Best-of-5 + verifier (pred history)
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/as_grpo_bon5_pred"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] as_grpo_bon5_pred"
else
    echo "=== Best-of-5 + verifier (pred history) ==="
    mkdir -p "$OUTPUT"
    python "$PROJECT_ROOT/v19_step_aware/eval_best_of_n.py" \
        --test_data "$TEST_DATA" \
        --predict_url http://localhost:8000/v1 \
        --predict_model as_grpo \
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
echo "ACTION-SPACE GRPO RESULTS"
echo "=========================================="
echo "Baseline: TSR=23.3% (base_sft greedy), 26.6% (base best-of-5 logprob)"
echo "Token-space GRPO: R1=10.9% (collapse), R2=23.6% (no signal), R3=TBD"
echo "RFT: 23.5%"
echo ""

for exp in as_grpo_greedy_pred as_grpo_bon5_pred; do
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
