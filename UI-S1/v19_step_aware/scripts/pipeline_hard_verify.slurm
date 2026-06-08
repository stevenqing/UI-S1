#!/bin/bash
#SBATCH --job-name=hard_verify
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/hard_verify_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/hard_verify_%j.err
#SBATCH --time=16:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Full Pipeline: Hard Negative Verifier
#
# Phase 1: Run base_sft predictions on TRAIN data
# Phase 2: Build hard negative verification data
# Phase 3: Retrain verifier LoRA with hard negatives
# Phase 4: Merge LoRA
# Phase 5: Eval with dual model (base predict + verifier verify)
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_SFT="$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_balanced_full_sft/checkpoint-272"
TRAIN_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_train_2000_balanced.jsonl"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p "$PROJECT_ROOT/v19_step_aware/scripts/logs"

# Node-local HF cache
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_${SLURM_JOB_ID}
export HF_HOME=/tmp/hf_home_${SLURM_JOB_ID}
mkdir -p $HF_DATASETS_CACHE $HF_HOME

echo "=========================================="
echo "Hard Negative Verifier Pipeline"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

# ═════════════════════════════════════════════
# Phase 1: Predict on training data
# ═════════════════════════════════════════════
TRAIN_PRED_DIR="$PROJECT_ROOT/v19_step_aware/outputs/train_predictions"
if ls "$TRAIN_PRED_DIR"/eval_results_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] Phase 1 — train predictions already exist"
else
    echo ""
    echo "=== Phase 1: Predict on training data ==="
    echo "Start: $(date)"

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

    WAIT_COUNT=0
    while [ $WAIT_COUNT -lt 600 ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "vLLM ready after ${WAIT_COUNT}s"
            break
        fi
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
    done
    if [ $WAIT_COUNT -ge 600 ]; then
        echo "ERROR: vLLM failed to start"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi

    mkdir -p "$TRAIN_PRED_DIR"
    python "$PROJECT_ROOT/v19_step_aware/eval_step_aware.py" \
        --prompt_variant type_focused \
        --test_data "$TRAIN_DATA" \
        --api_url http://localhost:8000/v1 \
        --model_name base_sft \
        --output_dir "$TRAIN_PRED_DIR" \
        --threads 128 \
        --gt_history

    kill $VLLM_PID 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    sleep 5

    echo "Phase 1 done: $(date)"
fi

# ═════════════════════════════════════════════
# Phase 2: Build hard negative data
# ═════════════════════════════════════════════
HARD_DATA="$PROJECT_ROOT/v19_step_aware/data/verify_hard.jsonl"
if [ -f "$HARD_DATA" ]; then
    echo "[SKIP] Phase 2 — hard negative data already exists"
else
    echo ""
    echo "=== Phase 2: Build hard negative verification data ==="
    echo "Start: $(date)"

    EVAL_RESULTS=$(ls "$TRAIN_PRED_DIR"/eval_results_*.json | tail -1)
    python "$PROJECT_ROOT/v19_step_aware/prepare_verify_hard_negatives.py" \
        --eval_results "$EVAL_RESULTS" \
        --train_data "$TRAIN_DATA" \
        --output "$HARD_DATA" \
        --synthetic_ratio 0.3

    echo "Phase 2 done: $(date)"
fi

# ═════════════════════════════════════════════
# Phase 3: Register dataset + retrain verifier
# ═════════════════════════════════════════════
VERIFIER_V2="$PROJECT_ROOT/v19_step_aware/output/verifier_hard_lora_r128"
if [ -f "$VERIFIER_V2/adapter_config.json" ]; then
    echo "[SKIP] Phase 3 — verifier v2 already trained"
else
    echo ""
    echo "=== Phase 3: Train verifier with hard negatives ==="
    echo "Start: $(date)"

    # Add hard negative datasets to dataset_info.json if not already there
    python3 -c "
import json
info_path = 'train_GUI_360/llamafactory/data/dataset_info.json'
with open(info_path) as f:
    info = json.load(f)
entry = {
    'formatting': 'sharegpt',
    'columns': {'messages': 'conversations', 'images': 'images'},
    'tags': {'role_tag': 'from', 'content_tag': 'value', 'user_tag': 'human', 'assistant_tag': 'gpt'}
}
changed = False
if 'verify_hard_train' not in info:
    info['verify_hard_train'] = {**entry, 'file_name': 'verify_hard_train.json'}
    changed = True
if 'verify_hard_val' not in info:
    info['verify_hard_val'] = {**entry, 'file_name': 'verify_hard_val.json'}
    changed = True
if changed:
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print('Updated dataset_info.json')
else:
    print('dataset_info.json already up to date')
"

    # Create training config
    cat > "$PROJECT_ROOT/v19_step_aware/train_verifier_hard_lora.yaml" << 'YAML'
### model — start from base SFT checkpoint
model_name_or_path: /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/llamafactory/output/gui360_balanced_full_sft/checkpoint-272
image_max_pixels: 1003520
video_max_pixels: 16384
trust_remote_code: true

### method — LoRA SFT for verification
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 128
lora_alpha: 256
lora_target: all
freeze_vision_tower: true
freeze_multi_modal_projector: true

### dataset — hard negative verification data
dataset: verify_hard_train
dataset_dir: train_GUI_360/llamafactory/data
template: qwen2_vl
cutoff_len: 4096
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: v19_step_aware/output/verifier_hard_lora_r128
logging_steps: 10
save_steps: 200
save_total_limit: 3
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none
run_name: verifier_hard_lora_r128

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 5.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.01
bf16: true
ddp_timeout: 180000000

### eval
eval_dataset: verify_hard_val
per_device_eval_batch_size: 4
eval_strategy: steps
eval_steps: 200
YAML

    # Train
    export MASTER_ADDR=localhost
    export MASTER_PORT=29500
    export NCCL_DEBUG=WARN
    export NCCL_TIMEOUT=10800

    FORCE_TORCHRUN=1 \
    NNODES=1 \
    NODE_RANK=0 \
    MASTER_ADDR=$MASTER_ADDR \
    MASTER_PORT=$MASTER_PORT \
    NPROC_PER_NODE=4 \
    llamafactory-cli train "$PROJECT_ROOT/v19_step_aware/train_verifier_hard_lora.yaml"

    echo "Phase 3 done: $(date)"
fi

# ═════════════════════════════════════════════
# Phase 4: Merge LoRA
# ═════════════════════════════════════════════
MERGED_V2="$PROJECT_ROOT/v19_step_aware/output/verifier_hard_merged"
if [ -f "$MERGED_V2/config.json" ]; then
    echo "[SKIP] Phase 4 — merged model already exists"
else
    echo ""
    echo "=== Phase 4: Merge LoRA ==="
    echo "Start: $(date)"

    llamafactory-cli export \
        --model_name_or_path "$BASE_SFT" \
        --adapter_name_or_path "$VERIFIER_V2" \
        --template qwen2_vl \
        --finetuning_type lora \
        --export_dir "$MERGED_V2" \
        --export_size 5 \
        --export_legacy_format false \
        --trust_remote_code true

    echo "Phase 4 done: $(date)"
fi

# ═════════════════════════════════════════════
# Phase 5: Eval with dual model
# base_sft on port 8000 (predict)
# verifier on port 8001 (verify)
# ═════════════════════════════════════════════
echo ""
echo "=== Phase 5: Dual-model eval ==="
echo "Start: $(date)"

# Serve base_sft on port 8000 (TP=2, 2 GPUs)
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_SFT" \
    --served-model-name "base_sft" \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image": 2}' \
    &
PRED_PID=$!

# Serve verifier on port 8001 (TP=2, 2 GPUs)
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
    --model "$MERGED_V2" \
    --served-model-name "verifier" \
    --port 8001 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image": 3}' \
    &
VERIFY_PID=$!

# Wait for both servers
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
    if [ $WAIT_COUNT -ge 600 ]; then
        echo "ERROR: Server on port ${PORT} failed to start"
        kill $PRED_PID $VERIFY_PID 2>/dev/null || true
        exit 1
    fi
done

# Exp 1: Dual model, pred history
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/dual_hard_verify_pred"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] dual_hard_verify_pred — already done"
else
    mkdir -p "$OUTPUT"
    python "$PROJECT_ROOT/v19_step_aware/eval_dual_verify.py" \
        --test_data "$TEST_DATA" \
        --predict_url http://localhost:8000/v1 \
        --predict_model base_sft \
        --verify_url http://localhost:8001/v1 \
        --verify_model verifier \
        --output_dir "$OUTPUT" \
        --threads 48
fi

# Exp 2: Dual model, GT history
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/dual_hard_verify_gt"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] dual_hard_verify_gt — already done"
else
    mkdir -p "$OUTPUT"
    python "$PROJECT_ROOT/v19_step_aware/eval_dual_verify.py" \
        --test_data "$TEST_DATA" \
        --predict_url http://localhost:8000/v1 \
        --predict_model base_sft \
        --verify_url http://localhost:8001/v1 \
        --verify_model verifier \
        --output_dir "$OUTPUT" \
        --threads 48 \
        --gt_history
fi

# Cleanup
kill $PRED_PID $VERIFY_PID 2>/dev/null || true
wait $PRED_PID $VERIFY_PID 2>/dev/null || true

# ═════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════
echo ""
echo "=========================================="
echo "DUAL MODEL + HARD NEGATIVE VERIFIER RESULTS"
echo "=========================================="
echo "Baseline: TPR=84%/FPR=57% (untrained) | TPR=53%/FPR=9% (synthetic-trained)"
echo ""
for exp in dual_hard_verify_pred dual_hard_verify_gt; do
    OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/${exp}"
    S=$(ls "$OUTPUT"/eval_summary_*.json 2>/dev/null | tail -1)
    if [ -n "$S" ]; then
        echo "--- ${exp} ---"
        python3 -c "
import json
with open('$S') as f:
    d = json.load(f)
vm = d['verification_metrics']
sim = d['simulated_self_correction']
print(f\"  TSR={d['tsr']*100:.1f}%  StepSR={d['step_sr']*100:.1f}%\")
print(f\"  Detection Rate (TPR): {vm['detection_rate_TPR']*100:.1f}%\")
print(f\"  False Alarm (FPR):    {vm['false_alarm_FPR']*100:.1f}%\")
print(f\"  Precision:            {vm['precision']*100:.1f}%\")
print(f\"  Overall Accuracy:     {vm['accuracy']*100:.1f}%\")
print(f\"  Confusion: TP={vm['tp']} FN={vm['fn']} FP={vm['fp']} TN={vm['tn']}\")
print(f\"  Simulated: base_p={sim['base_step_accuracy']*100:.1f}% → 1retry_p={sim['effective_step_accuracy_1retry']*100:.1f}%\")
print(f\"  Simulated 6-step TSR: {sim['simulated_TSR_6step_base']*100:.1f}% → {sim['simulated_TSR_6step_1retry']*100:.1f}%\")
vt = d.get('verify_by_action_type', {})
for t in ['click','type','swipe']:
    if t in vt:
        m = vt[t]
        print(f\"  {t:>8s}: TPR={m['TPR']*100:5.1f}% FPR={m['FPR']*100:5.1f}% (wrong={m['wrong_actions']}, correct={m['correct_actions']})\")
"
        echo ""
    fi
done

echo "=========================================="
echo "Pipeline done: $(date)"
echo "=========================================="
