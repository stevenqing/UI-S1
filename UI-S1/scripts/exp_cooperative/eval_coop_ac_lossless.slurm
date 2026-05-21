#!/bin/bash
#SBATCH --job-name=eval_ac_ll
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_coop_ac_lossless_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_coop_ac_lossless_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Lossless eval of cooperative LoRA v6.5 on AC
#
# 1) Lossless merge: LoRA_A into base + correction adapters
# 2) Run 4 HF-based shards (1 GPU each) for AR trajectory eval
#
# Usage:
#   COOP_EPOCH=2 sbatch eval_coop_ac_lossless.slurm
#   COOP_EPOCH=4 sbatch eval_coop_ac_lossless.slurm
# ============================================

set -e

COOP_EPOCH="${COOP_EPOCH:-4}"
NUM_SHARDS=4

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
COOP_CKPT="$PROJECT_ROOT/train_GUI_360/llamafactory/output/cooperative_v6_5_ac/epoch-${COOP_EPOCH}"
MERGED_MODEL="/tmp/merged_lossless_ac_ep${COOP_EPOCH}_${SLURM_JOB_ID}"
DATASET_PATH="$PROJECT_ROOT/datasets/android_control_evaluation_std.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/outputs/eval_cooperative_v6_5_ac/epoch${COOP_EPOCH}_lossless"

echo "=========================================="
echo "Lossless Eval: Cooperative v6.5 AC epoch-${COOP_EPOCH}"
echo "=========================================="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Base:      $BASE_MODEL"
echo "COOP_CKPT: $COOP_CKPT"
echo "Merged:    $MERGED_MODEL"
echo "Shards:    $NUM_SHARDS"
echo "Output:    $OUTPUT_DIR"
echo "Start:     $(date)"
echo "=========================================="

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p "$OUTPUT_DIR"

# Step 1: Lossless merge (LoRA_A into base + correction adapters)
echo "Step 1: Lossless merge..."
python evaluation/merge_cooperative_lossless.py \
    --base_model "$BASE_MODEL" \
    --coop_checkpoint "$COOP_CKPT" \
    --output_dir "$MERGED_MODEL"

MERGE_EXIT=$?
if [ $MERGE_EXIT -ne 0 ]; then
    echo "ERROR: Lossless merge failed with exit code $MERGE_EXIT"
    exit $MERGE_EXIT
fi
echo "Merge complete: $MERGED_MODEL"
ls -la "$MERGED_MODEL"/correction_*

# Step 2: Launch 4 shards (1 GPU each) for AR trajectory eval
echo "Step 2: Running AR trajectory eval ($NUM_SHARDS shards)..."

PIDS=()
for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    echo "  Launching shard $SHARD_ID on GPU $SHARD_ID..."
    CUDA_VISIBLE_DEVICES=$SHARD_ID python evaluation/eval_cooperative_ac_trajectory.py \
        --merged_dir "$MERGED_MODEL" \
        --jsonl_file "$DATASET_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --gpu_id 0 \
        --shard_id $SHARD_ID \
        --num_shards $NUM_SHARDS \
        --n_history_image_limit 2 \
        --no_stop \
        &
    PIDS+=($!)
done

echo "  Waiting for ${#PIDS[@]} shards..."
EVAL_EXIT=0
for PID in "${PIDS[@]}"; do
    wait $PID || EVAL_EXIT=$?
done

# Step 3: Aggregate results
echo "Step 3: Aggregating shard results..."
python -c "
import json, os, glob

output_dir = '$OUTPUT_DIR'
all_results = []
for f in sorted(glob.glob(os.path.join(output_dir, 'trajectory_results_shard*.jsonl'))):
    with open(f) as fh:
        for line in fh:
            all_results.append(json.loads(line.strip()))

n = len(all_results)
if n == 0:
    print('No results found!')
    exit(1)

success = sum(1 for r in all_results if r['task_success'])
tsr = success / n
total_steps = sum(r['num_steps'] for r in all_results)
total_correct = sum(r['final_step_id'] for r in all_results)
scattered = total_correct / total_steps if total_steps else 0
progresses = [r['final_step_id'] / r['num_steps'] for r in all_results if r['num_steps'] > 0]
avg_prog = sum(progresses) / len(progresses) if progresses else 0

summary = {
    'total_episodes': n,
    'tsr': tsr,
    'avg_progress': avg_prog,
    'scattered_progress': scattered,
    'success_count': success,
}
with open(os.path.join(output_dir, 'summary_all.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f'TSR={tsr:.4f} ({success}/{n})  AvgProg={avg_prog:.4f}  Scattered={scattered:.4f}')
"

# Cleanup
rm -rf "$MERGED_MODEL"

echo "=========================================="
echo "Done: $(date)  exit=$EVAL_EXIT"
echo "Results: $OUTPUT_DIR"
echo "=========================================="
exit $EVAL_EXIT
