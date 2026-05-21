#!/bin/bash
#SBATCH --job-name=ev_v13_abl
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/eval_ablation_gui360tmpl_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/eval_ablation_gui360tmpl_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --dependency=afterok:4569412

# ============================================
# V13 Ablation: LoRA masking × Communication
#
# Uses the BEST epoch from re-eval job 4569412, runs 5 ablation variants
# with eval_gui360_template.py (corrected prompt matching training).
#
# Variants:
#   1. normal + no_comm    (both experts, routing intact, comm disabled)
#   2. expert_1_only + comm
#   3. expert_1_only + no_comm
#   4. expert_2_only + comm
#   5. expert_2_only + no_comm
#
# (The full model = normal + comm is already evaluated by job 4569412)
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
CKPT_ROOT="$PROJECT_ROOT/checkpoints/v13_gui360_rl_balanced"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p "$PROJECT_ROOT/v13_gui_360/scripts/logs"

echo "=========================================="
echo "V13 Ablation: LoRA masking × Communication"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

# ---- Auto-select best epoch from re-eval results ----
select_best_epoch() {
    local best_epoch=""
    local best_tsr=-1

    for ep in epoch-0 epoch-1 epoch-2; do
        local summary_dir="$PROJECT_ROOT/v13_gui_360/outputs/balanced_gui360tmpl_${ep}"
        local summary_file=$(ls "$summary_dir"/eval_summary_*.json 2>/dev/null | tail -1)
        if [ -z "$summary_file" ]; then
            echo "WARNING: No re-eval summary for $ep, skipping." >&2
            continue
        fi
        local tsr=$(python3 -c "import json; d=json.load(open('$summary_file')); print(d['tsr'])")
        echo "  $ep: TSR=$tsr" >&2
        local is_better=$(python3 -c "print(1 if $tsr > $best_tsr else 0)")
        if [ "$is_better" = "1" ]; then
            best_tsr=$tsr
            best_epoch=$ep
        fi
    done

    if [ -z "$best_epoch" ]; then
        echo "ERROR: No valid re-eval summaries found!" >&2
        exit 1
    fi

    echo "  => Best: $best_epoch (TSR=$best_tsr)" >&2
    echo "$best_epoch"
}

echo ""
echo "Selecting best epoch from re-eval..."
BEST_EPOCH=$(select_best_epoch)
COOP_DIR="$CKPT_ROOT/${BEST_EPOCH}/cooperative"
echo "Using checkpoint: $COOP_DIR"
echo ""

# Verify checkpoint
if [ ! -f "$COOP_DIR/lora_weights.pt" ]; then
    echo "ERROR: lora_weights.pt not found at $COOP_DIR"
    exit 1
fi

# ---- Generic eval function ----
eval_variant() {
    local TAG="$1"          # e.g. "nocomm_normal", "comm_expert_1_only"
    local EXPERT_FLAG="$2"  # "" or "--expert_mode expert_1_only"
    local COMM_FLAG="$3"    # "" or "--no_comm"

    local MODEL_NAME="v13_abl_${TAG}"
    local OUTPUT_DIR="$PROJECT_ROOT/v13_gui_360/outputs/ablation_gui360tmpl_${BEST_EPOCH}_${TAG}"

    if ls "$OUTPUT_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
        echo "[${TAG}] Already evaluated, skipping."
        return 0
    fi

    echo ""
    echo "=========================================="
    echo "Ablation: ${TAG}  (${BEST_EPOCH})"
    echo "Time: $(date)"
    echo "=========================================="

    mkdir -p "$OUTPUT_DIR"

    # Start server
    python "$PROJECT_ROOT/v13_gui_360/serve_cooperative_direct.py" \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$COOP_DIR" \
        --version v13 \
        --served-model-name "$MODEL_NAME" \
        $EXPERT_FLAG \
        $COMM_FLAG \
        --port 8000 \
        &
    local SERVER_PID=$!

    # Wait for server
    local WAIT_COUNT=0
    while [ $WAIT_COUNT -lt 600 ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "Server ready after ${WAIT_COUNT}s"
            break
        fi
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
    done

    if [ $WAIT_COUNT -ge 600 ]; then
        echo "ERROR: Server failed to start for ${TAG}"
        kill $SERVER_PID 2>/dev/null || true
        return 1
    fi

    # Eval with corrected gui360_template
    python "$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --threads 4

    # Shutdown
    echo "Shutting down server for ${TAG}..."
    kill $SERVER_PID 2>/dev/null || true
    pkill -f "serve_cooperative_direct" 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 10
}

# ---- Run all 5 ablation variants ----

# 1. Both experts, no communication
eval_variant "nocomm_normal" "" "--no_comm"

# 2. Expert 1 only, with communication
eval_variant "comm_expert1" "--expert_mode expert_1_only" ""

# 3. Expert 1 only, no communication
eval_variant "nocomm_expert1" "--expert_mode expert_1_only" "--no_comm"

# 4. Expert 2 only, with communication
eval_variant "comm_expert2" "--expert_mode expert_2_only" ""

# 5. Expert 2 only, no communication
eval_variant "nocomm_expert2" "--expert_mode expert_2_only" "--no_comm"

# ============================================
# Collect results and update report
# ============================================
echo ""
echo "=========================================="
echo "Collecting ablation results..."
echo "=========================================="

python3 - "$BEST_EPOCH" <<'PYEOF'
import json
import os
import sys
import glob
import datetime

BEST_EPOCH = sys.argv[1]
PROJECT = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
OUTPUT_ROOT = f"{PROJECT}/v13_gui_360/outputs"
REPORT_PATH = f"{PROJECT}/v13_gui_360/fair_comparison_baseline.md"

def load_summary(pattern):
    files = sorted(glob.glob(pattern))
    if files:
        with open(files[-1]) as f:
            return json.load(f)
    return None

# Load full-model result from re-eval
full_model = load_summary(f"{OUTPUT_ROOT}/balanced_gui360tmpl_{BEST_EPOCH}/eval_summary_*.json")

# Load all re-eval epoch results
reeval_results = {}
for ep in range(3):
    tag = f"epoch-{ep}"
    r = load_summary(f"{OUTPUT_ROOT}/balanced_gui360tmpl_{tag}/eval_summary_*.json")
    if r:
        reeval_results[tag] = r

# Load old eval results for comparison
old_results = {}
for ep in range(3):
    tag = f"epoch-{ep}"
    r = load_summary(f"{OUTPUT_ROOT}/balanced_{tag}/eval_summary_*.json")
    if r:
        old_results[tag] = r

# Load ablation results
ablation_tags = [
    ("nocomm_normal",  "Both experts", "No"),
    ("comm_expert1",   "Expert 1 only", "Yes"),
    ("nocomm_expert1", "Expert 1 only", "No"),
    ("comm_expert2",   "Expert 2 only", "Yes"),
    ("nocomm_expert2", "Expert 2 only", "No"),
]
ablation_results = {}
for tag, _, _ in ablation_tags:
    r = load_summary(f"{OUTPUT_ROOT}/ablation_gui360tmpl_{BEST_EPOCH}_{tag}/eval_summary_*.json")
    if r:
        ablation_results[tag] = r
        print(f"  {tag}: TSR={r['tsr']*100:.1f}%  StepSR={r['step_sr']*100:.1f}%  Progress={r['avg_progress']*100:.1f}%")
    else:
        print(f"  {tag}: no results")

# Load SFT baselines
sft_results = {}
for name, pattern in [
    ("lora_r256", f"{PROJECT}/v15_gui_360/outputs/eval_lora_r256/eval_summary_*.json"),
    ("gui_action", f"{PROJECT}/v13_gui_360/outputs/gui_action_balanced/eval_summary_*.json"),
    ("base_model", f"{PROJECT}/v13_gui_360/outputs/base_gui360_test/eval_summary_*.json"),
]:
    r = load_summary(pattern)
    if r:
        sft_results[name] = r

# ── Build report ──
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

lines = []
lines.append("# Fair Comparison: V13 Cooperative RL — Corrected Eval + Ablations")
lines.append("")
lines.append(f"_Last updated: {ts}_")
lines.append("")

# ── What changed ──
lines.append("## What Changed")
lines.append("")
lines.append("Previous V13 RL eval used `eval_cooperative.py` with a **different prompt format**")
lines.append("than training (`Task: {goal}...` + `<action>` output). Now re-evaluated with")
lines.append("`eval_gui360_template.py` using the **exact same prompt** as training:")
lines.append("- `SUPPORTED_ACTIONS`: character-identical to training")
lines.append("- History format: `Step N: click(coordinate=[x, y])` (matching training)")
lines.append("")
lines.append("---")
lines.append("")

# ── Setup ──
lines.append("## Experiment Setup")
lines.append("")
lines.append("- Base model: **Qwen2.5-VL-7B-Instruct**")
lines.append("- Training data: **balanced 2000-episode** (17,264 steps)")
lines.append("- Test set: `gui360_test_1000_balanced.jsonl` (1000 episodes, stop-on-error)")
lines.append("- All evals use **gui360_template** format (= training format)")
lines.append("")
lines.append("---")
lines.append("")

# ── Re-eval results ──
lines.append("## V13 Cooperative RL — Re-eval (corrected prompt)")
lines.append("")
lines.append("| Checkpoint | TSR | StepSR | Progress |")
lines.append("|---|---|---|---|")
for ep in range(3):
    tag = f"epoch-{ep}"
    if tag in reeval_results:
        r = reeval_results[tag]
        best = " **(best)**" if tag == BEST_EPOCH else ""
        lines.append(f"| {tag}{best} | {r['tsr']*100:.1f}% | {r['step_sr']*100:.1f}% | {r['avg_progress']*100:.1f}% |")
lines.append("")

# Old vs new comparison
if old_results:
    lines.append("### Old eval vs corrected eval")
    lines.append("")
    lines.append("| Checkpoint | Old TSR (wrong prompt) | New TSR (corrected) | Delta |")
    lines.append("|---|---|---|---|")
    for ep in range(3):
        tag = f"epoch-{ep}"
        old_tsr = f"{old_results[tag]['tsr']*100:.1f}%" if tag in old_results else "—"
        new_tsr = f"{reeval_results[tag]['tsr']*100:.1f}%" if tag in reeval_results else "—"
        if tag in old_results and tag in reeval_results:
            delta = (reeval_results[tag]['tsr'] - old_results[tag]['tsr']) * 100
            delta_str = f"{delta:+.1f}pp"
        else:
            delta_str = "—"
        lines.append(f"| {tag} | {old_tsr} | {new_tsr} | {delta_str} |")
    lines.append("")

lines.append("---")
lines.append("")

# ── Ablation table ──
lines.append(f"## Ablation Study ({BEST_EPOCH})")
lines.append("")
lines.append("| LoRA Experts | Communication | TSR | StepSR | Progress |")
lines.append("|---|---|---|---|---|")

# Full model (from re-eval)
if full_model:
    lines.append(f"| Both (soft routing) | Yes (T=2) | **{full_model['tsr']*100:.1f}%** | {full_model['step_sr']*100:.1f}% | {full_model['avg_progress']*100:.1f}% |")

# Ablation variants
for tag, expert_label, comm_label in ablation_tags:
    if tag in ablation_results:
        r = ablation_results[tag]
        lines.append(f"| {expert_label} | {comm_label} | {r['tsr']*100:.1f}% | {r['step_sr']*100:.1f}% | {r['avg_progress']*100:.1f}% |")
    else:
        lines.append(f"| {expert_label} | {comm_label} | — | — | — |")

lines.append("")

# ── Analysis ──
lines.append("### Ablation Analysis")
lines.append("")

if full_model and ablation_results:
    full_tsr = full_model['tsr']

    if "nocomm_normal" in ablation_results:
        nocomm_tsr = ablation_results["nocomm_normal"]["tsr"]
        comm_delta = (full_tsr - nocomm_tsr) * 100
        lines.append(f"- **Communication effect**: {full_tsr*100:.1f}% → {nocomm_tsr*100:.1f}% without comm "
                     f"({comm_delta:+.1f}pp)")

    for exp_label, comm_tag, nocomm_tag in [
        ("Expert 1", "comm_expert1", "nocomm_expert1"),
        ("Expert 2", "comm_expert2", "nocomm_expert2"),
    ]:
        if comm_tag in ablation_results and nocomm_tag in ablation_results:
            c = ablation_results[comm_tag]["tsr"]
            nc = ablation_results[nocomm_tag]["tsr"]
            lines.append(f"- **{exp_label} comm effect**: {c*100:.1f}% (comm) vs {nc*100:.1f}% (no comm) "
                         f"({(c-nc)*100:+.1f}pp)")
        if comm_tag in ablation_results:
            e = ablation_results[comm_tag]["tsr"]
            lines.append(f"- **{exp_label} vs full**: {e*100:.1f}% vs {full_tsr*100:.1f}% "
                         f"({(e-full_tsr)*100:+.1f}pp)")

    if "comm_expert1" in ablation_results and "comm_expert2" in ablation_results:
        e1 = ablation_results["comm_expert1"]["tsr"]
        e2 = ablation_results["comm_expert2"]["tsr"]
        lines.append(f"- **Expert specialization gap**: E1={e1*100:.1f}% vs E2={e2*100:.1f}% "
                     f"(delta={abs(e1-e2)*100:.1f}pp)")

lines.append("")
lines.append("---")
lines.append("")

# ── Reference baselines ──
lines.append("## Reference Baselines (gui360_template)")
lines.append("")
lines.append("| Model | TSR | StepSR | Progress | Notes |")
lines.append("|---|---|---|---|---|")

if "base_model" in sft_results:
    r = sft_results["base_model"]
    lines.append(f"| base_model (no FT) | {r['tsr']*100:.1f}% | {r['step_sr']*100:.1f}% | {r['avg_progress']*100:.1f}% | |")
else:
    lines.append("| base_model (no FT) | 2.4% | 26.4% | 8.0% | from prev eval |")

if "lora_r256" in sft_results:
    r = sft_results["lora_r256"]
    lines.append(f"| Standard LoRA r=256 SFT | {r['tsr']*100:.1f}% | {r['step_sr']*100:.1f}% | {r['avg_progress']*100:.1f}% | 4 epochs, same data |")
else:
    lines.append("| Standard LoRA r=256 SFT | 6.4% | 48.8% | 18.1% | from prev eval |")

if "gui_action" in sft_results:
    r = sft_results["gui_action"]
    lines.append(f"| gui_action (ref, 97K data) | {r['tsr']*100:.1f}% | {r['step_sr']*100:.1f}% | {r['avg_progress']*100:.1f}% | upper bound |")
else:
    lines.append("| gui_action (ref, 97K data) | 14.4% | 65.2% | 30.6% | from prev eval |")

lines.append("")

report = "\n".join(lines)
with open(REPORT_PATH, "w") as f:
    f.write(report)

print(f"\nReport written to {REPORT_PATH}")
PYEOF

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
