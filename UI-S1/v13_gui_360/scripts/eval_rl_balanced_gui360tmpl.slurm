#!/bin/bash
#SBATCH --job-name=ev_v13_g360
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/eval_rl_balanced_gui360tmpl_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/eval_rl_balanced_gui360tmpl_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V13 GUI-360 RE-EVAL — RL balanced, gui360_template format
#
# Re-evaluates V13 RL checkpoints (trained from base) using
# eval_gui360_template.py with the CORRECTED prompt:
#   - SUPPORTED_ACTIONS identical to training
#   - History formatted as action strings (not model thoughts)
#
# Previous eval used eval_cooperative.py which had a completely
# different prompt format ("Task: {goal}..." + <action> output)
# vs training ("You are a helpful assistant..." + <tool_call> output).
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
echo "V13 RE-EVAL: RL balanced with gui360_template"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

eval_coop() {
    local CKPT_TAG="$1"
    local COOP_DIR="$CKPT_ROOT/${CKPT_TAG}/cooperative"
    local MODEL_NAME="v13_bal_${CKPT_TAG}"
    local OUTPUT_DIR="$PROJECT_ROOT/v13_gui_360/outputs/balanced_gui360tmpl_${CKPT_TAG}"

    # Skip if already evaluated
    if ls "$OUTPUT_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
        echo "[${CKPT_TAG}] Already evaluated, skipping."
        return 0
    fi

    # Check checkpoint exists
    if [ ! -f "$COOP_DIR/lora_weights.pt" ]; then
        echo "[${CKPT_TAG}] Checkpoint not found at $COOP_DIR, skipping."
        return 0
    fi

    echo ""
    echo "=========================================="
    echo "Evaluating: V13 RL ${CKPT_TAG} (gui360_template)"
    echo "Time: $(date)"
    echo "=========================================="

    mkdir -p "$OUTPUT_DIR"

    # Start cooperative model server
    python "$PROJECT_ROOT/v13_gui_360/serve_cooperative_direct.py" \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$COOP_DIR" \
        --version v13 \
        --served-model-name "$MODEL_NAME" \
        --port 8000 \
        &
    local SERVER_PID=$!

    # Wait for server readiness
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
        echo "ERROR: Server failed to start for ${CKPT_TAG}"
        kill $SERVER_PID 2>/dev/null || true
        return 1
    fi

    # Run eval with gui360_template (matches training prompt)
    python "$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --threads 4

    # Shutdown server
    echo "Shutting down server for ${CKPT_TAG}..."
    kill $SERVER_PID 2>/dev/null || true
    pkill -f "serve_cooperative_direct" 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 10
}

# Evaluate epoch-0 through epoch-2
for ep in epoch-0 epoch-1 epoch-2; do
    eval_coop "$ep"
done

# ============================================
# After all evals: collect results and update report
# ============================================
echo ""
echo "=========================================="
echo "Collecting results..."
echo "=========================================="

python3 - <<'PYEOF'
import json
import os
import glob
import datetime

PROJECT = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
OUTPUT_ROOT = f"{PROJECT}/v13_gui_360/outputs"
REPORT_PATH = f"{PROJECT}/v13_gui_360/fair_comparison_baseline.md"

# Collect V13 RL balanced results (gui360_template re-eval)
rl_results = {}
for ep in range(3):
    tag = f"epoch-{ep}"
    out_dir = f"{OUTPUT_ROOT}/balanced_gui360tmpl_{tag}"
    summaries = sorted(glob.glob(f"{out_dir}/eval_summary_*.json"))
    if summaries:
        with open(summaries[-1]) as f:
            data = json.load(f)
        rl_results[tag] = data
        print(f"  {tag}: TSR={data['tsr']*100:.1f}%  StepSR={data['step_sr']*100:.1f}%  Progress={data['avg_progress']*100:.1f}%")
    else:
        print(f"  {tag}: no results found")

if not rl_results:
    print("ERROR: No results collected, skipping report update.")
    exit(0)

# Find best epoch
best_tag = max(rl_results, key=lambda t: rl_results[t]["tsr"])
best = rl_results[best_tag]

# Also try to collect old eval results for comparison
old_results = {}
for ep in range(3):
    tag = f"epoch-{ep}"
    out_dir = f"{OUTPUT_ROOT}/balanced_{tag}"
    summaries = sorted(glob.glob(f"{out_dir}/eval_summary_*.json"))
    if summaries:
        with open(summaries[-1]) as f:
            old_results[tag] = json.load(f)

# Also collect SFT baseline results if available
sft_results = {}
for name, pattern in [
    ("lora_r256", f"{PROJECT}/v15_gui_360/outputs/eval_lora_r256/eval_summary_*.json"),
    ("gui_action", f"{PROJECT}/v13_gui_360/outputs/gui_action_balanced/eval_summary_*.json"),
    ("base_model", f"{PROJECT}/v13_gui_360/outputs/base_gui360_test/eval_summary_*.json"),
]:
    files = sorted(glob.glob(pattern))
    if files:
        with open(files[-1]) as f:
            sft_results[name] = json.load(f)

# Build the updated report
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

lines = []
lines.append("# Fair Comparison: V13 Cooperative RL (from base) — Corrected Eval")
lines.append("")
lines.append(f"_Last updated: {ts}_")
lines.append("")
lines.append("## What Changed")
lines.append("")
lines.append("Previous V13 RL eval used `eval_cooperative.py` with a **different prompt format**")
lines.append("than training. The training script uses the GUI-360 template (`USER_PROMPT_TEMPLATE`")
lines.append("+ `SUPPORTED_ACTIONS` + `<tool_call>` output), but the old eval used a minimal format")
lines.append('(`"Task: {goal}\\nStep {i}: What action?"` + `<action>` output).')
lines.append("")
lines.append("Fixes applied to `eval_gui360_template.py`:")
lines.append("1. **SUPPORTED_ACTIONS**: now character-identical to training")
lines.append("2. **History format**: predicted actions formatted as `Step N: click(coordinate=[x, y])`")
lines.append("   (matching training), instead of raw model reasoning text")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## Experiment Setup")
lines.append("")
lines.append("All models start from **Qwen2.5-VL-7B-Instruct**, trained on the")
lines.append("**same balanced 2000-episode data** (17,264 steps).")
lines.append("")
lines.append("Test set: `gui360_test_1000_balanced.jsonl` (1000 episodes, stop-on-error)")
lines.append("")
lines.append("All models below evaluated with **gui360_template** format (= training format).")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## V13 Cooperative RL (from base, no SFT warmstart)")
lines.append("")
lines.append("| Checkpoint | TSR | StepSR | Progress | Episodes |")
lines.append("|---|---|---|---|---|")

for ep in range(3):
    tag = f"epoch-{ep}"
    if tag in rl_results:
        r = rl_results[tag]
        marker = " **best**" if tag == best_tag else ""
        lines.append(f"| {tag} | {r['tsr']*100:.1f}% | {r['step_sr']*100:.1f}% | {r['avg_progress']*100:.1f}% | {r['num_episodes']}{marker} |")
    else:
        lines.append(f"| {tag} | — | — | — | — |")

lines.append("")

# Old eval comparison
if old_results:
    lines.append("### Previous eval (wrong prompt format, for reference)")
    lines.append("")
    lines.append("These used `eval_cooperative.py` with a minimal prompt format that did NOT")
    lines.append("match training. Included for comparison only.")
    lines.append("")
    lines.append("| Checkpoint | TSR (old) | TSR (corrected) | Delta |")
    lines.append("|---|---|---|---|")
    for ep in range(3):
        tag = f"epoch-{ep}"
        old_tsr = f"{old_results[tag]['tsr']*100:.1f}%" if tag in old_results else "—"
        new_tsr = f"{rl_results[tag]['tsr']*100:.1f}%" if tag in rl_results else "—"
        if tag in old_results and tag in rl_results:
            delta = (rl_results[tag]['tsr'] - old_results[tag]['tsr']) * 100
            delta_str = f"{delta:+.1f}pp"
        else:
            delta_str = "—"
        lines.append(f"| {tag} | {old_tsr} | {new_tsr} | {delta_str} |")
    lines.append("")

lines.append("---")
lines.append("")
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
lines.append("---")
lines.append("")
lines.append("## Analysis")
lines.append("")

if rl_results:
    best_r = rl_results[best_tag]
    lines.append(f"- **Best V13 RL checkpoint**: {best_tag} with TSR={best_r['tsr']*100:.1f}%")
    lines.append(f"- **V13 RL vs base**: {best_r['tsr']*100:.1f}% vs ~2.4% = "
                 f"{best_r['tsr']/0.024:.1f}x relative improvement")
    if "lora_r256" in sft_results:
        lora_tsr = sft_results["lora_r256"]["tsr"]
        lines.append(f"- **V13 RL vs LoRA SFT**: {best_r['tsr']*100:.1f}% vs {lora_tsr*100:.1f}% "
                     f"(same prompt format, apples-to-apples)")

if old_results and rl_results:
    lines.append("")
    lines.append("### Impact of prompt fix")
    for ep in range(3):
        tag = f"epoch-{ep}"
        if tag in old_results and tag in rl_results:
            delta = (rl_results[tag]["tsr"] - old_results[tag]["tsr"]) * 100
            lines.append(f"- {tag}: {old_results[tag]['tsr']*100:.1f}% → {rl_results[tag]['tsr']*100:.1f}% ({delta:+.1f}pp)")

lines.append("")

report = "\n".join(lines)
with open(REPORT_PATH, "w") as f:
    f.write(report)

print(f"\nReport written to {REPORT_PATH}")
PYEOF

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
