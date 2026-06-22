# Multi-Agent Memory Router Framework

## Why The Verifier Should Be An Agent

The current method should not be described as a single classifier that routes memory. A classifier can reproduce the offline behavior labels, but it does not give us the right framework claim.

The correct framing is:

```text
multiple context agents propose candidate actions
an explicit verifier agent adjudicates the candidate packet
an execution coordinator commits, rejects, escalates, or replans
```

This makes the method a multi-agent framework rather than a feature-only router.

## Key Distinction From Earlier Two-Agent Failures

Previous two-agent experiments failed because the second actor received an out-of-distribution guided prompt and had to produce the final low-level action.

This verifier is different:

```text
It does not generate a new coordinate/action from scratch.
It receives structured candidate actions from context agents.
It outputs a route decision plus reasons.
```

So the verifier agent is a decision/critique agent, not another action actor.

## Agent Roles

### 1. Local Context Agent

Input:

```text
goal + current screenshot/state + current instruction
```

Output:

```text
no_history candidate action
```

Purpose:

```text
Estimate what the current screen alone supports.
```

### 2. Segment Memory Agent

Input:

```text
goal + current state + compact segment memory + carried values
```

Output:

```text
segment_summary candidate action
```

Purpose:

```text
Propose memory-conditioned repair.
```

### 3. Full History Agent

Input:

```text
goal + current state + recent raw action history
```

Output:

```text
full_history candidate action
```

Purpose:

```text
Provide an independent support signal for candidate validity and rare raw-history rescue.
```

### 4. Distractor Memory Probe Agent

Input:

```text
goal + current state + unrelated segment memory from another episode
```

Output:

```text
wrong_summary candidate action
```

Purpose:

```text
Test memory specificity.
If true segment memory and distractor memory induce the same candidate, the candidate is not memory-specific.
```

### 5. Verifier Agent

Input:

```text
current task packet
all candidate actions
memory proposal score
specificity/progress/full-history consistency evidence
```

Output:

```json
{
  "decision": "use_no_history | commit_segment | use_full_history | replan",
  "selected_condition": "no_history | segment_summary | full_history | null",
  "confidence": "high | medium | low",
  "reason_codes": ["..."],
  "rationale": "short explanation"
}
```

Purpose:

```text
Commit only candidates that are memory-specific, task-progressing, and independently supported.
Reject or escalate ambiguous candidates.
```

### 6. Execution Coordinator

Input:

```text
Verifier Agent decision
```

Action:

```text
use_no_history     -> execute local candidate
commit_segment     -> execute segment-memory candidate
use_full_history   -> execute full-history candidate
replan             -> ask another candidate source or stronger verifier
```

The coordinator is deterministic; the verifier agent owns the reasoning decision.

## Verifier Agent Supervision

The verifier target is derived from counterfactual behavior vectors:

| behavior pattern | verifier decision | reason |
|---|---|---|
| no_history succeeds | use_no_history | current screen is sufficient |
| no_history fails, segment succeeds, wrong fails | commit_segment | memory-specific rescue |
| no_history fails, segment fails, full succeeds | use_full_history | compact memory insufficient |
| segment succeeds, wrong also succeeds | replan | nonspecific context success |
| all contexts fail | replan | candidates unreliable |
| no_history succeeds, segment fails | use_no_history | avoid segment regression |

This keeps the verifier aligned with the research object:

```text
conditional memory utility + candidate validity
```

## Generated Data

Implemented data builder:

```text
scripts/build_verifier_agent_data.py
```

Generated local artifacts:

```text
datasets/verifier_agent_gui_odyssey_all
datasets/verifier_agent_gui_odyssey_hard
```

The hard-only split removes easy current-screen-sufficient examples and focuses training on:

```text
commit_segment
use_full_history
replan
```

Hard-only GUI-Odyssey target distribution:

| split | commit_segment | use_full_history | replan |
|---|---:|---:|---:|
| train | 160 | 173 | 2684 |
| dev | 26 | 29 | 340 |
| test | 21 | 23 | 341 |

This distribution shows why a trained verifier agent must be class-balanced or sampled. Replan dominates the hard set.

## Rule-Agent Baselines

Implemented evaluator:

```text
scripts/evaluate_verifier_agent.py
```

The evaluator supports:

```text
rule baselines
future LLM verifier outputs in the same JSON decision schema
```

Rule-agent baseline on hard-only GUI-Odyssey test:

| baseline | accuracy | macro F1 | main failure |
|---|---:|---:|---|
| commit if segment/full same type | 0.0468 | 0.0261 | predicts commit/no_history, misses replan |
| commit/full/replan rule | 0.1403 | 0.2045 | over-commits segment on hard replan cases |
| specificity+progress+full-support rule | 0.0675 | 0.2162 | still weak on replan separation |

Interpretation:

```text
Simple rules are not enough for the hard verifier task.
The main missing ability is to recognize candidates that look like repairs but should be replanned.
```

This is exactly where an agentic verifier is useful: it can reason over the whole candidate packet and produce explicit reason codes.

## Training Plan

### Stage 1: SFT Verifier Agent

Train on hard-only verifier packets with class-balanced sampling:

```text
commit_segment
use_full_history
replan
```

Input:

```text
candidate packet JSON
```

Target:

```text
strict verifier JSON decision
```

Loss:

```text
standard SFT / JSON decision imitation
```

Prepared balanced SFT data:

```text
scripts/prepare_verifier_agent_sft_data.py
datasets/verifier_agent_gui_odyssey_sft_balanced
```

Balanced train split:

| decision | train rows |
|---|---:|
| commit_segment | 1024 |
| use_full_history | 1024 |
| replan | 1024 |

Original-distribution evaluation splits:

| split | rows | commit_segment | use_full_history | replan |
|---|---:|---:|---:|---:|
| dev | 395 | 26 | 29 | 340 |
| test | 385 | 21 | 23 | 341 |

Generated SFT files:

```text
datasets/verifier_agent_gui_odyssey_sft_balanced/train_balanced.parquet
datasets/verifier_agent_gui_odyssey_sft_balanced/dev.parquet
datasets/verifier_agent_gui_odyssey_sft_balanced/test.parquet
datasets/verifier_agent_gui_odyssey_sft_balanced/dev_balanced.parquet
datasets/verifier_agent_gui_odyssey_sft_balanced/test_balanced.parquet
```

Training entrypoint:

```bash
bash scripts/run_verifier_agent_sft.sh
```

Useful overrides:

```bash
MODEL_PATH=/path/to/text-model \
N_GPUS=4 \
OUTPUT_DIR=outputs/verifier_agent_sft_qwen35 \
bash scripts/run_verifier_agent_sft.sh
```

Default configuration:

```text
model: checkpoints/Qwen3.5-9B
method: LoRA SFT
lora_rank: 32
max_length: 2048
model_dtype: fp32
train file: train_balanced.parquet
val file: dev.parquet
dataset class: GUIMultiTurnSFTDataset
```

Primary metrics:

```text
macro F1
commit_segment precision/recall
use_full_history recall
replan recall
invalid JSON rate
```

### Overnight Post-Train Runbook

Implemented post-train automation:

```text
scripts/generate_verifier_agent_predictions.py
scripts/run_verifier_agent_post_train.sh
```

After SFT finishes, run:

```bash
RUN_DIR=outputs/verifier_agent_sft_qwen35_4gpu_fp32_len2048 \
BASE_MODEL=checkpoints/Qwen3.5-9B \
bash scripts/run_verifier_agent_post_train.sh
```

Or let it wait for the final checkpoint:

```bash
WAIT_FOR_CHECKPOINT=1 \
RUN_DIR=outputs/verifier_agent_sft_qwen35_4gpu_fp32_len2048 \
bash scripts/run_verifier_agent_post_train.sh
```

The script will:

```text
1. locate the latest global_step_* checkpoint,
2. reject DCP-only checkpoints and require the final HF/PEFT checkpoint,
3. generate verifier decisions for dev/test/dev_balanced/test_balanced with batched non-thinking Qwen inference,
4. evaluate JSON decisions with scripts/evaluate_verifier_agent.py,
5. write post_train_summary.md with accuracy, macro F1, class precision/recall, and invalid JSON count.
```

Useful runtime controls:

```text
BATCH_SIZE=8
MAX_NEW_TOKENS=96
SKIP_EXISTING=1
```

`SKIP_EXISTING=1` resumes a partial run by skipping splits that already have both predictions and metrics.

Expected output:

```text
outputs/verifier_agent_sft_qwen35_4gpu_fp32_len2048/post_train_eval/run_info.json
outputs/verifier_agent_sft_qwen35_4gpu_fp32_len2048/post_train_eval/dev_predictions.jsonl
outputs/verifier_agent_sft_qwen35_4gpu_fp32_len2048/post_train_eval/test_predictions.jsonl
outputs/verifier_agent_sft_qwen35_4gpu_fp32_len2048/post_train_eval/*_eval/verifier_eval_report.md
outputs/verifier_agent_sft_qwen35_4gpu_fp32_len2048/post_train_eval/post_train_summary.md
outputs/verifier_agent_sft_qwen35_4gpu_fp32_len2048/post_train_eval/runtime_summary.md
outputs/verifier_agent_sft_qwen35_4gpu_fp32_len2048/coordinator_eval/*/verifier_safety_gate_commands.jsonl
outputs/verifier_agent_sft_qwen35_4gpu_fp32_len2048/coordinator_eval/*/coordinator_report.md
```

### Post-Train Results

Verifier Agent SFT completed successfully:

```text
checkpoint: outputs/verifier_agent_sft_qwen35_4gpu_fp32_len2048/checkpoints/global_step_144
final val/loss: 0.0057
```

Post-train evaluation:

| split | accuracy | macro F1 | commit P/R | full P/R | replan P/R | invalid pred |
|---|---:|---:|---:|---:|---:|---:|
| dev | 0.9038 | 0.5844 | 0.6571/0.8846 | 0.7619/0.5517 | 0.9521/0.9353 | 5 |
| test | 0.9325 | 0.6350 | 0.6364/1.0000 | 0.8182/0.7826 | 0.9877/0.9384 | 6 |
| dev_balanced | 0.7917 | 0.5913 | 0.9587/0.9062 | 0.9577/0.5312 | 0.6250/0.9375 | 0 |
| test_balanced | 0.8880 | 0.6672 | 0.9209/1.0000 | 0.9796/0.7500 | 0.8125/0.9141 | 3 |

Interpretation:

```text
The trained Verifier Agent beats the best rule-agent hard-only macro F1 of 0.2162 by a large margin: test macro F1 is 0.6350.
The agent preserves high replan quality on the original hard test distribution and recovers the rare commit_segment/use_full_history routes.
Remaining invalid predictions are small in count and mostly malformed escaped JSON for replan decisions, not missing route knowledge.
```

### Runtime Coordinator Layer

Recovered runtime scripts:

```text
scripts/verifier_agent_runtime.py
scripts/apply_verifier_agent_coordinator.py
scripts/evaluate_verifier_agent_coordinator.py
```

The Verifier Agent is used as an execution safety gate:

| verifier decision | coordinator command |
|---|---|
| use_no_history | execute no_history_agent candidate |
| commit_segment | execute segment_memory_agent candidate |
| use_full_history | execute full_history_agent candidate |
| replan | emit replan_request; do not execute |
| invalid | emit replan_request; do not execute |

Agent-facing command format:

```json
{
  "status": "execute|replan",
  "verifier_decision": "commit_segment",
  "selected_agent": "segment_memory_agent",
  "selected_condition": "segment_summary",
  "action": {"action": "type", "text": "..."},
  "replan_request": null
}
```

For replan cases, `action` is null and `replan_request` contains the task, memory, all candidate agents, computed evidence, the raw verifier output, and recommended next steps:

```text
generate_alternative_candidate
recover_missing_carried_value
rewrite_current_instruction
rerun_verifier_on_new_packet
```

Historical hard-split coordinator replay from the trained checkpoint:

| split | policy | execute rate | action acc all | executed acc | unsafe exec | replan abstain recall |
|---|---|---:|---:|---:|---:|---:|
| dev | verifier_safety_gate | 0.1418 | 0.0987 | 0.6964 | 0.3036 | 0.9500 |
| test | verifier_safety_gate | 0.1429 | 0.1013 | 0.7091 | 0.2909 | 0.9560 |
| test | always_full_history | 1.0000 | 0.1351 | 0.1351 | 0.8649 | 0.0000 |
| test | oracle_coordinator | 0.1143 | 0.1143 | 1.0000 | 0.0000 | 1.0000 |

Interpretation:

```text
The verifier helps the agent by approving a small set of high-confidence actions and withholding most replan states.
Do not map replan back to no_history or full_history; that destroys the safety behavior.
The next runtime component is a replan resolver that produces a new candidate packet when the verifier refuses the current candidates.
```

### Stage 2: Calibrated Deployment With Coordinator

Use Stage A proposal score before invoking verifier:

```text
if memory proposal score below threshold:
    use_no_history
else:
    call verifier agent
```

This keeps inference cheap and avoids sending easy cases to the verifier.

### Stage 3: Prospective Evaluation

Run the full multi-agent system:

```text
Local Context Agent
Segment Memory Agent
Full History Agent
Distractor Memory Probe Agent
Verifier Agent
Execution Coordinator
```

Compare against:

```text
always no_history
always segment_summary
always full_history
specificity+progress scalar scorer
rule-based cascade
trained verifier agent
```

## Cross-Benchmark Generalization

The verifier agent should generalize better than a scalar classifier because its input schema is benchmark-normalized:

```text
goal
current instruction
candidate actions
segment memory
distractor memory
full-history candidate
computed evidence
```

This does not depend on GUI-Odyssey-specific categories. For AndroidControl and GUI-360, we only need benchmark adapters that produce the same candidate packet.

## Claim We Can Make

With this framing, the method becomes:

```text
A multi-agent counterfactual memory routing framework in which context-specialized agents propose candidate actions and a verifier agent adjudicates memory-specific, task-progressing, and valid candidates.
```

The current evidence supports:

```text
1. GUI-Odyssey behavior data can supervise the verifier agent.
2. Rule baselines are insufficient on hard verifier cases.
3. The verifier-agent packet is benchmark-agnostic.
```

The next required evidence is:

```text
train the verifier agent and show it beats rule-agent baselines on hard GUI-Odyssey cases, then transfer the same packet protocol to AndroidControl.
```

## Runtime Integration After Qwen3.5 Verifier SFT

The trained verifier should be used as a selective hard-state safety gate, not as a full replacement for the base GUI actor.

Recommended online policy:

```text
1. Local Context Agent always proposes no_history action.
2. Cheap hard-state detector decides whether the current state needs arbitration.
3. If not hard, execute no_history immediately.
4. If hard, build the candidate packet:
  - no_history_agent
  - segment_memory_agent
  - full_history_agent
  - distractor_memory_agent
5. Run Verifier Agent on the packet.
6. Execution Coordinator:
  - commit_segment/use_full_history -> execute selected candidate if safety filter passes
  - replan/invalid/rejected executable route -> emit replan_request
7. Resolver Agent consumes replan_request and generates a new candidate packet.
```

The important design rule is:

```text
replan is an escalation signal, not a fallback to no_history or full_history.
```

The concrete CLI for offline replay is:

```bash
.venv/bin/python scripts/apply_verifier_agent_hybrid_policy.py \
  --all-data datasets/verifier_agent_gui_odyssey_all_restore_20260620/test.jsonl \
  --hard-data datasets/verifier_agent_gui_odyssey_hard_restore_20260620/test.jsonl \
  --hard-predictions outputs/verifier_agent_sft_qwen35_retrain_bf16_mb4_len2048_8gpu_epoch_ckpt/post_train_eval/test_predictions.jsonl \
  --output outputs/verifier_agent_sft_qwen35_retrain_bf16_mb4_len2048_8gpu_epoch_ckpt/hybrid_policy/balanced/test/hybrid_commands.jsonl \
  --summary outputs/verifier_agent_sft_qwen35_retrain_bf16_mb4_len2048_8gpu_epoch_ckpt/hybrid_policy/balanced/test/hybrid_summary.json \
  --safety-mode balanced
```

Full-distribution GUI-Odyssey replay with checkpoint `global_step_144`.

Primary metric is strict episode success: an episode succeeds only if all evaluated steps in that episode are correct. Immediate `replan` is counted as failure until a resolver solves it.

| policy | split | episode success | delta vs no_history | resolver-oracle episode success | step/action acc | step delta | hard unsafe exec |
|---|---|---:|---:|---:|---:|---:|---:|
| no_history baseline | dev | 0.7751 | 0 | n/a | 0.9316 | 0 | n/a |
| hybrid raw | dev | 0.7886 | +9 | 0.9925 | 0.9376 | +27 | 8 |
| hybrid balanced | dev | 0.7886 | +9 | 0.9955 | 0.9376 | +27 | 4 |
| hybrid high_precision | dev | 0.7811 | +4 | 0.9970 | 0.9352 | +16 | 3 |
| no_history baseline | test | 0.7796 | 0 | n/a | 0.9284 | 0 | n/a |
| hybrid raw | test | 0.7946 | +10 | 0.9835 | 0.9338 | +23 | 20 |
| hybrid balanced | test | 0.7946 | +10 | 0.9925 | 0.9336 | +22 | 9 |
| hybrid high_precision | test | 0.7931 | +9 | 0.9955 | 0.9322 | +16 | 5 |

Step/action-level diagnostics:

| policy | split | immediate action acc | delta vs no_history | execute rate | hard executed acc | hard unsafe exec | replan count | missed executable |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| no_history baseline | dev | 0.9316 | 0 | 1.0000 | n/a | n/a | 0 | n/a |
| hybrid raw | dev | 0.9376 | +27 | 0.9394 | 0.7714 | 8 | 271 | 0 |
| hybrid balanced | dev | 0.9376 | +27 | 0.9385 | 0.8710 | 4 | 275 | 0 |
| hybrid high_precision | dev | 0.9352 | +16 | 0.9358 | 0.8421 | 3 | 287 | 9 |
| no_history baseline | test | 0.9284 | 0 | 1.0000 | n/a | n/a | 0 | n/a |
| hybrid raw | test | 0.9338 | +23 | 0.9385 | 0.5349 | 20 | 263 | 2 |
| hybrid balanced | test | 0.9336 | +22 | 0.9357 | 0.7097 | 9 | 275 | 5 |
| hybrid high_precision | test | 0.9322 | +16 | 0.9333 | 0.7619 | 5 | 285 | 10 |

Deployment recommendation:

```text
Use balanced mode by default: it keeps almost all raw-mode gains while removing most unsafe hard-state executions.
Use high_precision mode when safety is much more important than immediate recovery.
Use raw mode only when immediate success is more important than avoiding unsafe hard-state execution.
```

This gives a direct improvement over the no-history agent on the full distribution while preserving the multi-agent safety claim. The next performance gain should come from the resolver agent: every hard-state `replan_request` is a chance to recover an otherwise failing step.

## Resolver Agent Training Target

The next agent should consume `replan_request` and emit a corrected low-level GUI action. We prepared this dataset with:

```bash
.venv/bin/python scripts/prepare_resolver_agent_sft_data.py \
  --hard-dir datasets/verifier_agent_gui_odyssey_hard_restore_20260620 \
  --hybrid-dir outputs/verifier_agent_sft_qwen35_retrain_bf16_mb4_len2048_8gpu_epoch_ckpt/hybrid_policy \
  --mode balanced \
  --output-dir datasets/resolver_agent_gui_odyssey_balanced_20260622
```

Generated split sizes:

| split | rows | main target actions |
|---|---:|---|
| train | 2225 | terminate:1086, swipe:341, system_button:332, click:241, type:223 |
| dev | 275 | terminate:123, swipe:45, system_button:39, type:37, click:28 |
| test | 275 | terminate:135, click:52, system_button:40, swipe:26, type:22 |

Resolver upper-bound on top of balanced hybrid policy:

| split | balanced episode success | resolver oracle episode success | balanced step/action acc | resolver oracle step/action acc |
|---|---:|---:|---:|---:|
| dev | 0.7886 | 0.9955 | 0.9376 | 0.9991 |
| test | 0.7946 | 0.9925 | 0.9336 | 0.9979 |

This makes the resolver the most promising next performance lever. The balanced verifier already improves strict test episode success by 10 episodes over no-history. If a resolver solves the remaining replan requests, the oracle upper bound is 0.9925 episode success on test.
