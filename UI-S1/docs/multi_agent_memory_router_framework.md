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

Primary metrics:

```text
macro F1
commit_segment precision/recall
use_full_history recall
replan recall
invalid JSON rate
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
