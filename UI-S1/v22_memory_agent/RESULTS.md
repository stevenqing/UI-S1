# V21-V22 Exploration Results Summary

## Baseline Reference

| Method | TSR | StepSR | Progress | Cost |
|--------|-----|--------|----------|------|
| Standard SFT-272 | 21.7% | 47.1% | 34.8% | 1x |
| Type Focused prompt | 23.6% | 47.1% | 36.5% | 1x |
| BoN-5 LogProb | 26.0% | 53.1% | 40.6% | 5x |
| Oracle subtask thought | 35.2% | — | — | 1x |

---

## V21: Thought SFT (Training Method)

Train on oracle thought annotations via SFT, hoping the model internalizes reasoning.

### Results

| Config | TSR | StepSR | Progress |
|--------|-----|--------|----------|
| ckpt-80, GT history | 15.2% | 52.2% | 28.6% |
| ckpt-80, pred history | **14.8%** | 44.9% | 28.4% |
| ckpt-80, train set, GT hist | 17.3% | 54.6% | 31.6% |
| ckpt-80, train set, pred hist | 16.5% | 46.6% | 31.1% |

### Per-Task-Length Breakdown (best: ckpt-80, GT history)

| Length | Baseline | V21 |
|--------|----------|-----|
| 1 step | 63.9% | 53.8% |
| 2-3 | 31.8% | 22.1% |
| 4-5 | 23.7% | 12.6% |
| 6+ | 6.6% | 4.1% |

### Conclusion

Complete failure. All length buckets regressed. The model learned the thought format but hallucinated wrong reasoning, causing worse performance than no reasoning at all.

---

## V22: Memory-Augmented Multi-Angle Reasoning (Inference-Time Method)

External retrieved knowledge (Dim 1: memory types) + diverse reasoning angles (Dim 3) + consensus voting.

### Dim 1: Memory Types (standard angle, original run)

| Memory | TSR | StepSR | Progress | 1-step | 2-3 | 4-5 | 6+ |
|--------|-----|--------|----------|--------|-----|-----|-----|
| goal | 21.5% | 46.5% | 34.6% | 63.9% | 35.4% | 21.7% | 5.5% |
| procedural | 22.1% | 46.0% | 35.5% | 64.7% | 32.8% | 23.7% | 6.8% |
| type | 22.0% | 46.3% | 35.2% | 66.4% | 34.9% | 22.7% | 5.7% |
| visual | 22.5% | 46.4% | 34.9% | 67.2% | 34.9% | 24.2% | 5.9% |

### Dim 3: Reasoning Angles (no memory)

| Angle | TSR | StepSR | Progress | 1-step | 2-3 | 4-5 | 6+ |
|-------|-----|--------|----------|--------|-----|-----|-----|
| what_type | 22.2% | 47.2% | 35.3% | 63.0% | 30.8% | **28.3%** | 6.4% |
| where | 21.8% | 46.9% | 35.5% | 62.2% | 32.8% | 25.3% | 6.1% |
| when | 21.4% | 46.2% | 35.1% | 62.2% | 30.8% | 25.3% | 6.1% |
| why | 21.9% | 46.9% | 35.2% | 62.2% | 33.3% | 25.3% | 6.1% |

### Ensembles (5x cost)

| Ensemble | TSR | StepSR | Progress | 1-step | 2-3 | 4-5 | 6+ |
|----------|-----|--------|----------|--------|-----|-----|-----|
| 4 memory + anchor | 22.9% | 48.7% | 36.5% | 67.2% | 35.4% | 25.3% | 6.1% |
| 4 angle + anchor | 21.9% | 47.5% | 35.2% | 64.7% | 31.8% | 25.3% | 6.1% |
| 4 mem+angle + anchor | 23.0% | 48.1% | 36.4% | 66.4% | 34.9% | 24.7% | 7.0% |

### Ablation: Memory Content vs Template Effect

Controlled experiment to isolate whether improvements come from memory content or the GUIDED prompt template.

| Experiment | TSR | vs | Delta | Interpretation |
|------------|-----|----|-------|----------------|
| standard baseline | 21.7% | — | — | STANDARD template, no memory |
| random memory | 21.6% | vs standard | -0.1% | Template effect = ~0 |
| goal memory (fixed) | 21.9% | vs random | +0.3% | Goal content ineffective |
| type memory (fixed) | 22.1% | vs random | +0.5% | Marginal |
| **visual memory (fixed)** | **23.1%** | vs random | **+1.5%** | Only effective memory |

Note: "fixed" variants correct a GT action-type leak in the original `memory_type` experiment (impact was negligible: original 22.0% vs fixed 22.1%).

---

## Key Findings

1. **Training thought (V21) failed completely** — TSR dropped from 21.9% to 14.8%. Hallucinated reasoning is worse than no reasoning.

2. **Inference-time memory (V22) has a low ceiling** — Best single-model (visual memory 23.1%) still below type_focused prompt (23.6%) and far below BoN-5 (26.0%).

3. **Only visual memory provides real value** — Ablation confirmed goal/procedural/type memory content contributes at most +0.5% vs random examples. Visual memory's app-matching + position-proximity gives +1.5% genuine improvement.

4. **Angle prompts are mostly ineffective** — All 4 angles hover near baseline. One exception: `what_type` reaches 28.3% on 4-5 step tasks (vs baseline 23.7%), confirming type-error is addressable.

5. **Ensembles are cost-inefficient vs BoN** — 5x cost yields only 23.0% TSR, while BoN-5 at same cost reaches 26.0%. BoN uses logprob-based oracle selection which is fundamentally stronger than consensus voting.

6. **Planning remains the #1 bottleneck** — Oracle thought reaches 35.2%, but neither training (V21) nor retrieval (V22) can effectively provide planning capability to the model.
