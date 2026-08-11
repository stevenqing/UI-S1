# Amendment 001: Blind Visual Eligibility Anchor

Date: 2026-08-11

Timing: frozen after data-pipeline smoke validation and before any Qwen3-VL VUS prediction or label-based VUS analysis.

The full nested LoRA grid in the main specification is expensive. Before training, run one blind zero-shot eligibility anchor with the unchanged retained Qwen3-VL-8B-Instruct model.

1. Inference reads only `public_records.jsonl`. It cannot open `private_labels.jsonl`, CEV result labels, target boxes, positive DOM candidates, or evaluator outputs.
2. The prompt uses the original task, compact prior actions, dynamically rendered candidate overlay, candidate legend, and exact fold-local CEV-A fallback index. The fallback index is computed from frozen upstream CEV policies without reading row correctness.
3. Candidate order uses the preregistered deterministic epoch-0 permutation.
4. Qwen3-VL emits one-step logits for the single-token labels A--M. A--L map to real candidates and M maps to `KEEP_CEV`.
5. Eight independent shards run on GPU 0--7. Protected PID 2274 remains untouched.
6. Only after all blind logits are retained may the anchor adjudicator read private labels. For each outer fold it selects benchmark/arm safe thresholds on the other four folds and evaluates the held-out fold once.
7. The direct candidate is the largest A--L probability, regardless of whether M is the raw largest label. Candidate-vs-KEEP margin is `p(direct) - p(M)`. The fallback display label and M denote the same deployed fallback, so fallback-wrong score is exactly `max(0, 1 - p(M) - p(fallback_display_label))`.
8. For every outer fold, benchmark-pooled thresholds are selected first on the other four folds by maximizing equal-arm Step-SR delta, subject to every arm losing at most 0.5 MDE and the equal-arm mean losing at most 0.25 MDE. A cell with at least 200 changed-candidate opportunities then selects its own threshold by maximizing cell Step-SR delta subject to at most 0.5 MDE loss; lower-support cells use the benchmark threshold exactly. Threshold candidates on each axis are infinity, zero, and deciles of strictly positive observed values. Ties prefer the larger fallback-wrong threshold and then the larger margin threshold.

Eligibility to start LoRA is satisfied when either condition holds:

- `A1`: at least one benchmark has positive held-out equal-arm safe delta over CEV-A, no cell loses one MDE, and the equal-benchmark standardized point effect is positive; or
- `A2`: utility-positive versus non-positive candidate ranking AUROC is at least 0.55 on both benchmarks.

The anchor is not the VUS main result and cannot satisfy V1--V5. Failure closes the expensive LoRA branch rather than tuning the prompt or anchor thresholds after labels are viewed.
