# Amendment 002: H1 MDE Proposal Perturbation

Date: 2026-07-31

Status: frozen before any H1 candidate generation or result.

Official MVP attention and greedy coordinate generation are deterministic. The three MDE seeds therefore perturb only the order in which the already generated, GT-free attention regions enter the N=10 candidate set.

For each row, request the official top 18 unique ranked regions. Retain each region's integer attention-coverage count `q_i`. For seed `s`, draw independent Gumbel noise using NumPy `PCG64(SeedSequence([s, stable_row_index]))` and rank regions by

`log(q_i + 1) + 0.25 * Gumbel_i`,

descending, with original official rank as the deterministic tie break. Select the first nine regions, preserving the full-image prediction as candidate zero. Coordinates remain greedy and all official crop/resize/prompt/model semantics are unchanged. Seeds are 20260731, 20260732, and 20260733.

This perturbation reads no target bbox, success label, or evaluator field. The target bbox is joined only after candidate generation for scoring. MDE is twice the sample standard deviation of M1 N=10 held-out accuracy across the three seeds.

The main H1 N=2/4/10 rows do not use this perturbation and pass `max_inferences=N-1` directly to the official deterministic proposal generator.