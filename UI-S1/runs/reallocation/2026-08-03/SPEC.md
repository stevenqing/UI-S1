# Difficulty-Conditioned Reallocation

Date: 2026-08-03

Status: result-blind preregistration. No R1 stratified final-accuracy curve, R2 reallocation result, R3 conditional-proposal result, R4 risk-coverage result, or R5 pollution result existed when this protocol was frozen.

Upstream:

- `runs/neff/2026-08-03/`
- `runs/cala/2026-08-03/`
- `runs/diversity-axis/2026-08-02/`
- `runs/allocation-law/2026-08-01/`

## R1: stratified accuracy gate

Compute SafeGround official-code transfer uncertainty from the complete Uniform Mixed N12 pool. The score uses patch size 28 and activation threshold 0.0; higher means more disagreement. Sort by `(uncertainty,row_id)` and split into five NumPy `array_split` bins.

Within each frozen bin, report unchanged B3, fold-local M1 and pass@N for Uniform Mixed prefixes N=4/8/12/16/24. The highest-disagreement quintile is the only gate stratum.

R1 passes when highest-quintile B3 N24-minus-N4 exceeds 0.007043345177520599 and its 10,000-replicate fold-stratified application-group paired-bootstrap 99% CI lower bound is positive. Report pass@N and `B3_delta / pass_delta` as the headroom realization ratio when pass delta is positive.

If R1 fails, R2 and R3 are cancelled. No alternate uncertainty score, binning or endpoint is tested.

## R2: budget-conserving reallocation

R2 runs only if R1 passes. Every fold learns uncertainty cut points on the other four application folds and applies them to the held-out fold. Uncertainty itself is label-free; thresholds and budget maps are frozen in `configs/r2_policies.yaml`.

Policies:

- U4/U8/U12: fixed Uniform Mixed prefixes.
- S1: quintile budget vector `[4,4,4,4,24]`, target mean 8.
- S2: decile budget vector `[4,4,4,12,12,12,12,12,24,24]`, target mean 12.
- S3-8: decile budget vector `[4,4,4,4,4,4,8,8,16,24]`, target mean 8.
- S3-12: decile budget vector `[4,4,4,8,8,12,16,16,24,24]`, target mean 12.
- S4: for each S policy and outer fold, deterministically permute that policy's exact held-out budget multiset across held-out row identities. This preserves the realized budget histogram and mean exactly while breaking association with disagreement.

Fixed sequence prefixes are the existing Uniform Mixed view-major/model-minor sequence. No candidate duplication or padding is allowed.

Primary success: a target-12 signal policy B3 exceeds U12 and has paired 99% CI lower bound above zero. Secondary efficiency success: a target-8 signal policy has mean forwards at most 8.05 and B3 at least U12 minus the frozen 0.70 pp MDE.

R-K2 triggers when every successful signal-policy claim is matched by its S4 control without a positive paired 99% CI for signal-minus-random.

## R3: conditional second-round proposals

R3 is inference-eligible only if R1 passes. Inference is launched only after R2 is complete and a separate result-blind runtime amendment freezes crop size, clustering, random-crop distribution, model prompts and checkpoint revisions.

The target subset is the R1 highest-disagreement quintile. First-round actions are three lineages on shared views 0 and 1. C-uni uses Uniform Mixed actions 7-12; C-cond uses two candidate-cloud cluster-conditioned crops scored by all three lineages; C-rand uses two seed-frozen random crops scored by all three lineages. All cells total 12 model forwards.

C-cond must exceed both C-uni and C-rand by more than MDE with positive paired 99% CI lower bounds. Otherwise R-K3 triggers.

## R4: selective accuracy

Independently compute risk-coverage for Mixed N12 and V-only N12 using each pool's own SafeGround official-code transfer uncertainty and unchanged B3 correctness. Retain the least-uncertain 90%, 80% and 70% of rows using `(uncertainty,row_id)` ordering. Report retained accuracy, rejected failures/successes and selective-risk reduction from full coverage.

Random rejection is mandatory: 10,000 seed-20260803 permutations at each retained coverage, preserving the exact retained row count. Report random mean and 99% interval.

R4 is a reporting result, not an allocation claim. It remains valid regardless of R1-R3.

## R5: 72B candidate pollution

Use the completed and N3-audited 72B N8 traces. Compare 7B and 72B Uniform Mixed N8:

- pairwise distances among failed candidates, normalized by image diagonal;
- largest 14-pixel failed-candidate cluster fraction;
- model composition of the B3-selected cluster;
- B3-selected cluster correctness.

The pollution hypothesis is supported only if 72B failed candidates are tighter than 7B in point estimate and the B3-selected wrong cluster is nonuniform across models. This is descriptive and does not repair 72B values.

## Statistics and boundaries

- 10,000 paired fold-stratified application-group bootstrap replicates.
- Seed 20260803.
- 99% percentile intervals and plus-one one-sided p-values.
- Existing selectors and candidate traces remain unchanged.
- Paper-only numbers never enter local paired differences.
- S4, C-rand and random rejection are mandatory.
- No R3 inference when R1 fails.
