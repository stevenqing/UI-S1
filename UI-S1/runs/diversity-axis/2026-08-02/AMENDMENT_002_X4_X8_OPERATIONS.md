# Amendment 002: X4-X8 Zero-GPU Operations

Date: 2026-08-02

Status: frozen after X1/X3 and before any X4-X8 result in this workdir. X2 remains blocked by its useful-forward accounting gate.

## X4: GMS availability

The Scanner+Locator baseline is GMS, arXiv:2509.24133v1. The paper specifies a five-stage adaptive pipeline with recursive 3x3 search, top-k region selection, Locator verification, Scanner consensus, and adaptive-resolution fusion. It does not publish executable code, checkpoints, per-row traces, or a fixed per-row forward-count mapping. GitHub repository searches by exact title and method name returned no implementation.

The paper's reported ScreenSpot-Pro values are reference-only because they do not share the frozen 12-forward contract. X4 is `UNAVAILABLE_NO_RELEASED_IMPLEMENTATION_OR_FIXED_BUDGET_TRACE`; X-K4 is `NOT_EVALUATED`. A paper-table comparison cannot pass or trigger X4.

## X5: allocation topology

Pure parallel Q3 exists. Pure serial and three-lineage serial-four-step pools require the blocked X2 adaptive implementation and do not exist. X5 is `BLOCKED_ON_X2`; no partial triangle winner is declared.

## X6: unlabeled pool ranking

The eight L2 pools are development pools, not held-out validation. The preregistered held-out pools must come from X2/X5, which do not yet exist. X6 may inventory the eight development observations but must not report their in-sample correlation as held-out performance. Status is `BLOCKED_NO_HELDOUT_POOLS`; the Spearman > 0.7 criterion is `NOT_EVALUATED`.

## X7: SafeGround migration diagnostic

Official source is `UCSB-AI/SAFEGROUND` commit `5e8fca7ef091bc6751cad9703ca430e775aa4433`. Relevant blobs are `uncertainty.py` `fd6a37d016d9a4da89d5b7dd2efd2c18aefc9785`, `heatmap.py` `967a8ff68098faab2c29ce63d1a2f11367d50448`, `regions.py` `017ab9d2f6bf496acac7c5ff1cccf8962681e773`, and `combined.py` `cbfe1bc821d3db989bb4de53ba27ab9f22d2405e`.

The executable repository and paper differ. The repository defaults to 28-pixel patches and activation threshold 0, while paper v1 describes 14-pixel patches and beta 0.3. Both are reported. `official_code` (28, 0) is primary because it is executable and pinned; `paper_v1` (14, 0.3) is secondary. Both use strict `P > beta * P_max`, 4-connectivity, average component density, and combined weights margin 0.2, entropy 0.2, concentration 0.6.

Three source pools are evaluated:

- `stochastic_GTA1_N4`: the first four of the existing five temperature-0.7 samples. This is below SafeGround's K=10, temperature-1.0 protocol and is an algorithm-level diagnostic only. GUI-RC and B3 correctness are reported.
- `v_only_zoom_N12`: GTA1 deterministic attention-view candidates. M1 correctness is primary and B3 correctness secondary.
- `mixed_lineage_N12`: the frozen three-lineage deterministic pool. M1 correctness is primary and B3 correctness secondary.

For each label, AUROC treats failure as the positive class and uncertainty as the score. For comparison with the existing S_gap correctness AUROC, also report correctness AUROC using negative uncertainty. The comparison anchor is 0.39310293492742826 from AndroidControl Low discovery A5d-risk and is explicitly cross-task; it is not a significance threshold.

The N12 candidate pools are not stochastic samples, so no SafeGround FDR guarantee or official-reproduction claim is made. Official GTA1 K=10 AUROC 0.6344 is a paper reference only and is not used as an anchor.

## X8: Mind2Web lineage-only alternative

Use the frozen Collision-Law deployability band on 2,080 aligned full-view rows. The six deployable models split into TongUI 3, UI-TARS 2, and CogAgent 1. The requested equal-budget N=6 same-lineage pool does not exist. X8 is `BLOCKED_NO_SAME_LINEAGE_N6`; original L3 and L-K4 remain unchanged. No model duplication, non-deployable model, or view reuse is allowed.
