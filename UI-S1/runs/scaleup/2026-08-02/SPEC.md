# Scale-Up Gate

Date: 2026-08-02

Status: result-blind preregistration. This run tests whether the frozen cross-lineage allocation result scales to released 72B-class systems and closes the remaining 7B reporting gaps. It introduces no new method.

Upstream:

- `runs/ccm-h2h/2026-07-31/`
- `runs/allocation-law/2026-08-01/`
- `runs/diversity-axis/2026-08-02/`
- `runs/closing/2026-08-02/`

## G1 lineage gate

Run one deterministic full-image prediction for each of the three models frozen in `configs/g1_roster.yaml` on the same 1,581 ScreenSpot-Pro identities. Report local bare accuracy, absolute difference from the paper-only reference, the three pairwise failure Cohen kappas, 1,000 matched-marginal permutations, and pass@3.

G1 passes when at least one cross-model kappa is below 0.45 and pass@3 is at least 0.78. G2 is cancelled when pass@3 is below 0.75. The interval [0.75, 0.78) is a marginal gate: G2 may run, but G1 is not called passed. If every pairwise kappa is at least 0.55, the roster is classified as lineage-concentrated and 70.4 becomes the effective G2 threshold while 73.1 remains the stretch threshold. No prompt, parser, checkpoint, or generation parameter is changed to improve anchor agreement.

## G2 mixed 72B pool

Use GTA1-72B as the sole attention proposer. Score the frozen full image plus three GTA1 attention crops with each of GTA1-72B, UI-Venus-Ground-72B, and Qwen3.5-122B-A10B. The primary P2 pool therefore contains 12 candidates and 18,972 scoring forwards. Report unchanged B3 and fold-local M1.

P1 is a same-budget GTA1-72B control. It uses 12 GTA1 candidates when available and may use the preregistered 8-forward fallback only if a pre-inference resource check makes N=12 infeasible. Paper-only values 73.1 and 70.4 are context thresholds and never enter a paired row-level difference or significance calculation.

Proposal sensitivity follows H1: three label-free Gumbel perturbation seeds select three crops from the official top-18 GTA1 proposals. Scores are generated once per unique region in the cross-seed union and reused without alteration. MDE is twice the sample standard deviation of the three perturbed P2 M1 accuracies. This diagnostic costs additional scoring forwards beyond the 18,972-forward primary pool; actual unique-region accounting must be reported.

## Z1-Z5 closing work

Z1 uses 10,000 fold-stratified application-group paired bootstrap replicates and seed 20260802. Z2 moves the degenerate N=2 H1 column to appendix scope. Z3 assigns the budget-decline claim only to Allocation-Law L1 N=4 through N=16 plus the X3 slope CI. Z4 uses only v1-only MDE values in the main text. Z5 reuses the already completed 16-sample GTA1 Closing run and recomputes the frozen GUI-RC/B3 sampling curves; no duplicate inference is launched.

## Writing boundaries

- No absolute open-source SOTA claim is made unless G2 clears the applicable frozen threshold and MDE rule.
- UI-Zoomer K3/K8 results do not enter G1, G2, or Z1-Z5. Existing partial K8 traces are archived and no Scale-Up GPU time is assigned to them.
- Checkpoints, raw traces, images, and logs remain untracked.
- PID 1814 is never killed, paused, or modified.
