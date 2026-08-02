# Scale-Up Reference Audit

Date: 2026-08-02

## ZoomClick 73.1

The 73.1 ScreenSpot-Pro reference is independently verified.

- Paper: *Zoom in, Click out: Unlocking and Evaluating the Potential of Zooming for GUI Grounding*, arXiv:2512.05941v1.
- Official repository: `Princeton-AI2-Lab/ZoomClick`.
- Audited repository commit: `6a0c3633eb6985a21d99c94400bc64fe54ac5221`.
- The paper abstract and repository README both state that ZoomClick with UI-Venus-72B reaches 73.1% success rate on ScreenSpot-Pro.

This is a system result, but it is not a same-environment control. The released 72B wrapper uses a bbox grounding prompt, deterministic generation, approximately 3.11M to 48M input pixels, and ZoomClick iterative crops. The local G1 bare run follows the frozen UI-Venus model-card prompt and 2M to 4.8M pixel range. The value is therefore a verified paper-only threshold and is excluded from paired row-level calculations.

The repository's example Slurm script is labeled for UI-Venus-7B even though a separate `ui_venus_ground_72b.py` wrapper exists. This packaging inconsistency does not invalidate the paper/README statement, but it is another reason not to treat 73.1 as a local anchor.

## Qwen3.5 70.4

The 70.4 ScreenSpot-Pro model reference is independently verified in the official `Qwen/Qwen3.5-122B-A10B` model card at revision `dc4d348443bc740c68e2d77492492c11606384d5`.

The model card reports 70.4 in its Visual Agent table but does not publish a ScreenSpot-Pro grounding prompt. The local G1 run therefore uses the result-blind standardized point prompt frozen in `configs/g1_roster.yaml`. Anchor disagreement is a reproducibility observation and never triggers prompt retuning.

## Use Boundary

Both 73.1 and 70.4 are visibly marked paper-only in `MAIN_TABLE.md` and `REPORT.md`. They may define preregistered reporting thresholds, but they do not enter paired bootstrap deltas, confidence intervals, or p-values.