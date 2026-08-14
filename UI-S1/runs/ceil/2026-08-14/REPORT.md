# CEIL Closure Diagnostic Report

Date: 2026-08-14

Status: `CEIL_COMPLETE_OPEN_NEW_SPEC_C_D2`

## Scope

CEIL is a zero-GPU closure analysis. It does not change F1, Q1, `TRIVUS_NOT_PROMOTED`, VUS-SR, SPLIT, or MASK. Arm A is post-hoc descriptive. Arm B is evaluation-side and cannot define a runtime selector.

## Arm B: Conditional ranking signal

Mind2Web has 2,021 recoverable samples. Cheap-ranker candidate AUROC is **0.688**, with 99% CI **[0.665, 0.709]**. The lower bound exceeds 0.65, so Mind2Web triggers **C-D2**. Blind visual AUROC is 0.585; the cheap ranker contains conditional signal beyond the frozen visual ordering, although its recoverable-subset top-1 rate is only 22.0%.

ScreenSpot-Pro has 968 recoverable samples. Cheap-ranker AUROC is **0.540**, with 99% CI **[0.501, 0.583]**. The upper bound is below 0.60, so ScreenSpot-Pro triggers **C-D1**. Its conditional cheap signal is effectively absent under the frozen threshold.

The benchmark split is therefore substantive: CEIL does not support a shared candidate-reweighting conclusion. Because one eligible benchmark triggers C-D2, the overall branch is `OPEN_NEW_SPEC_C_D2`. This authorizes only a new preregistration for Mind2Web full-candidate reweighting; no experiment is authorized inside CEIL.

## Arm A: Post-hoc effective-vote ceilings

Arm A enumerates all 4,095 nonempty subsets in five independent panels. It strictly reuses MASK generalized $N_{\mathrm{eff}}$, reports the observed support separately from the full pool, and obtains $\Delta_\infty$ only from the frozen monotone saturating family. Full numerical results and 99% grouped-bootstrap intervals are in `MAIN_TABLE.md`; curves are in `ARM_A_CURVES.pdf`.

These values remain post-hoc and benchmark/arm-specific. They do not restore a universal one-dimensional effective-sample-size law. Isotonic extrapolation is limited to the finite ideal-three-vote target and is not interpreted as an infinite ceiling.

The Mind2Web parametric asymptotes are weakly identified: their $\Delta_\infty$ values range from roughly +22 to +73 pp and extrapolate far beyond observed support, with several fits approaching the bounded accuracy ceiling. They are sensitivity outputs, not precise recoverable headroom. By contrast, the finite ideal-three-vote isotonic gains range from about -0.13 to +3.32 pp across Mind2Web panels. ScreenSpot-Pro parametric $\Delta_\infty$ remains near zero with intervals crossing zero.

## Conclusion

> Conditional candidate-ranking signal survives on Mind2Web but not ScreenSpot-Pro. The current consensus-geometry sequence closes for ScreenSpot-Pro; Mind2Web alone meets the preregistered threshold for a separately preregistered full-candidate reweighting study.

No current status changes, method promotion, runtime rule, or within-round rescue is allowed.
