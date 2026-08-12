# Amendment 008: Exact TriVUS Training and Adjudication

Date: 2026-08-12

Timing: after the representation gate passed and was committed as `add6fd76bc897227c9da389ab662f26ef47f40f6`, before TriVUS fallback-context generation, unified data assembly, real-data optimizer steps, nested fits, or formal results.

## 1. Comparison boundary

TriVUS is optimized and evaluated on Mind2Web, ScreenSpot-Pro, and the paired 2,000-row AndroidControl Low/High sample. All comparisons are paired success-bit comparisons on exact row identities. Current frozen VUS-SR artifacts do not retain action objects or chosen candidate indices, so no action-semantic compatibility claim versus VUS-SR is allowed. AndroidControl claims are sample-scoped and cannot be called unbiased full-7,650-row estimates.

## 2. Locked fallback contexts

Formal workers may not load candidate-bank objects containing all-fold labels. Before training, one trusted context builder creates exactly 391,524 records containing only context key, sample key, outer fold, inner holdout or final role, fit folds, and fallback index.

- 14,644 Mind2Web/ScreenSpot-Pro samples x 21 contexts = 307,524;
- 4,000 AndroidControl samples x 21 contexts = 84,000.

For each outer fold and each of four development OOF holdouts:

1. checkpoint fold is the cyclic first candidate-training fold after the holdout;
2. the other two folds are model-training folds;
3. Mind2Web/ScreenSpot-Pro CEV policies are fit exactly as VUS-SR on the two model-training folds and applied to all four development folds;
4. Android source reliability is fit on the two model-training folds and exact-action plurality plus reliability/canonical-order tie-break is applied to all four development folds.

Final contexts fit on all four outer-development folds and apply to all five folds. Context generation may inspect labels only on each declared fit fold. It outputs no correctness, reliability, configuration score, aggregate accuracy, or target. The context file and a full split/provenance manifest are hash-locked before any formal worker starts.

## 3. Exact candidate features

Every row has a `12 x 115` float32 tensor and boolean mask. K is exactly 12 for Mind2Web/ScreenSpot-Pro and 3 for AndroidControl. Slots outside K are zero padding.

Generic dimensions 0--102:

1. canonical action one-hot, 10 dimensions: POINT, CLICK, TYPE, SELECT, OPEN, BACK, SCROLL, WAIT, LONG_PRESS, OTHER;
2. x, y, coordinate-present, parse-ok, parameter-present, clipped parameter length / 256, 6 dimensions;
3. lowercase, L2-normalized, alternate-sign-free 64-dimensional character 2--4-gram parameter hash using sklearn `HashingVectorizer`;
4. five source-agnostic set features, excluding self:
   - same canonical-action fraction;
   - mean pair kernel;
   - maximum pair kernel;
   - coordinate-neighborhood fraction at normalized distance `<0.14`;
   - mean token-F1 among same-action parameter-bearing pairs;
5. seven blind visual features restored to public candidate order:
   - centered logit;
   - log probability;
   - probability;
   - normalized rank using denominator K-1;
   - entropy divided by log K;
   - centered-logit minus fallback;
   - probability minus fallback;
6. fallback flag and K/12, 2 dimensions;
7. family one-hot Mind2Web/ScreenSpot-Pro/AndroidControl, 3 dimensions;
8. cell one-hot C-uni/C-cond/C-rand/C-self/Android-Low/Android-High, 6 dimensions.

Pair kernel is zero for different canonical actions. For equal actions it starts at one, multiplies by `exp(-d^2/(2*0.07^2))` when both coordinates exist (zero when only one exists), and multiplies by lowercased whitespace-token F1 when either parameter is nonempty.

Reserved dimensions 103--114 are zero for all variants except `RANDOM_ID_PLACEBO`. That variant assigns each valid candidate a deterministic row-hash-permuted pseudo-identity one-hot over the first K reserved dimensions. It never uses true source, model, lineage, stage, or slot identity.

Action canonicalization:

- POINT: point;
- CLICK: click, doubleclick, rightclick, moveto;
- LONG_PRESS: long_press, longpress;
- TYPE: type, input_text;
- SELECT: select;
- OPEN: open, open_app, launch_app;
- BACK: back, go_back, navigate_back, press_back;
- SCROLL: scroll, swipe;
- WAIT: wait, idle;
- OTHER: all remaining/missing actions.

## 4. Targets and weights

For fallback b and private success vector y:

- if y_b is false and one or more candidates succeed, target mass is uniform over successful candidates;
- otherwise target mass is one on KEEP;
- fallback-correct target is y_b;
- inactive rows where all candidate utilities relative to fallback are equal receive zero training weight.

JOINT3 gives total mass one to each family. Within Mind2Web/ScreenSpot-Pro, four arms split family mass equally; within AndroidControl, Low/High split family mass equally; within a cell, active rows have equal mass. TARGET_ONLY gives its one family mass one. JOINT2 gives Mind2Web and ScreenSpot-Pro mass one each.

Train-only standardization is fit over valid candidates from active weighted model-training rows. Validation/OOF/test use frozen statistics. Padding remains zero. NO_VISUAL zeros the seven visual dimensions before fitting and applying its own standardizer. Non-placebo variants zero all reserved dimensions.

## 5. Models and fit count

All models use the already frozen masked S2 architecture/loss and independent same-seed initialization:

- JOINT3: all three families;
- TARGET_ONLY-Mind2Web, TARGET_ONLY-ScreenSpot-Pro, TARGET_ONLY-AndroidControl;
- JOINT2_NO_ANDROID: Mind2Web and ScreenSpot-Pro;
- NO_VISUAL: all three families;
- RANDOM_ID_PLACEBO: all three families.

TARGET_ONLY predictions from the three models form one control policy. Per outer fold there are 7 models x (4 inner + 1 final) = 35 fits; total 175 fits.

Optimizer is AdamW, learning rate 3e-4, weight decay 1e-3, batch size 256, gradient norm 1.0, maximum 30 epochs, patience five, minimum validation improvement 1e-5, and exactly one accumulated optimizer step per epoch with full split-weight normalization. Checkpoint ties choose the earliest epoch. Final epoch is the half-up median of four selected inner epochs, separately per model.

All model specs in the same split independently initialize with the same seed `20260812 + 1000*outer_fold + 10*oof_holdout`; final seed is `20260812 + 1000*outer_fold + 999`. Epoch row order and 12-slot candidate permutations are separately SHA-256-derived from model seed, epoch, and sample key. No model-specific seed offset or warm start is allowed.

## 6. Safe thresholds

Each learned policy chooses direct candidate argmax, candidate-minus-KEEP margin, and fallback-wrong score `1-sigmoid(fallback_correct_logit)`. Threshold candidates are infinity, zero, and deciles of strictly positive changed-row values on both axes. Exact ties prefer larger fallback-wrong threshold, then larger margin threshold.

Cells are the four arms for Mind2Web, four arms for ScreenSpot-Pro, and Low/High for AndroidControl. A cell with at least 200 changed OOF opportunities selects its own threshold maximizing point delta subject to loss no worse than 0.5 cell MDE. Otherwise it uses the family-pooled threshold. Family-pooled thresholds require every cell loss at most 0.5 MDE and equal-cell mean loss at most 0.25 MDE.

MDEs are 0.006106589385659482, 0.007, and 0.01 for Mind2Web, ScreenSpot-Pro, and AndroidControl.

## 7. Nested sealing

For each outer fold, all inner training, checkpoint selection, OOF prediction, thresholds, final epochs, final model fits, feature hashes, standardizer hashes, context hash, public/blind hashes, and opened development-label hashes are atomically fsynced to `outer-k.pretest.json`. Only then may workers open the two physically sealed outer-label files for VUS and AndroidControl.

Outer-test evaluation occurs once. All variants are fixed and none may be selected from outer results.

## 8. Final baselines and statistics

Frozen baseline registry:

- Mind2Web and ScreenSpot-Pro: frozen VUS-SR safe success bits;
- AndroidControl Low/High primary baseline: R1 fold-local majority success bits;
- AndroidControl strongest loss-cap baseline: UI-AGILE success bits in both settings, fixed because its known point accuracy exceeds majority in Low and High.

Every comparison requires exact key-set equality. Use 10,000 paired episode/application-grouped resamples within frozen folds and 99% percentile intervals.

Bootstrap cell order is Mind2Web C-uni/C-cond/C-rand/C-self, ScreenSpot-Pro C-uni/C-cond/C-rand/C-self, AndroidControl Low/High. Seed is `20260900 + control_offset + cell_index`, with offsets PRIMARY=0, TARGET_ONLY=100, NO_VISUAL=200, STRONGEST=300. Cell replicate arrays at the same index are averaged for equal-cell/family and standardized three-family intervals.

Family effects average cells equally. The standardized three-family sample is:

$$
\Delta_{3F}=\frac{1}{3}\left(
\frac{\overline\Delta_{M2W}}{0.006106589385659482}+
\frac{\overline\Delta_{SSP}}{0.007}+
\frac{\overline\Delta_{AC}}{0.01}
\right),
$$

where AndroidControl averages Low and High equally.

## 9. Promotion gates

All seven original TriVUS gates are executable as:

1. every Mind2Web and ScreenSpot-Pro arm has JOINT3-minus-VUS-SR 99% CI lower bound greater than negative benchmark MDE;
2. AndroidControl Low and High each have JOINT3-minus-majority 99% CI lower bound greater than -0.01;
3. at least one family equal-cell JOINT3-minus-primary-baseline 99% CI lower bound is positive;
4. JOINT3-minus-primary-baseline three-family standardized 99% CI lower bound is positive;
5. JOINT3-minus-TARGET_ONLY three-family standardized 99% CI lower bound is positive;
6. JOINT3-minus-NO_VISUAL three-family standardized 99% CI lower bound is positive;
7. every Mind2Web/ScreenSpot-Pro arm satisfies Gate 1 and AndroidControl Low/High JOINT3-minus-UI-AGILE 99% CI lower bounds are greater than -0.01.

JOINT2_NO_ANDROID and RANDOM_ID_PLACEBO are mandatory descriptive controls. Promotion requires all seven gates. Comparisons establish paired outcome utility only, not action-semantic compatibility.

## 10. Execution boundary

Fallback-context generation and data-adapter implementation are authorized next. Real-data model fits remain prohibited until context/data manifests, all adapter tests, the complete nested runner, final adjudicator, and a separate implementation-bound authorization are committed result-free.