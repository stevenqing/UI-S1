# Amendment 010: Unified TriVUS Data Adapter

Date: 2026-08-12

Timing: after exact fallback-context seal commit `9e5b22bc92d0ba6f75200fd21d54ec9f01dbc346`, before unified real-data assembly, optimizer steps, checkpoint selection, thresholds, outer labels, or model results.

## 1. Exact feature layout

Every row is a `12 x 115` float32 tensor plus boolean candidate mask. Valid K is exactly 3 or 12. Padding is zero.

- `0:10`: canonical action one-hot;
- `10:16`: x, y, coordinate-present, parse-ok, parameter-present, clipped parameter length / 256;
- `16:80`: lowercase alternate-sign-free L2-normalized character 2--4-gram HashingVectorizer output;
- `80:85`: source-free set features;
- `85:92`: blind visual features;
- `92:94`: fallback flag and K/12;
- `94:97`: family one-hot;
- `97:103`: cell one-hot;
- `103:115`: reserved placebo dimensions.

For each candidate, set features use exactly the K-1 nonself peers:

1. same canonical-action fraction;
2. mean pair kernel;
3. maximum pair kernel;
4. fraction with both coordinates present and normalized Euclidean distance strictly below 0.14;
5. mean lowercase whitespace-token F1 over peers with the same canonical action and a nonempty parameter on both candidates; zero if none are eligible.

Pair-kernel mean/max include all nonself peers. Different actions have kernel zero. Equal actions start at one, multiply by `exp(-d^2/(2*0.07^2))` when both coordinates exist, become zero when exactly one coordinate exists, and multiply by token F1 when either parameter is nonempty.

Visual features in public candidate order are centered logit, clipped log probability, probability, stable descending normalized rank, entropy divided by `log(K)`, centered-logit minus fallback, and probability minus fallback.

## 2. Variants

`NO_VISUAL` zeros `85:92` before standardization. All variants except `RANDOM_ID_PLACEBO` zero `103:115`. The placebo computes a SHA-256-derived permutation from the full fallback-context key and assigns one pseudo-identity one-hot per valid candidate in the first K reserved dimensions. Distinct fallback contexts for the same sample therefore receive independently derived permutations. It uses no true source, model, lineage, stage, or slot identity.

## 3. Targets and activity

If fallback is wrong and at least one candidate succeeds, target mass is uniform over all successful candidates. Otherwise target mass is one on KEEP. Fallback-correct is the fallback success bit. A row is active iff at least one candidate success differs from fallback success. Inactive rows have zero training weight but remain evaluable.

## 4. Weighting

Weights are assigned after selecting a model's included families. Each included family has total active-row mass one. Within Mind2Web and ScreenSpot-Pro, each nonempty arm has one quarter family mass. Within AndroidControl, Low and High each have one half. Active rows within a cell divide cell mass equally. Missing required cells are an error. Total weight equals the number of included families.

## 5. Standardization

Variant masking occurs before fitting statistics. The standardizer receives only the explicit model-training subset and uses only active valid candidates with positive row weight. Each dimension uses population standard deviation; scale below `1e-6` becomes one. Applying frozen statistics to checkpoint/OOF/test rows rezeros padding. No validation, OOF, or outer row contributes to statistics.

Every raw `TriVUSData` object must pass a unified validator before standardization or model conversion. It enforces shapes, finite values, prefix-valid K=3/12 masks, zero padding, valid fallback, target simplex, no target mass on padding, target/label/fallback/activity equivalence, binary labels and fallback targets, family/cell legality, fold range, metadata lengths, unique context keys, inactive-row zero weights, and optional exact included-family weights. Candidate permutation occurs only after checked conversion to a model batch; raw adapter masks remain prefix-valid.

## 6. Execution boundary

The first implementation is limited to pure feature/target/weight/standardization primitives and synthetic tests. It may inspect public schemas but may not open private label folds or perform optimizer steps. A separate result-free commit is required before a one-development-split real-data smoke. Formal nested training remains unauthorized.