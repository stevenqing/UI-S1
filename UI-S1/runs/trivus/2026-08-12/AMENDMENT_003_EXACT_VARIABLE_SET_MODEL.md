# Amendment 003: Exact Variable-Set Model

Date: 2026-08-12

Timing: frozen while result-blind R0 recovery is running, before R0 completion, R1 evaluation, AndroidControl selector inference, private-label bank construction, or any TriVUS fit/result.

## Input tensor contract

All rows are padded to 12 candidate slots and carry an explicit boolean candidate mask. Valid counts are exactly 3 for AndroidControl and 12 for Mind2Web/ScreenSpot-Pro. Padding slots have zero features, zero target mass, and utility logits masked to the minimum finite dtype value. The fallback index must point to a valid candidate.

Generic per-candidate features are frozen as:

- canonical action one-hot: POINT/CLICK/TYPE/SELECT/OPEN/BACK/SCROLL/WAIT/LONG_PRESS/OTHER;
- normalized coordinate, coordinate-present, parse-ok;
- parameter-present, clipped parameter length, fixed 64-dimensional alternate-sign-free character 2--4-gram hash;
- source-agnostic candidate-set evidence: same-action support, mean/max pair kernel, coordinate-neighborhood support, parameter-similarity support;
- blind selector centered logit, log probability, probability, rank divided by $K-1$, entropy divided by $\log K$, centered-logit/probability differences from fallback;
- fallback flag and candidate-count fraction $K/12$;
- benchmark cell state: Mind2Web, ScreenSpot-Pro, AndroidControl; four existing arms or Android Low/High.

No instruction hash enters the small set model. The blind VLM channel already conditions on instruction and screenshot, and CIVA-A0 did not establish an independent hashed-text contribution. Source/model/slot identity remains prohibited.

## Architecture

- maximum candidates: 12;
- shared candidate MLP: input to width 64, GELU, LayerNorm;
- two Transformer encoder layers, four heads, feed-forward width 256, dropout 0.1, pre-norm;
- no candidate/source/slot positional embeddings;
- padding mask passed to every attention layer;
- KEEP representation equals encoded fallback candidate before set attention plus one learned width-64 delta;
- shared scalar utility head over 12 candidates plus KEEP;
- fallback-correct scalar head from the encoded KEEP token.

## Objective and optimizer

The only primary objective is the empirically selected VUS-SR S2 objective:

- soft listwise repair-or-KEEP cross entropy, weight 1.0;
- fallback-correct BCE, weight 0.5.

No U-GRPO, consistency penalty, channel gate, expert loss, or post-result loss search is allowed. Candidate permutation equivariance is architectural and is checked exactly in tests.

Optimizer is AdamW, learning rate `3e-4`, weight decay `1e-3`, batch size 256, gradient norm 1.0, maximum 30 epochs, patience five, and minimum validation improvement `1e-5`. One accumulated optimizer step is taken per epoch using exact split-weight normalization. Final epoch is the half-up median of four inner selected epochs.

The result-free implementation frozen with this amendment covers only the architecture, masked forward pass, permutation transforms, S2 loss, and synthetic contract tests. It does not claim to implement optimizer accumulation, checkpoint selection, nested data assembly, or outer execution. Those components remain unauthorized and must be frozen in a later amendment after R1 passes.

## Variants

Every learned variant has identical architecture and independent same-seed initialization:

- `JOINT3`: all three benchmark families, primary;
- `TARGET_ONLY`: one separately trained model per benchmark family;
- `JOINT2_NO_ANDROID`: Mind2Web plus ScreenSpot-Pro only;
- `NO_VISUAL`: selector dimensions zeroed before train and test;
- `RANDOM_ID_PLACEBO`: adds a row-hash-permuted three-way pseudo-source feature to AndroidControl and row-hash-permuted 12-way pseudo-slot feature to other benchmarks. It never uses true identity.

## Selection boundary

Thresholds use candidate-minus-KEEP margin and fallback-wrong score, with infinity, zero, and positive deciles. They are selected per benchmark cell under the already frozen loss constraints. No variant, feature, epoch, or threshold can be selected from outer-test results.

The model implementation and synthetic tests may be committed before R1. Data adapters, blind AndroidControl inference, exact training runner, and formal execution remain unauthorized until their prior stage gates pass and are separately frozen.