# Aggregator Closure and Native-Prompt Spec

Date: 2026-08-08

Status: result-blind execution freeze.

Upstream: `runs/xfer/2026-08-07/`. Mind2Web XF1/XF2 pass under the frozen unified product-action prompt. C-cond exceeds C-uni by 4.90 pp, but same-pool majority reaches 32.31% versus 31.59% for the sequential complete-link aggregator. Absolute full-image scores are roughly 20 pp below historical native-adapter results.

## Order

1. E1 four-arm by aggregator matrix, zero GPU.
2. E3 containment mechanism, zero GPU in parallel.
3. E2 native-prompt anchor and four-arm rerun only if E1 passes.
4. AndroidControl resumes only if E1 and E2 pass.

## E1

Evaluate C-uni, C-cond, C-rand, and C-self under majority, A0, ours, A1, A2, A3, and A4 on the exact retained candidate traces. The primary comparison is C-cond-majority minus C-uni-majority. C-rand and C-self are mandatory controls. Use 10,000 grouped 99% bootstrap replicates and benchmark-specific MDEs.

## E2

Freeze native adapters and strict format-only parsers before inference. Run full-image anchors first. Continue the native four-arm rerun only when all three anchors fall within 2 pp of TongUI-7B 52.9%, CogAgent-18B 50.1%, and UI-TARS-7B 33.7%.

## E3

Compare proposer containment-by-rank with V-only budget curves on ScreenSpot-Pro and Mind2Web. This is a two-benchmark qualitative mechanism diagnostic, not a fitted law.

## Kill Conditions

- E-K1: C-cond-majority does not exceed C-uni-majority beyond MDE with positive 99% CI. Cancel E2 and AndroidControl; restrict the claim to aggregator-specific candidate improvement.
- E-K2: any native single-model anchor misses by more than 2 pp. E2 becomes diagnostic and SOTA remains open.
- E-K3: anchors pass but native C-cond does not beat native C-uni. Retract Mind2Web transfer as a low-baseline effect.
- E-K4: containment does not qualitatively explain curve-sign differences. Keep XF4 unsupported and rank-decay benchmark-specific.

## Retention

All new JSONL traces are flushed and fsynced per row, SHA-256 recorded after each lane, copied to the independent blobfuse root in `configs/retention.yaml`, and never recursively deleted.
