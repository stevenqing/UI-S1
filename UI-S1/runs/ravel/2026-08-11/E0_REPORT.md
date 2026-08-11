# RAVEL E0 Token-Matched Local Evidence Report

Date: 2026-08-11

Outcome: `STOPPED_BY_RAVEL_K4`

## 1. Protocol and retention anchors

Each row-arm used one Qwen3-VL invocation. The main input allocated the frozen visual budget to an unmarked global screenshot plus fine/context candidate mosaics. Actual local pixel ratio versus VUS was 0.921--0.967, mean 0.933; mean visual tokens were 833 versus 882 for global-only.

Five modes were completed and blind-locked before labels were opened:

| Mode | Predictions SHA-256 |
| --- | --- |
| local | `467ce1f2d6e6c57bf2679d4f85d8233d5aca3ec21e53e43cdb68cb7b92a3e8f1` |
| random centers | `03f7fb254d9d3e2a64b47ffba0a66b321706a9457f22466a1ddc0de7e1f9efc8` |
| global only | `93541d1d1381a627ff1b5dfc6d75a170efb141bd9509fd0254b0ea5d7805feaf` |
| fine only | `102c40dfe05bf858f86001ab6726f4b9c9578b0f10214507ed644bf7553c73ec` |
| context only | `49ee984916cd7cc9a042b4279241d41be2c811e5eee9c8a5cd3f4dc01fbd3979` |

All modes cover 14,644 records, use the same model/public hashes, and record `private_labels_opened=false`.

## 2. Representation result

Local evidence beats token-identical random centers in utility-positive AUROC by +0.0467 on Mind2Web and +0.0463 on ScreenSpot-Pro, validating that candidate-centered pixels contain information.

It does not beat frozen VUS evidence:

- Mind2Web AUROC: 0.640→0.597;
- ScreenSpot AUROC: 0.560→0.575, below the +0.03 gate;
- Mind2Web unique-correct recall: 41.14%→27.49%;
- ScreenSpot unique-correct recall: 22.87%→21.26%.

Small-target Mind2Web recall improves by +2.57 pp, but the representation loses substantially more global/unique-candidate information.

Descriptive controls expose evidence competition:

- global-only direct accuracy: 32.33% / 51.66%;
- fine-only: 17.67% / 44.34%;
- context-only: 21.98% / 40.42%;
- early-fused local: 28.51% / 45.76%.

No local scale replaces global semantics, and early fusion inside one fixed-token prompt does not preserve each channel's useful information.

## 3. Final utility result

The unchanged VUS-SR architecture and nested protocol were retrained with locked local logits. Fold selections were S2/S1/S2/S2/S2, all at 30 epochs. Relative to frozen VUS-SR:

- Mind2Web equal-arm: −2.19 pp, 99% CI [−2.98,−1.41];
- ScreenSpot-Pro equal-arm: −0.03 pp, [−0.24,+0.16].

Mind2Web C-uni, C-rand, and C-self are individually significantly worse. Structure/downside training cannot recover the information lost by the early-fusion query.

## 4. Decision

Both E0 routes fail. `RAVEL-K4` triggers. The following are cancelled:

- relational REPAIR/SAME/BREAK training;
- lower-bound safety calibration;
- random-center safe-step training;
- full Qwen3-VL LoRA;
- third-benchmark confirmation for RAVEL.

The supported mechanism is narrower: local pixels are informative, but fixed-budget early fusion causes destructive evidence competition. Late fusion of independently locked evidence channels requires a new post-RAVEL protocol and cannot rescue this study.
