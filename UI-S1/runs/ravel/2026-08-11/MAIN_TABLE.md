# RAVEL E0 Main Table

Status: `STOPPED_BY_RAVEL_K4`

## Evidence diagnostics

| Equal-arm metric | Mind2Web VUS | Mind2Web local | Delta | ScreenSpot VUS | ScreenSpot local | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Utility-positive AUROC | 0.640 | 0.597 | **−0.043** | 0.560 | 0.575 | +0.015 |
| Direct accuracy | 30.49% | 28.51% | −1.98 pp | 45.21% | 45.76% | +0.55 pp |
| Unique-correct recall | 41.14% | 27.49% | **−13.66 pp** | 22.87% | 21.26% | −1.62 pp |
| Small-target recall given coverage | 31.95% | 34.52% | +2.57 pp | 33.93% | 33.96% | +0.03 pp |

Local minus random-center AUROC is +0.0467 on Mind2Web and +0.0463 on ScreenSpot-Pro. Candidate-centered crops contain real signal, but the fixed-token early-fusion prompt loses global/identity evidence.

## Nested safe Step-SR

| Local-evidence VUS-SR − frozen VUS-SR | Point delta | 99% paired CI |
| --- | ---: | ---: |
| Mind2Web C-uni | **−2.31 pp** | **[−3.88,−0.73]** |
| Mind2Web C-cond | −1.15 pp | [−2.65,+0.39] |
| Mind2Web C-rand | **−3.03 pp** | **[−4.64,−1.51]** |
| Mind2Web C-self | **−2.26 pp** | **[−3.87,−0.73]** |
| Mind2Web equal-arm | **−2.19 pp** | **[−2.98,−1.41]** |
| ScreenSpot-Pro equal-arm | −0.03 pp | [−0.24,+0.16] |

RAVEL E0 fails both frozen routes. `RAVEL-K4` triggers. Relational training, lower-bound calibration, random safe-step training, and full VLM LoRA are cancelled.
