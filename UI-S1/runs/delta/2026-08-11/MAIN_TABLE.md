# DELTA Main Table

Status: `DELTA_NOT_SUPPORTED`

## FULL versus frozen VUS-SR

| Equal-arm Step-SR | FULL | VUS-SR | Delta | 99% paired CI |
| --- | ---: | ---: | ---: | ---: |
| Mind2Web | 34.51% | 34.92% | -0.41 pp | `[-1.20,+0.40]` |
| ScreenSpot-Pro | 64.37% | 64.26% | +0.11 pp | `[-0.20,+0.41]` |

## Mandatory capacity and evidence controls

| FULL minus control | Balanced point (MDE units) | 99% paired CI | Interpretation |
| --- | ---: | ---: | --- |
| VUS_ONLY | -0.523 | `[-1.155,+0.101]` | no same-capacity gain |
| RANDOM_PLACEBO | -0.200 | `[-0.772,+0.367]` | no real-channel gain |
| VUS_GLOBAL | **-0.671** | **`[-1.257,-0.074]`** | fine/context addition is harmful overall |
| VUS_LOCAL | **+0.771** | **`[+0.304,+1.238]`** | global/binding evidence is necessary |
| FIXED_AVERAGE | **+1.252** | **`[+0.486,+1.997]`** | learned fusion beats naive averaging |

DELTA-1/3/4/5 fail; DELTA-2/6 pass. Distillation and third-benchmark confirmation are not run.