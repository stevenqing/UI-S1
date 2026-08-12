# CIVA-A0 Main Table

Status: `CIVA_ADMISSION_NOT_SUPPORTED`

## REAL_FULL versus raw VUS-binding direct

| Equal-arm accuracy | Baseline | REAL_FULL | Delta | 99% paired CI |
| --- | ---: | ---: | ---: | ---: |
| Mind2Web | 30.49% | **32.07%** | **+1.57 pp** | **`[+0.79,+2.39]`** |
| ScreenSpot-Pro | 45.21% | **50.62%** | **+5.41 pp** | **`[+4.12,+6.81]`** |

## Mandatory controls

| REAL_FULL minus control | Balanced point (MDE units) | 99% paired CI | Gate |
| --- | ---: | ---: | --- |
| baseline | +5.163 | `[+4.020,+6.334]` | PASS |
| matched random | +6.891 | `[+5.696,+8.036]` | PASS |
| random-channel placebo | +3.253 | `[+2.134,+4.409]` | PASS |
| no text | -0.428 | `[-1.161,+0.305]` | **FAIL** |
| text only | +2.092 | `[+1.203,+2.989]` | descriptive |

CIVA-1/2/3/4 pass; CIVA-5/6 fail. A0 is not promoted and no policy-level follow-up is run.