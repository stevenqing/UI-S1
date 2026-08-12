# TriVUS AndroidControl Representation Gate Report

Date: 2026-08-12

Outcome: `PROCEED_TO_TRIVUS_TRAINING_IMPLEMENTATION`

## Policy accuracy

| Setting | Blind direct | Fold-local majority | Hash-random |
| --- | ---: | ---: | ---: |
| Low | 74.15% | 78.20% | 62.40% |
| High | 57.15% | 59.90% | 44.05% |

Blind direct is not a replacement for the frozen fallback:

- Low direct minus majority: -4.05 pp, 99% CI `[-5.99,-2.17]`;
- High direct minus majority: -2.75 pp, 99% CI `[-4.93,-0.50]`.

It is strongly better than a label-independent hash-random candidate:

- Low: +11.75 pp, 99% CI `[+9.04,+14.45]`;
- High: +13.10 pp, 99% CI `[+10.25,+15.96]`.

Fallback-relative repair ranking AUROC is:

- Low: 0.6125 over 6,000 candidate instances, 132 positives;
- High: 0.6083 over 6,000 candidate instances, 267 positives.

## Frozen gates

| Gate | Result | Decision |
| --- | --- | --- |
| RG-S: direct beats hash-random in both settings | both 99% lower bounds positive | PASS |
| RG-A1: direct safely beats majority | direct is significantly worse in both | FAIL |
| RG-A2: repair AUROC >= 0.55 in both | 0.6125 / 0.6083 | PASS |

The proceed rule is `RG-S AND (RG-A1 OR RG-A2)`, so training implementation is authorized.

## Interpretation

The result supports the exact TriVUS mechanism: Qwen probabilities contain useful information about which candidate can repair a failed majority fallback, but raw Qwen argmax discards the strong majority prior and is harmful. The next model must therefore preserve majority/CEV fallback and learn listwise repair-or-KEEP utility plus fallback downside. It must not deploy blind direct.

The complete result was independently recomputed and matched the saved JSON exactly. `REPRESENTATION_GATE.json` SHA-256 is `c1bbf8648e25388594b43cd4acec044a393b94e23bd867176ba68b71e7ee0fc5`.

AndroidControl claims remain scoped to the paired 2,000-row Low/High sample. This gate does not establish full-7,650-row performance, action-semantic compatibility with frozen VUS-SR, or a final method result.