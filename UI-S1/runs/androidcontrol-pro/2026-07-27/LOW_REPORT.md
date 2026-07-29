# AndroidControl-Low OS-Atlas-Pro-7B

Verdict: **COMPLETE**

The public OS-Atlas-Pro-7B successor model was evaluated on all 7,708 pinned
`ac_idx` steps with the AndroidControl-Low prompt. The first three prompts match
the released OS-Atlas samples byte-for-byte.

## Results

| Metric | Result | Count |
| --- | ---: | ---: |
| Upstream-exact parse rate | 100.0000% | 7,708 / 7,708 |
| Type Accuracy | 93.4743% | 7,205 / 7,708 |
| Grounding Accuracy | 86.7576% | click-correct / click-type-match |
| Coordinate Grounding incl. long press | 86.7498% | coordinate-correct / coordinate-type-match |
| Step Success Rate | 83.9647% | 6,472 / 7,708 |

The flexible parser produces exactly the same result as the upstream-exact
parser, and scoring records zero runtime errors.

## Integrity

- Four disjoint shards: 1,927 predictions each.
- Merged coverage: 7,708 unique identities, 0 missing, 0 duplicate, 0 extra.
- Merged order and GT actions exactly match the prepared `ac_idx` input.
- Raw predictions SHA-256:
  `411d0f4b4629f18c3a2963a81ac74059b3a15a19ddd62044b88255c3cdfa1a81`.
- Score SHA-256:
  `f443e5dfbda109b4556a585ade952af2706ea0a5f28078a756ed8a3841e825e9`.
- Independent score recomputation is byte-for-byte identical.

Artifacts are under `artifacts/low_merged/`.

This is a Pro successor baseline, not the unavailable Table 5 zero-shot model.