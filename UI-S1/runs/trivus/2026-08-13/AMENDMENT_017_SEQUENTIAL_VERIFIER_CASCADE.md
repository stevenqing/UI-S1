# Amendment 017: Sequential Verifier Cascade

Date: 2026-08-13

Timing: after the post-hoc sequential budget atlas. Existing labels are open, so the atlas and design are exploratory only.

## 1. Compact method

The method uses one cheap contextual scorer to order all candidates, then applies a stronger calibrated verifier sequentially. At each inspected candidate it chooses one of three actions:

- `accept`: use the candidate;
- `continue`: inspect the next candidate;
- `fallback`: return the benchmark's strongest frozen fallback.

Runtime success is never observed. `First true success` and `hit@k` are evaluation metrics only.

## 2. Stopping rule

For candidate success probability `p` and fallback success probability `b`, accept only if:

- expected gain `p - b` is at least the calibrated minimum delta; and
- fallback-loss risk `b * (1 - p)` is at most the calibrated maximum loss risk.

Otherwise continue while budget remains. If no candidate is accepted, return fallback. Thresholds and maximum budget are selected from OOF development rows independently per benchmark and sealed before held-out access.

## 3. Budget evidence

Under the frozen blind visual ordering, AndroidControl needs budget 2 to recover at least 90% of its full oracle in both cells. Mind2Web needs budget 6 in all four cells. ScreenSpot-Pro needs budget 4 in two cells, 5 in one, and 6 in one.

These are diagnostic upper-bound curves because `hit@k` uses labels for evaluation. They justify a sequential verifier but do not select deployment thresholds.

## 4. Training protocol

The cheap ordering scorer uses the full-candidate objective in Amendment 016. The stronger verifier may reuse its contextual representation with an additional candidate-specific semantic verification head. Candidate verifier outputs used for stopping calibration must be OOF. All benchmark-specific budgets, calibration maps, and stopping thresholds require a separate frozen real-data training config before optimizer execution.