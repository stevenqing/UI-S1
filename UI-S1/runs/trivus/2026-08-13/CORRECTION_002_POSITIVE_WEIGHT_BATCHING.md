# Correction 002: Positive-Weight Batching

Date: 2026-08-13

Timing: after failed exploratory authorization nonce `8f68c38f765c39bf55b2e7344b4f0e5331e8031c18713e58048a283f74d53ece`, before any replacement authorization.

The first cheap OOF phase exposed that inactive rows have zero benchmark/cell weight. Random batching over all family rows can produce a batch containing only zero-weight rows, which correctly fails the candidate-success loss denominator contract. Synthetic tests had used only positive row weights and did not expose this case.

Training and checkpoint evaluation now construct their batch order from `row_weights > 0` only. Zero-weight rows are never forwarded for loss computation. The global normalization remains the total positive weight mass, preserving one optimizer step per epoch and the frozen benchmark/cell weighting.

A regression test uses six zero-weight rows, two positive-weight rows, and batch size one. The launcher also stops scheduling after the first failed worker and terminates only its own remaining child workers.

The failed nonce, receipt, attempt, logs, and partial artifacts are retained. They are invalid for analysis and cannot be replayed or published.