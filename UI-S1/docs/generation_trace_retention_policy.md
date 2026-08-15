# Generation Trace Retention Policy

Effective date: 2026-08-15

Status: required for new generating-model forward runs.

DECOMP Arm 3 found no generating-model per-token logprobs, token IDs, coordinate-token spans, or sequence scores in the retained ScreenSpot-Pro and Mind2Web candidate traces. Downstream selector logits are not a substitute.

Every new generating-model trace row must retain:

- stable sample ID and image/prompt SHA-256;
- model ID, revision, and model-index SHA-256;
- decoded response and generated token IDs;
- per-token logprobs, or `logprobs_unavailable: true` with backend/version/reason;
- coordinate-token span indices when a coordinate is parsed;
- aggregate coordinate-token logprob and its formula;
- raw sequence logprob and length-normalized sequence score with normalization formula;
- temperature, top-p, top-k, seed, decoding mode, and maximum generated tokens;
- parsed action/coordinate/parameter and parse status.

Each shard must be JSONL with per-row write/flush/fsync. A manifest records row count, bytes, SHA-256, schema version, field availability, backend version, and whether any field was unavailable. Missing logprobs must never be represented as zero.

Evaluation labels, target boxes, correctness, rewards, and private folds remain separate from generation traces. Publication packages must continue to exclude them unless a separate private-label protocol explicitly authorizes inclusion.