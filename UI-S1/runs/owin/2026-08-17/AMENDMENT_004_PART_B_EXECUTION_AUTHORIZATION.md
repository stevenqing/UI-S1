# OWIN Amendment 004 Part B: one-time Arm A execution authorization

Round: `owin`

Amendment: `004B`

Date: 2026-08-17

Status: `AUTHORIZED_ONE_TIME_OWIN_ARM_A_6000`

This commit constitutes one-time authorization for the 36-call smoke test and, only after smoke passes, exactly 6,000 formal GTA1-7B forwards. It authorizes no other model, sample, window, prompt, parser, device, shard, retry, or additional call.

## Prerequisite verification

All prerequisites were committed and pushed before this authorization:

| Prerequisite | Commit |
| --- | --- |
| Amendment 004 Part A | `5f8ef009f8a7dfd967b9bb531f7801149c043ee7` |
| Geometry calibration preflight | `cf3747e` |
| Arm B outputs | `1f135a9` |
| 500-row sample and 12-window manifest | `20b8923` |
| Label-free formal/smoke inputs and failed-attempt retention | `93956e2` |
| Runner and tests | `53c1610` |
| Non-authorizing execution config | `8cbea59e7f4e8275c29180aeb1fb5d8aab06c907` |

Preflight selected R=300 as an interior grid value. Formal input contains 500 unique rows and 6,000 fixed slots. Smoke contains three non-overlapping rows and 36 fixed slots. Runner validation and 14 unit tests passed with GPU hidden before this authorization.

## Bound execution contract

| Field | Bound value |
| --- | --- |
| Model | GTA1-7B |
| Model revision | `701bedc80b447863bd60e3318ae44f6cbbfafd78` |
| Official source revision | `988ff3c61b9f7632d780ae27c83260de75b3c95f` |
| Model index SHA-256 | `3067e9b0f35596ff3426a0d0ec8c982a51fa1e110c4fc30dcf3be9ea37409df6` |
| Runner SHA-256 | `04e972e225f85de4c0f6a35a80601c4b3c92ff6e853d0d21bf470ea32daccae3` |
| Execution config SHA-256 | `71a78cc17b1eae233bc3b6406b5db60069165621dba5b3c4c5e860ff51e1b7d1` |
| Hardware | one NVIDIA A100-SXM4-80GB, physical index 3, UUID `GPU-477c4cfb-2b63-208d-53b8-5baef1a65b36` |
| Visibility | `CUDA_VISIBLE_DEVICES=3`; runner device `cuda:0` |
| Driver / CUDA | 580.167.08 / Torch CUDA 13.2 |
| Backend | Python 3.12.13; Torch 2.13.0+cu132; Transformers 5.14.1; qwen-vl-utils 0.0.14 |
| Logical shards | 3, ordered common, partial, uncovered |
| Boundaries | common 200×12=2,400; partial 150×12=1,800; uncovered 150×12=1,800 |
| Formal calls | exactly 6,000 |
| Smoke calls | exactly 36, excluded from every estimator |
| Prompt template SHA-256 | `b7a49e56bb9cf4f8d5e4689f35f40c091b6d92ec0c4419458f496219a7807993` |
| Processor bounds | min_pixels 3,136; max_pixels 8,847,360 |
| Pre-smart-resize | full image 1×; crop 2×, matching H1 |
| Decoding | greedy, do_sample false, temperature 0, max_new_tokens 32, use_cache true, pad 151645 |
| Parser function SHA-256 | `86d92fd28135567647a88f7b08e1c0b32ec2a8b7ab381839299d5d1d951e2877` |
| MVP source SHA-256 | `bf2b218d8770a6cc950b87b1d652431da932bbd70fcfe2121f681dd26ffcd4ec` |
| Logprob collection | enabled with return_dict_in_generate/output_scores; explicit unavailable reason if unsupported |
| One-time nonce | `5765da8a85444a4518564e23475c460019771fdce792a87843fd488b7420afef` |

The complete machine-readable contract is `configs/arm_a_execution.yaml`. GPU 2 is excluded because an unrelated vLLM engine is present. Protected PID 2274 is present across GPUs and must not be signaled, stopped, reprioritized, or modified. The bound process may coexist on GPU 3 without altering that process.

## Disclosed differences from historical H1

The intended study intervention replaces attention-proposed crops with committed oracle windows. The runner enables `return_dict_in_generate` and `output_scores` solely for trace retention. Historical runtime package versions were not retained; current versions are bound above rather than claimed identical. Formal execution uses one bound GPU and three sequential logical shards instead of historical eight-way generation. All known invariant fields are model/source revisions, prompt bytes, processor bounds, full/crop resize semantics, greedy decoding, max tokens, parser, and crop-local coordinate transform.

Any unlisted difference in a bound field is a protocol violation and invalidates formal outputs.

## Smoke authorization

Run exactly three committed non-sample rows, 12 slots each, writing only to `smoke/`. Smoke does not consume the formal nonce. It must verify:

1. selected-R integer offsets and committed windows match;
2. crop-local to full-image coordinate transformation follows the tested H1 contract;
3. every successful trace has the complete retained schema and token scores, or an explicit backend/version unavailable reason;
4. traces contain no target bbox, correctness, reward, stratum, or evaluation label.

Any smoke failure blocks formal execution. `PASS_OWIN_SMOKE_36` is mandatory.

## Formal order and nonce

After smoke passes, execute common, partial, then uncovered. The runner writes `raw/NONCE_CONSUMED.json` before the first formal model load. At that point the nonce is consumed regardless of success or failure. No formal shard may run twice under this authorization.

Generation performs no correctness or endpoint calculation. Common outputs are parsed first only after all generation integrity requirements are satisfied; no common output may alter later shard settings.

## Failure handling

Each failed call is retained with `status=failed` and backend error. It is never silently replaced. Unparsable output is retained and later evaluated as incorrect.

One whole-shard rerun is permissible only when failure rate exceeds 1%, a retained audit attributes it to OOM, node loss, or backend crash, and every bound field remains identical. Such a rerun requires an additional execution amendment because the one-time nonce is consumed; this authorization alone cannot launch it. Non-infrastructure failure above 1% stops execution.

Unsupported token logprobs do not stop execution if every row records `logprobs_unavailable`, backend/version, and reason; missing values are never zero-filled.

## Integrity before parsing

Formal traces must contain exactly 6,000 unique `(sample_id,row_id,slot)` rows, exactly 12 slots per committed row, and bit-identical window coordinates. Generation traces must contain no evaluation fields. Every JSONL uses per-row write, flush, and fsync. Any mismatch stops parsing and requires a retained audit.

Final raw traces, private evaluation joins, manifests, statuses, smoke outputs, and failed attempts must be independently backed up under `/scratch/workspaceblobstore/owin/2026-08-17`; `STATUS.json` records backup manifest path and SHA-256.

No sample resizing, extra calls, prompt/parser/coordinate changes, alternate settings, or same-round rerun is authorized.