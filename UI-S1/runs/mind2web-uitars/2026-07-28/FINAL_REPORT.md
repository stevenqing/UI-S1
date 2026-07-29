# UI-TARS Mind2Web Public Transfer Report

Status: `TWO_MODELS_COMPLETE_AUDIT_PASS_NOT_PAPER_REPRODUCTION`

## Results

Mind2Web Cross-Task, 2,080 actions and 252 episodes. Main metrics are
episode-macro percentages.

| Model / setting | Element Accuracy | Operation F1 | Step Success Rate | Parse |
| --- | ---: | ---: | ---: | ---: |
| UI-TARS-2B-SFT public transfer | **28.3011** | **65.1850** | **22.6630** | **77.8846%** |
| UI-TARS-7B-SFT public transfer | **44.1924** | **84.7497** | **37.8746** | **98.5096%** |
| 2B paper anchor | 62.3 | 90.0 | 56.3 | - |
| 7B paper anchor | 73.1 | 92.2 | 67.1 | - |

These are controlled public-checkpoint transfers, not paper-anchor
reproductions. The release does not include its Mind2Web prompt/history
serializer or action converter.

## Fixed Contract

- Models: `ByteDance-Seed/UI-TARS-2B-SFT` and `UI-TARS-7B-SFT`.
- Revisions: `f366a1db3e7f29635f5b236d6a71dea367a0a700` and
  `3434901a9dd04dd3625617d839a5724fe5e2db20`.
- Prompt: released generic computer-use prompt, single screenshot and task.
- Generation: greedy, frequency penalty 1, maximum 128 new tokens.
- Coordinates: released 0-1000 points divided by 1000.
- Mapping: predicted `click` to `CLICK` and predicted `type` to `TYPE`.
- Unsupported `scroll`, `wait`, `finished`, and `hotkey` outputs score zero.
- No `type` is reinterpreted as `SELECT`, and no GT action/bbox repair is used.

The legacy checkpoint image bounds remain exactly 3,136 to 2,116,800 pixels.
vLLM uses a fixed 4 GiB KV cache so unrelated shared-GPU memory changes cannot
invalidate initialization profiling.

## Coverage And Parsing

Both models completed four shards of exactly 520 rows, ordered merge indices
0 through 2,079, 2,080 unique identities, and 252 episodes.

- 2B parsed 1,620/2,080, all `CLICK`; strict failures were 459 `scroll` and
  one `wait`.
- 7B parsed 2,049/2,080: `CLICK=1,975`, `TYPE=74`; strict failures were 27
  `scroll`, two `finished`, and two `hotkey`.
- The released generic `type(content=...)` action has no target point, so its
  element score is zero under this visual grounding evaluator.
- The released action space has no explicit `SELECT`; those GT rows are not
  repaired using labels.

## Audits

Both complete audits passed identity/order, image/bbox/answer provenance,
prompt hashes, fixed model revisions, generation configuration, score
coverage, and artifact hashes.

| Model | Predictions SHA-256 | Score SHA-256 |
| --- | --- | --- |
| 2B | `131511e4f5bbdcbd4b5037c5af559e4add59da7e9b05cd50a02de38e66d0a7d4` | `371484a8e4c78f52e782c81aa3acb29d9a6ec874723716b5d2ed1cb55e54ca0d` |
| 7B | `5e531c0d69d38d725733b7e90f002317fe934f25c9e9de51e9f7ffcc456f0436` | `9a2ce91ed59b0717a4d2cd1ef85e97d60cd1e64330dcc94797ce3d82d282b942` |

Metadata SHA-256 for both:
`e3dbd288037f14849ca713f92468adaef67f859a3f2f520fffc2b190a82054ef`.

Artifacts are under `artifacts/2b/merged/` and `artifacts/7b/merged/`.
