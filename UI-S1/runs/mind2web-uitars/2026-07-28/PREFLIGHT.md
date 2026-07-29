# Mind2Web UI-TARS Preflight

Status: `UI_TARS_2B_7B_PUBLIC_TRANSFER_COMPLETE_AUDIT_PASS`

## Paper targets

| Model | Element Accuracy | Operation F1 | Step Success Rate |
| --- | ---: | ---: | ---: |
| UI-TARS-2B-SFT | 62.3% | 90.0% | 56.3% |
| UI-TARS-7B-SFT | 73.1% | 92.2% | 67.1% |

The UI-TARS paper states that offline benchmark main-table results use the
annealing-stage SFT model, not DPO.

## Pinned checkpoints

- `ByteDance-Seed/UI-TARS-2B-SFT`
  - revision: `f366a1db3e7f29635f5b236d6a71dea367a0a700`
  - two shards downloaded and hash/index verified
- `ByteDance-Seed/UI-TARS-7B-SFT`
  - revision: `3434901a9dd04dd3625617d839a5724fe5e2db20`
  - seven shards downloaded and hash/index verified

Official source: `bytedance/UI-TARS` revision
`582f3a7ea5d285ee8ed9e2e84048d1ab01453c49`.

Forbidden substitutes:

- `ByteDance-Seed/UI-TARS-7B-DPO@727b0df39207dafc6cf211a61f29d84b7659c39c`
- `ByteDance-Seed/UI-TARS-1.5-7B`

## Remaining gate

The original checkpoints use Qwen2-VL and emit the UI-TARS action syntax with
coordinates in a 0-1000 relative space. They cannot be evaluated by the TongUI
JSON parser or by treating those values as pixels/0-1 coordinates. The model
cards and `README_v1.md` recover the coordinate conversion, but the repository
does not release a Mind2Web evaluator. Before inference, the exact original
Mind2Web prompt, history serialization, action-type mapping, and 1000-space-to-
normalized coordinate conversion must be recovered from the pinned UI-TARS
release. The paper-comparable score remains blocked until this evaluator gate
passes. Separately labeled public checkpoint transfer scores have completed
under the generic contract and are reported in `FINAL_REPORT.md`.

Recovered generic settings from `README_v1.md`:

- computer-use prompt with screenshot/action history;
- `Thought:` followed by function-style `Action:`;
- at most five images per prompt;
- 128-token generation cap;
- `click`, `type`, `scroll`, and desktop-specific actions;
- output points divided by 1000 for normalized coordinates.

Still missing for paper-comparable Mind2Web evaluation:

- the exact split-specific prompt/history construction;
- the mapping from UI-TARS functions to Mind2Web `CLICK`/`TYPE`/`SELECT`;
- especially the treatment of `SELECT`, which is absent from the released
  generic computer action space;
- the released grouping/scoring implementation.

No mapping may use the ground-truth action type to reinterpret a prediction.

## UI-TARS-2B model integrity

- `model-00001-of-00002.safetensors`: 4,982,048,792 bytes,
  SHA-256 `8c1fe51e8b3b73cbbcdf97f0e087602fbe1377989d23da5b840c9bab0825598a`
- `model-00002-of-00002.safetensors`: 4,787,467,560 bytes,
  SHA-256 `b1504c414c865888c250f207e5619a739507923f34bd90adb66d71bc4d751545`
- Index SHA-256:
  `afe6dc9214c4bcf414d7a171fbd26ca7d7a441a2abb3b571d212b56ba5ba58af`

## UI-TARS-7B model integrity

- Seven safetensor shards: 33,165,581,800 bytes total.
- Shard SHA-256 values:
  - `5291c331701a3a115c1f6b2c0f6887d80b45b8a843b76dda7e08443e186b4513`
  - `5159c931e56bcfe421823083b36b93c02cbb0bf25ba4c48d77649d2b9c203da4`
  - `7ca2f68a7a6b30a9dab1b718fc135eb7b4bc0f28ebfca79590adbac9b5360e3f`
  - `6d9b4f8b8ac2927f7e3d64f83ad2a48f6178e7cc1962aaf2d8111864b17674cc`
  - `e51b6adfac7783bb026de2b58aa78e94dc81caac2e76b275ae0b2c7bf6f3f20c`
  - `5d7131a7be9569e3dd89e2feb26ab5e536d948786ff3aa784bd5752edc6ca35d`
  - `8f26ab834f5cfad04f1c1a2716aa037270bd9ae4ede160a4a15e420c39f3834c`
- Index SHA-256:
  `25b162a0f0f47af097d6a49b7da3d5c7d9c2b352490131c8cde5ca59d285f18b`