# TongUI-32B Mind2Web Report

Status: `COMPLETE_AUDIT_PASS_CONTROLLED_RESULT`

## Result

Mind2Web Cross-Task, 2,080 actions and 252 episodes. Main metrics are
episode-macro percentages.

| Setting | Element Accuracy | Operation F1 | Step Success Rate | Parse |
| --- | ---: | ---: | ---: | ---: |
| TongUI-32B (GUI-Net-1M) | **58.9233** | **89.7424** | **54.3341** | **99.9519%** |
| Paper anchor | 57.2 | 88.1 | 52.4 | - |

Anchor deltas are `+1.7233 / +1.6424 / +1.9341 pp`. This is a complete
controlled result rather than a strict 1 pp reproduction pass.

## Fixed Checkpoint

- Base: `Qwen/Qwen2.5-VL-32B-Instruct`
  revision `7cfb30d71a1f4f49a57592323337a4a4727301da`.
- Adapter: `Bofeee5675/TongUI-32B`
  revision `be18be453c78e6145305a1692aa8e678454f93c2`.
- The official release contains the LoRA adapter only. vLLM 0.11.0 applies it
  natively to the fixed base under tensor parallel size 4.
- All 18 base shards, base index, and adapter were SHA-256 verified in
  `tongui32_checkpoint_manifest.json`.

## Contract

- Official `v2`, `num_history=2`, `vtvt` Mind2Web prompt/data path.
- Visual token bounds: 256 to 1,344.
- Greedy generation, seed 42, 128-token cap.
- Strict released parser; no best-of-N and no GT action repair.
- Input ID hashes match the completed TongUI-3B official-contract run for all
  2,080 rows.

## Coverage And Audit

- 2,080/2,080 ordered rows; 2,080 unique identities; 252 episodes.
- Parsed 2,079/2,080: `CLICK=1,820`, `TYPE=244`, `SELECT=15`.
- Complete independent audit passed checkpoint revisions, identity/order,
  image/bbox/answer provenance, byte-level input hashes, runtime configuration,
  score coverage, and artifact hashes.

- Predictions SHA-256:
  `a4582a93dd4c015fa0bb66baa07a6a0d18308369482b3d23465d558683c51e70`
- Score SHA-256:
  `8af78dad2dd595a5b864be47042114bd970e00e3d0d9882c7e7c8576c21ca82c`

Artifacts are under `artifacts/tongui-32b/full/`.