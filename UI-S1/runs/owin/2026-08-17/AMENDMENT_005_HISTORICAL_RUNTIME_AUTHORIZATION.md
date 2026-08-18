# OWIN Amendment 005: historical runtime repair and renewed authorization

Date: 2026-08-17

Status: `AUTHORIZED_ONE_TIME_OWIN_ARM_A_6000_RUNTIME_005`

Part B commit `f861859` is revoked for execution because its smoke failed before checkpoint loading under an incompatible Transformers 5.14.1 runtime. `SMOKE_FAILURE_001.md` and JSON retain the full audit: zero model forwards, no trace, and no formal nonce consumption.

Repository preflight evidence binds the historical passing H1 runtime to `runs/mind2web-tongui/2026-07-28/.venv/bin/python`, Torch 2.6.0+cu124, flash-attn 2.7.4.post1, and the frozen Transformers 4.51.2 overlay. A CPU-only `mvp_sspro` import gate and all 14 runner/geometry tests pass under that environment.

This commit renews authorization only for the same 36-call smoke and the same 6,000 formal calls. Sample, windows, R=300, model, prompt, processor, parser, decoding, shard boundaries, endpoints, and thresholds are unchanged.

## Revised bindings

| Field | Value |
| --- | --- |
| Runner SHA-256 | `8fe9790e10899d7853b1674cad68c47d0be54d7be7d8826579d1a7bd1686709e` |
| Config 005 SHA-256 | `af38d6fd2764005835c1389ca85eff964b24400c265eb29459143c4a12ead224` |
| Historical Python binary SHA-256 | `9544d2a29138833e6177d45dbc57468d37710b5080c901fbb579d53f251cdd6f` |
| Transformers | frozen overlay 4.51.2; init SHA-256 `dc39d01f278c7ddf32f973d3aee031ed5811351d9f93b652c577b8a2ec8fed94` |
| Torch / CUDA | 2.6.0+cu124 / 12.4 |
| flash-attn | 2.7.4.post1 |
| qwen-vl-utils | 0.0.11 |
| W3 asset preflight SHA-256 | `265bc1e968d8dd1620adebe1e73590849cd5de78803d605e368ab6eaf5ac5b0e` |
| GPU | unchanged physical GPU 3, A100-SXM4-80GB, UUID `GPU-477c4cfb-2b63-208d-53b8-5baef1a65b36` |
| Formal calls | exactly 6,000 |
| Smoke retry | exactly 36 calls; still excluded from estimators |
| New nonce | `2fe02b4ba750c9c74993b2d2af98094b16a40e8af4b3696763a7f00ae831a34a` |

The runner now requires `EXECUTION_AUTHORIZATION_005.json`; the old authorization file cannot launch it. Smoke still does not consume the formal nonce. Formal execution remains blocked until `PASS_OWIN_SMOKE_36` exists. At first formal start, the new nonce is consumed regardless of outcome.

All failure, integrity, retention, protected-process, no-retry, and parsing rules from Part B remain binding. No additional calls are authorized.