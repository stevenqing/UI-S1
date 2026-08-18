# OWIN smoke failure 001

Date: 2026-08-17

Status: `PRE_FORWARD_IMPORT_FAILURE_RETAINED`

The authorized 36-call smoke command was invoked once under Part B commit `f861859`. It failed while importing the vendored Qwen2.5-VL processor, before checkpoint loading and before any model forward.

The bound `.venv-scaleup` runtime supplied Transformers 5.14.1. The vendored processor imports `VideoInput` from `transformers.image_utils`, an API provided by the historical frozen overlay but absent in 5.14.1. The exception was:

`ImportError: cannot import name 'VideoInput' from 'transformers.image_utils'`.

No smoke trace file was created, formal calls remain zero, and `raw/NONCE_CONSUMED.json` does not exist. The formal nonce from Part B was not consumed. GPU 3 was not modified beyond the failed process startup, and no existing process was signaled or stopped.

Repository evidence identifies the correct historical runtime in `runs/collision-law/2026-07-30/W3_ASSET_PREFLIGHT.json`: Python environment `runs/mind2web-tongui/2026-07-28/.venv`, Torch 2.6.0+cu124, flash-attn 2.7.4.post1, and `w3_assets/mvp-overlay` Transformers 4.51.2. A CPU-only import gate under that exact environment passed.

Part B's backend binding is therefore revoked for execution. Amendment 005 must bind the historical runtime, a new runner hash, config hash, and nonce before smoke may be retried. This is an infrastructure import correction before any OWIN model output, not a change to sample, windows, model, prompt, parser, decoding, endpoint, or threshold.