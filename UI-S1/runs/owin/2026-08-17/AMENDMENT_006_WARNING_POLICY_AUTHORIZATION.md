# OWIN Amendment 006: historical warning policy and renewed authorization

Date: 2026-08-17

Status: `AUTHORIZED_ONE_TIME_OWIN_ARM_A_6000_RUNTIME_006`

Amendment 005 smoke loaded the model but produced 36 failed attempts because global `-W error` promoted the historical Transformers warning for `do_sample=false, temperature=0.0` to an exception before generation. All failed rows are retained; no token was generated and the formal nonce was not consumed.

Historical H1 logs contain the same warning and continue successfully. This amendment preserves decoding exactly and removes global `-W error` only from GPU execution commands. Unit tests remain strict. This is the only change from config 005.

New bindings:

- runner SHA-256: `46852e5f934b87c1348150287f013c3bd5fb0afbf2619f4a28e0a64bb6d1f7ed`;
- config 006 SHA-256: `4323f16723822a81c98e73a17edadea1feeff4e8464b3c604f1dae94604c33be`;
- authorization file: `EXECUTION_AUTHORIZATION_006.json`;
- one-time nonce: `bf0171f5360a7e544a40912653fa296038ed5b349dfd41abf5f835ce77cd15fe`.

Authorization covers exactly one 36-call smoke retry and, only after `PASS_OWIN_SMOKE_36`, the unchanged 6,000 formal calls. Old authorizations 004B and 005 are revoked for execution. Smoke does not consume the formal nonce. All other Part B and Amendment 005 integrity, failure, retention, hardware, and protected-process terms remain binding.