# OWIN smoke failure 002

Date: 2026-08-17

Status: `PRE_GENERATION_WARNING_PROMOTED_TO_ERROR_RETAINED`

Amendment 005 smoke loaded the bound checkpoint successfully, then all 36 attempts failed before token generation because the execution command included global `-W error`. Transformers 4.51.2 emits the historical H1 warning that `temperature=0.0` is ignored when `do_sample=false`; global warning promotion converted it to an exception.

The same warning appears in retained historical H1 logs and did not stop H1 generation. The frozen decoding contract remains greedy, `do_sample=false`, `temperature=0.0`. The repair removes `-W error` only from GPU execution commands. Unit tests and compilation continue under strict warning mode.

All 36 failed trace rows and the smoke status are retained under `failed_attempts/smoke_005_userwarning/`. They cover three unique non-sample rows and all 12 slots. No successful output or token was generated. Formal calls remain zero and the formal nonce remains unconsumed.

Amendment 005 is revoked for execution. Amendment 006 must bind unchanged runtime/model/data and a new authorization filename, runner hash, config hash, and nonce before retry.