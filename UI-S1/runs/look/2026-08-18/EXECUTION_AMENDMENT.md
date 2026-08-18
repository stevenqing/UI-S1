# LOOK one-time execution amendment

Date: 2026-08-18

Status: `AUTHORIZED_ONE_TIME_LOOK`

This commit authorizes exactly 9 smoke forwards and, only after smoke passes, exactly 1,290 formal LOOK forwards. It authorizes no method experiment, answer flip, alternate sample, window, prompt, model, or additional calls.

Bound artifacts:

- execution config SHA-256 `358d785756101e2b99aca174671af6b6305901e620c1877065a88629d48c440c`;
- runner SHA-256 `d7bbce08849016a036c129099826b203d8e2ae73098d9cd23a9dda6b6d5a10bb`;
- evaluator SHA-256 `d2f6e4f985a08fde7b0f570dbec3e84dcaff090fde3fc2b0143f4a0010dbe979`;
- formal input SHA-256 `f54867f19ff2a6cdacfe3f25fc357240d204cef043d126088dac350545ed42a9`;
- smoke input SHA-256 `d95b73d8bbf7bd508687a6ff2ba714f1dd4bdf2275c96b073fbc86961ef1de80`;
- GTA1 model revision `701bedc80b447863bd60e3318ae44f6cbbfafd78` and model-index SHA-256 `3067e9b0f35596ff3426a0d0ec8c982a51fa1e110c4fc30dcf3be9ea37409df6`;
- GPU generation runtime: historical H1 Torch 2.6.0+cu124, Transformers overlay 4.51.2, flash-attn 2.7.4.post1, qwen-vl-utils 0.0.11;
- zero-GPU evaluation runtime: `.venv-scaleup`, Python 3.12.13, scikit-learn 1.9.0;
- physical GPU 3, UUID `GPU-477c4cfb-2b63-208d-53b8-5baef1a65b36`, exposed as `cuda:0`;
- prompt SHA-256 `b7a49e56bb9cf4f8d5e4689f35f40c091b6d92ec0c4419458f496219a7807993`;
- greedy decoding, temperature 0, max_new_tokens 32, unchanged processor/parser/coordinate transform;
- one-time nonce `6a35ee2a8a705f5d2a142eff823d1e5e96d9e8040a659c6afa1dd28c567437fa`.

Protected PID 2274 must not be signaled, stopped, reprioritized, or modified. Smoke does not consume the formal nonce. A failing smoke blocks formal execution. At formal start, nonce is consumed regardless of outcome.

Every failure is retained; non-infrastructure failure rate above 1% stops execution. Traces must remain label-free and retain all extended generation fields. No rerun is authorized by this amendment.