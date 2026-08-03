# Amendment 009: Qwen3 Fixed-Region Runtime

Date: 2026-08-01

Status: runtime correction after all four Qwen3 workers failed during import, before model loading or any Qwen3 H3 prediction.

The pinned MVP overlay uses Transformers 4.51 and its vendored Qwen3 model imports `transformers.masking_utils`, which is unavailable in that overlay. Fixed-region H3 does not require MVP attention capture. Qwen3 therefore uses the downloaded checkpoint's standard `Qwen3VLForConditionalGeneration` under the repository's validated Transformers 4.57.1 / Torch 2.8 runtime.

That runtime does not contain the compiled `flash_attn` package. Qwen3 fixed-region inference uses Transformers/PyTorch `sdpa`. This backend is frozen before any successful Qwen3 smoke or prediction. UI-TARS continues under its validated FlashAttention2 runtime.

Prompt, greedy decoding, smart resize, crop geometry, and 0-1000 coordinate mapping remain identical to Amendment 007. No Qwen3 prediction existed before this correction. UI-TARS workers are unaffected.