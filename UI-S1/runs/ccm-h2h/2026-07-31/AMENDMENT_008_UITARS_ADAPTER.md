# Amendment 008: UI-TARS H3 Adapter

Date: 2026-08-01

Status: frozen after a one-row load smoke exposed architecture incompatibility, before any UI-TARS ScreenSpot prediction or eligibility result.

The local pinned `ByteDance-Seed/UI-TARS-7B-SFT` revision `3434901a9dd04dd3625617d839a5724fe5e2db20` declares `Qwen2VLForConditionalGeneration` / `qwen2_vl`. The MVP repository's generic `mvp_sspro.py` imports a modified Qwen2.5-VL class and cannot instantiate this checkpoint without unsupported architecture conversion.

UI-TARS H3 therefore uses the checkpoint-native Transformers `Qwen2VLForConditionalGeneration` and `AutoProcessor`, with the same ScreenSpot locator system prompt and greedy generation contract as H1. Parse the first coordinate pair from the released UI-TARS 0-1000 point space, accepting `<point>x y</point>`, `(x,y)`, or `[[x,y]]`; values outside `[0,1000]` fail closed. Map the point to the current full/crop image, then add the frozen crop offset. Parse failures map to `(0,0)` and are marked invalid.

The shared GTA1 regions remain fixed. No target bbox enters prompt, parsing, crop selection, or coordinate mapping. Eligibility uses full-image predictions only and is frozen before four-view generation.