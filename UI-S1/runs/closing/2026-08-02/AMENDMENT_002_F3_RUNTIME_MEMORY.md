# Amendment 002: F3 Runtime Memory

Date: 2026-08-02

Status: frozen before any F3 model inference or result.

Protected PID 1814 occupies approximately 20 GB on every 80 GB A100 and must not be killed, paused, or modified. Official UI-Zoomer sets vLLM `gpu_memory_utilization=0.85`, which would reserve approximately 69 GB in addition to the protected allocation and cannot coexist.

F3 uses `gpu_memory_utilization=0.65`. This parameter controls vLLM's KV-cache reservation only. Model weights, dtype, max model length, image preprocessing, prompt, K=8 samples, temperature, top-p, seed, token limit, log-probability extraction, gate, crop, and refinement are unchanged. The run fails closed if the reduced cache cannot serve one K=8 request.