# ORTH Amendment 001: CPU OCR execution

Date: 2026-08-14
Timing: after preflight commit `2f32484`, before any project-image OCR forward.

Two engines run independently with 12 deterministic row-interleaved shards each.

- EasyOCR 1.7.2: CPU, quantized English detector/recognizer, PyTorch intra/inter-op threads 1, and the exact `readtext` parameters in `PREFLIGHT.json` including explicit `min_size=10`.
- RapidOCR-ONNXRuntime 1.4.4: CPU, locked config/model files, `intra_op_num_threads=1`, `inter_op_num_threads=1`, detection/classification/recognition enabled.

Every lane writes one fsynced JSONL record per assigned row, including engine errors as records rather than silently dropping rows. It then writes a bytes/SHA-256/row-count manifest. Raw records contain no instruction, target bbox, candidate labels, row class, or matching score. All evaluation-side derivations happen only after raw manifests are locked.

This amendment only controls CPU resource use and raw retention. It does not narrow the exploratory matching grid or introduce a result threshold.
