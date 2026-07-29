# AndroidControl Qwen2.5-VL Base Preflight

Status: `QWEN3_QWEN7_LOW_HIGH_COMPLETE`

This lane evaluates unmodified Qwen2.5-VL base checkpoints as controlled
lower bounds. It reuses the exact 7,708 AndroidControl identities, OS-Atlas
Low/High prompt construction, and scorer used by the completed
OS-Atlas-Pro-7B successor baseline.

## Checkpoints

| Model | Revision | Status |
| --- | --- | --- |
| Qwen2.5-VL-3B-Instruct | `66285546d2b821cf421d4f5eb2576359d3770cd3` | Hash/index verified |
| Qwen2.5-VL-7B-Instruct | `cc594898137f460bfe9f0759e9844b3ce807cfb5` | Hash/index verified |

## Frozen contract

- Data: `data/prepared/ac_high.jsonl`, 7,708 unique ordered identities.
- Settings: High and Low, reported separately.
- Processor: explicit `use_fast=False`.
- Model family: `Qwen2_5_VLForConditionalGeneration`.
- Attention: SDPA, bf16, exactly one visible GPU per process.
- Generation: 128 tokens, `top_k=1`, temperature `0.01`, top-p `0.001`.
- Persistence: modulo-4 shards, duplicate-safe resume, per-row fsync.
- Scoring: upstream exact parser is primary; flexible parser is diagnostic only.
- Audit: ordered identity, episode/step, GT action, reconstructed prompt hash,
  model family/name/revision, processor, generation configuration, setting,
  and full score coverage.

The first-row rendered prompt hashes are:

- High: `2422f752c6e1bbdc1c265d0694ec9f9aa1ce234946e04c57109ea6dc59b8bbd9`.
- Low: `76221d24a4ee0ec4373bdddb509ddafd1a4d57664688953324a768f4beae90c6`.

CPU processor/tokenization dry-runs passed for both settings. No AndroidControl
Qwen metric is reportable until a full 7,708-row merge, deterministic score
recomputation, and independent audit pass.

Qwen2.5-VL-3B High and Low one-row GPU smoke tests pass with independently
reconstructed prompt hashes. A Transformers High partial run was stopped and
retained only as a backend diagnostic; it will not be merged with vLLM output.

The formal High lane completed all 7,708 rows using four vLLM 0.11.0 shards and
passed merge, byte-identical score recomputation, and independent audit. The
strict result is `0 / 0 / 0` because no response uses the exact lowercase
`actions:\n` delimiter; output normalization is not applied.

The formal Low lane completed all 7,708 rows and passed ordered merge,
byte-identical score recomputation, and independent audit. Its strict result is
also `0 / 0 / 0` because no response uses the exact lowercase `actions:\n`
delimiter. One empty shard initially hit a vLLM memory-profiling startup race
while another process released GPU memory; isolated restart completed all rows
without lost output.

Qwen2.5-VL-7B High and Low one-row vLLM smoke tests pass with the same frozen
prompt hashes. High completed all 7,708 rows and passed ordered merge,
byte-identical score recomputation, and independent audit. Its strict result is
`0 / 0 / 0` for the same delimiter-contract reason. Qwen7 Low completed all
7,708 rows and passed ordered merge, byte-identical score recomputation, and
independent audit. Shard 0 used a fixed 8 GiB KV cache to bypass unstable
shared-GPU memory profiling; this did not change prompt or sampling behavior.
Its strict result is also `0 / 0 / 0` because no response uses the exact
lowercase delimiter.