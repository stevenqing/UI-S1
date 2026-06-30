# GUI-360 GT-History Checkpoint Eval Report

Date: 2026-06-30

## Scope

This report records the completed format-matched GT-history checkpoint eval for the G-arm checkpoints:

- `checkpoint-13`
- `checkpoint-26`
- `checkpoint-39`
- `checkpoint-52`

Important caveat for the checkpoint-selection table: those four-checkpoint numbers are from `train_GUI_360/llamafactory/data/gui360_gt_history_val.json`, which is a train-tail validation slice with 32 episodes / 230 turns. They are useful for checkpoint selection and debugging, but they are not the final balanced test-set result.

After selecting `checkpoint-39` from the validation slice, I ran a full balanced test eval for that checkpoint. The test split was reconstructed into compact GT-history ShareGPT format because the balanced test parquet contains `goal`, `steps`, `screenshots`, and action labels, but `conversation_human` / `conversation_gpt` are `null`.

## Eval Form

The eval is format-matched to the G-arm training data: compact multi-turn ShareGPT, one screenshot per human turn, previous turns carrying GT actions.

Metrics:

- V1 matched: GT-history step accuracy under the matched multi-turn prompt.
- V1 none: current-step-only prompt, no previous turns.
- V1 delta: matched minus none. This is a history-format sensitivity signal, not a pre-registered utilization verdict.
- V2 injected drift: injected-error minus clean accuracy. More negative means stronger sensitivity to corrupted prior action history.
- V3 gap: near minus far on long-dependency pairs. It is only usable as long-horizon evidence when `shuffle_clean=true`.
- V4 plan gain: oracle-plan minus no-plan accuracy.

## Validation Results

| checkpoint | V1 matched | V1 none | V1 delta | V2 clean | V2 injected | V2 delta | V3 near | V3 far | V3 gap | V3 shuffle clean | V4 none | V4 oracle | V4 delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|
| checkpoint-13 | 15.2% | 2.6% | +12.6pp | 15.2% | 13.6% | -1.6pp | 10.0% | 3.3% | +6.7pp | false | 14.3% | 12.6% | -1.7pp |
| checkpoint-26 | 24.8% | 12.2% | +12.6pp | 25.2% | 20.7% | -4.5pp | 20.0% | 16.7% | +3.3pp | false | 24.8% | 24.3% | -0.4pp |
| checkpoint-39 | 28.3% | 11.3% | +17.0pp | 27.8% | 24.7% | -3.1pp | 20.0% | 16.7% | +3.3pp | false | 27.4% | 27.8% | +0.4pp |
| checkpoint-52 | 26.5% | 14.3% | +12.2pp | 26.5% | 23.7% | -2.8pp | 20.0% | 16.7% | +3.3pp | false | 26.5% | 26.5% | 0.0pp |

Full artifacts:

- Matrix markdown: `outputs/gui360_history_ab/checkpoint_evals_20260630_104554/checkpoint_eval_matrix.md`
- Matrix JSON: `outputs/gui360_history_ab/checkpoint_evals_20260630_104554/checkpoint_eval_matrix.json`
- Per-checkpoint rows and summaries: `outputs/gui360_history_ab/checkpoint_evals_20260630_104554/checkpoint-*/`

## Interpretation

`checkpoint-39` is the best checkpoint on the current validation slice. It has the highest matched GT-history accuracy at 28.3%, and the largest matched-minus-none gap at +17.0pp.

The final checkpoint, `checkpoint-52`, is slightly worse than `checkpoint-39` on matched GT-history accuracy: 26.5% vs 28.3%.

All V3 results have `shuffle_clean=false`, so the near/far gaps should not be interpreted as clean long-horizon memory evidence. The shuffle-control gaps did not collapse, which means the observed V3 near/far differences are likely pair/source artifacts or generic difficulty effects.

V4 oracle-plan gains are approximately zero or negative. Under this eval form, the checkpoints do not show a meaningful oracle-plan recovery effect.

The current evidence says the G-arm learns matched history-format consumption and action prediction, but this validation eval does not establish clean long-horizon memory use.

## Full Test Result

Selected checkpoint: `checkpoint-39`.

Dataset: reconstructed `gui360-balanced` test split, 1000 episodes / 7498 steps.

Eval form: matched compact GT-history ShareGPT prompt. The test run used two TP=4 vLLM replicas over GPUs 0-7, with `gpu_memory_utilization=0.60`, `max_model_len=32768`, `limit_mm_per_prompt={"image":32}`, and no fixed `kv-cache-memory-bytes`.

| metric | value |
|---|---:|
| Step accuracy | 31.36% |
| Correct steps | 2351 / 7498 |
| TSR | 9.80% |
| Successful episodes | 98 / 1000 |
| Avg progress | 16.87% |

Shard details:

| shard | episodes | steps | Step accuracy | TSR | Avg progress |
|---|---:|---:|---:|---:|---:|
| part0 | 500 | 3863 | 31.22% | 8.80% | 15.41% |
| part1 | 500 | 3635 | 31.50% | 10.80% | 18.33% |

Full test artifacts:

- Merged summary: `outputs/gui360_history_ab/test_matched_ckpt39_gpuutil_merged_20260630/summary.md`
- Merged JSON: `outputs/gui360_history_ab/test_matched_ckpt39_gpuutil_merged_20260630/summary.json`
- Merged rows: `outputs/gui360_history_ab/test_matched_ckpt39_gpuutil_merged_20260630/rows.jsonl`
- Shard 0: `outputs/gui360_history_ab/test_matched_ckpt39_gpuutil_part0_20260630/checkpoint-39/`
- Shard 1: `outputs/gui360_history_ab/test_matched_ckpt39_gpuutil_part1_20260630/checkpoint-39/`

Interpretation: checkpoint-39 remains the selected checkpoint, but the full reconstructed test score is much lower than the train-tail validation slice. The result should be read as matched GT-history compact-prompt performance on the available test labels, not as a direct reproduction of the old GUI-360 template TSR table.

## Original GUI-360 Template Test Result

To check whether the old GUI-360 eval form gives a better score, I also ran `checkpoint-39` with the original GUI-360 template evaluator: `v13_gui_360/eval_gui360_template.py --gt_history`.

Dataset: reconstructed `gui360-balanced` test JSONL, 1000 episodes / 7498 steps.

Eval form: original GUI-360 prompt template, teacher-forced GT history, full-history mode. The run used two TP=4 vLLM replicas over GPUs 0-7, with `gpu_memory_utilization=0.60`, `max_model_len=32768`, `limit_mm_per_prompt={"image":2}`, and no fixed `kv-cache-memory-bytes`.

| metric | value |
|---|---:|
| TSR | 7.10% |
| Successful episodes | 71 / 1000 |
| Avg progress | 18.80% |
| Step SR | 29.03% |
| Correct steps | 2177 / 7498 |
| Mean reward | 0.3958 |

Shard details:

| shard | episodes | steps | TSR | Avg progress | Step SR | Mean reward |
|---|---:|---:|---:|---:|---:|---:|
| part0 | 500 | 3863 | 6.20% | 17.06% | 28.04% | 0.3889 |
| part1 | 500 | 3635 | 8.00% | 20.54% | 30.10% | 0.4032 |

Original-template artifacts:

- Merged summary: `outputs/gui360_history_ab/original_template_ckpt39_gpuutil_merged_20260630/summary.md`
- Merged JSON: `outputs/gui360_history_ab/original_template_ckpt39_gpuutil_merged_20260630/summary.json`
- Shard 0: `outputs/gui360_history_ab/original_template_ckpt39_gpuutil_part0_20260630/`
- Shard 1: `outputs/gui360_history_ab/original_template_ckpt39_gpuutil_part1_20260630/`

Comparison:

| eval form | TSR | Avg progress | Step metric | Correct steps |
|---|---:|---:|---:|---:|
| compact matched GT-history | 9.80% | 16.87% | 31.36% | 2351 / 7498 |
| original GUI-360 template GT-history | 7.10% | 18.80% | 29.03% | 2177 / 7498 |

The original template does not improve TSR for this checkpoint. It does improve average progress, but step-level success and full task success are both lower than the compact matched GT-history eval.

## Operational Notes

The eval used vLLM with Qwen2.5-VL-7B constraints:

- Tensor parallel size must be 4 because Qwen2.5-VL-7B has 28 attention heads, which is not divisible by 8.
- The final full test run did not use fixed KV cache. It used `gpu_memory_utilization=0.60` and no `--kv-cache-memory-bytes` so vLLM selected the cache size from the same utilization rule.
- Earlier validation/debug runs used fixed KV cache only to work around vLLM memory-profiling races on shared GPUs; those fixed-KV runs are not the reported full-test result.
- The eval script is `scripts/run_gui360_gt_history_checkpoint_evals.sh`.

## Test Reconstruction Notes

The test split has 1000 episodes / 7498 steps and no stored conversation text. The test GT-history JSON was reconstructed from:

- `goal`
- per-step `action`
- per-step `bbox`
- per-step screenshot bytes

For test labels, many `type` actions lack text and `swipe` actions lack start/end coordinates. The evaluator therefore treats missing action arguments as unavailable labels, not as model mistakes beyond function mismatch.