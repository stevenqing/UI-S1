# Mind2Web ShowUI-ZS Cross-Task Report

Status: `COMPLETE_PUBLIC_CHECKPOINT_RESULT`

## Scope

This is the public `showlab/ShowUI-2B` checkpoint evaluated zero-shot on the
Mind2Web Cross-Task split. It is not the paper's downstream Mind2Web-fine-tuned
ShowUI row.

## Published anchor

| Baseline | Element Accuracy | Operation F1 | Step Success Rate |
| --- | ---: | ---: | ---: |
| ShowUI-ZS paper | 21.4% | 85.2% | 18.6% |

## Reproduction

All values in the primary result are means of per-episode means over the 252
Cross-Task episodes.

| Result | Steps | Element Accuracy | Operation F1 | Step SR | Parse rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Public checkpoint | 2,080 | 23.3366% | 81.9835% | 19.9609% | 100.0000% |
| Paper ShowUI-ZS | - | 21.4000% | 85.2000% | 18.6000% | - |
| Delta | - | +1.9366 pp | -3.2165 pp | +1.3609 pp | - |

This is a complete, controlled result for the public checkpoint, but not an
exact reproduction of the paper anchor. The released training notes state that
zero-shot action names are unstable and that the best intermediate pretraining
score was reported. No corresponding intermediate checkpoint is identified or
published with the public final checkpoint.

## Diagnostics

- Syntactically parseable responses: 2,080 / 2,080 (100.0000%).
- Responses using the benchmark's `CLICK`/`TYPE`/`SELECT` actions: 1,982 / 2,080
  (95.2885%).
- Parsed action counts: `CLICK` 1,974, `SELECT` 8, `INPUT` 77, `SCROLL` 14,
  `ENTER` 7, and `TYPE` 0.
- Micro diagnostics: Element 20.8654%, Operation F1 82.8606%, Step SR 18.1250%.
- The released action-category macro Operation F1 is 32.5860%; it is not the
  paper-comparison metric because zero-shot `INPUT` predictions give the `TYPE`
  category zero credit.

## Configuration

- Public general checkpoint, no downstream state dict or LoRA
- Two prior text-action history entries (`tttt`)
- Visual tokens: 256 minimum, 1,344 maximum
- UI-graph disabled at inference; no language-model token skipping
- Released greedy generation path, 128 maximum new tokens
- Four disjoint single-GPU shards on GPUs 0-3

## Audit notes

- Raw output is appended and fsynced after every prediction.
- The released ShowUI metadata converter stores history in a shape its dataset
  cannot consume. The local converter wraps each history action with its action
  representation, image path, image size, and task; a regression test invokes
  the released `get_answer` implementation on this wrapper.
- Unknown zero-shot action names such as `INPUT` are parseable but remain metric
  failures because the released Mind2Web evaluator accepts only `CLICK`, `TYPE`,
  and `SELECT`.
- The released dataset overwrites `anno_id` with the step index before its
  evaluator groups rows, so its logged `Macro` fields collapse to micro values.
  The primary comparison here preserves the original `annot_id` and computes the
  required mean of per-episode means; the released pseudo-macro values are kept
  only as diagnostics.

## Artifacts

- Predictions SHA-256:
  `e2e47600dfc11cc50bac064dbac7c5dc0b02ed01d08c82472e379d8bd6e97131`
- Score SHA-256:
  `db5cb323ed011e2ed32148b37307a2220e8760c1e16bdb638777f175fd915d59`
- Audit SHA-256:
  `0e46e1529c7cdd7ba1739a5541d823404f5d64c6647d07e7569beb8dbca25765`
- Prepared metadata SHA-256:
  `e3dbd288037f14849ca713f92468adaef67f859a3f2f520fffc2b190a82054ef`
- Audit status: `PASS` with 2,080 unique identities and 252 episodes.