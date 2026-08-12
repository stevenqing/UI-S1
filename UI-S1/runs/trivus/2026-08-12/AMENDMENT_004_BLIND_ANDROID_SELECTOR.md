# Amendment 004: Blind AndroidControl Selector Evidence

Date: 2026-08-12

Timing: after R1 passed and was committed, before generating the TriVUS public AndroidControl bank, before any AndroidControl Qwen3-VL selector inference, and before creating any TriVUS private-label file.

## Public bank

Create exactly 4,000 public records: 2,000 paired rows each for Low and High. Candidate order before display permutation follows the frozen model order UI-AGILE-7B, GUI-R1-7B, UI-R1-E-3B, but candidate source/model/slot identity is omitted from every public record and prompt.

Allowed row fields are schema version, sample key, benchmark, setting, row ID, frozen fold, episode group, image path/hash, instruction, compact history, and three candidates. Allowed candidate fields are canonical action, normalized coordinate or null, parameter, and parse-ok. Any `gt_*`, target, evaluator, success, label, source, model, slot, response, reliability, or fallback field is prohibited.

The public builder validates all six R0 lane hashes/identities against the committed recovery manifest, but it reads no GT field and imports no evaluator. `public_records.jsonl` is written with flush/fsync and locked by SHA-256 before inference.

## Blind query

Each public record receives one Qwen3-VL-8B selector forward. Candidate display order is a SHA-256 permutation of `(sample_key, seed=20260812)`. Coordinate candidates receive A--C screenshot markers; non-coordinate candidates remain text-only. The prompt contains task, compact history, and the three candidate action/coordinate/parameter descriptions.

The model is asked to reply A, B, or C, but inference uses the next-token logits of the three single-token labels rather than generated text. It receives no fallback or KEEP option, source identity, model response, prior reliability, benchmark result, target, evaluator, success, fold, or private label.

Frozen inference:

- model: existing Qwen3-VL-8B-Instruct, index SHA-256 `520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070`;
- bfloat16, SDPA, `use_cache=false`, batch size one;
- max rendered edge 1600;
- processor min pixels `256*28*28`, max pixels `1280*28*28`;
- eight deterministic shards mapped to physical GPUs 0--7;
- per-25-row and final fsync; resumable by unique sample key.

## Blind lock

The finalizer requires 4,000 unique predictions, exact public identity, complete display permutations over three candidates, finite logits/probabilities, probabilities summing to one, matching image/model hashes, prompt hashes, and no private-label file in the TriVUS run.

It writes a blind manifest containing public/prediction hashes and `private_labels_created=false`. Only after that manifest is committed may a separate builder access GT fields to create physically fold-sealed private candidate labels.

Selector direct accuracy, AUROC, R1 comparison, or any other label-derived diagnostic is prohibited before the blind manifest commit.