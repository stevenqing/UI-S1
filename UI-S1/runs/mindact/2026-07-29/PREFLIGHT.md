# MindAct Mind2Web HTML-Lane Preflight

Status: `DATA_MODEL_READY_GPU_EVALUATION_PENDING`

## Target

Mind2Web Cross-Task paper anchor: Element Accuracy 55.1, Operation F1 75.7,
Step Success Rate 52.0.

This is the original HTML-based MindAct lane. It contains 2,094 actions in 252
episodes and must not be mixed with the 2,080 scoreable screenshot/bbox lane.

## Fixed Public Artifacts

- Dataset: `osunlp/Mind2Web@17ece8eb89862368edc0cc806acee6fca5163474`.
- Candidate scores: official `scores_all_data.pkl`, covering 2,094/2,094
  Cross-Task actions.
- Action model: `osunlp/MindAct_ActionPrediction_flan-t5-xl`
  revision `848f8100c508e5a742ec2d3ec175b7baa704334c`.
- Tokenizer: `google/flan-t5-xl`
  revision `7d6315df2c2fb742f0f5b556879d730926ca9001`.
- Full hashes and sizes: `artifact_manifest.json`.

The password-protected official test archive was extracted locally with the
public password from the Mind2Web README. It contains `raw_html`,
`cleaned_html`, positive candidates, and negative candidates. No test data is
redistributed.

## Evaluator Contract

- Split: `test_task`.
- Candidate rank cutoff: top 50.
- Five candidates per tournament round.
- Official `MultiChoiceDataset`, HTML pruning, and
  `ActionEvaluatorMultiChoice` implementation.
- Model mode: multichoice; max HTML context 512 and instruction context 512.
- Seed: 123 for Python, NumPy, and PyTorch.
- No GT candidate insertion or output repair.

The original 2023 script calls Optimum `BetterTransformer` before inference.
The current runner omits that deprecated graph-only optimization while keeping
the same checkpoint, tokenizer, inputs, tournament order, and greedy generation.
This is a runtime compatibility boundary, not an action/evaluator change.

The full CPU data-construction check passed with 2,094 actions, 252 episodes,
and complete candidate-score injection. The two model shards exactly match the
published index; T5 config (`d_model=2048`, 24 layers) and the fixed 32,100-token
tokenizer are compatible. GPU smoke and full evaluation remain.
