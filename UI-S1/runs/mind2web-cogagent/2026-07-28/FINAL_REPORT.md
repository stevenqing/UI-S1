# CogAgent Mind2Web Public Transfer Report

Status: `COMPLETE_AUDIT_PASS_NOT_PAPER_REPRODUCTION`

## Result

Mind2Web Cross-Task, 2,080 actions and 252 episodes. Main metrics are
episode-macro percentages.

| Setting | Element Accuracy | Operation F1 | Step Success Rate | Parse |
| --- | ---: | ---: | ---: | ---: |
| `cogagent-chat-hf` public transfer | **58.9280** | **85.4882** | **56.0392** | **94.1346%** |
| Paper anchor | 22.4 | 53.0 | 17.6 | - |

The public-transfer deltas are `+36.5280 / +32.4882 / +38.4392 pp`. These
large differences are evidence of a contract/checkpoint mismatch, not an
anchor reproduction success.

## Fixed Contract

- Model: `zai-org/cogagent-chat-hf`
- Revision: `26eec27a44348fbe0c9fad89348cf6a505f5a5ae`
- Tokenizer: `lmsys/vicuna-7b-v1.5`
- Revision: `3321f76e3f527bd14065daf69dad9344000a201d`
- Prompt: official generic single-round task prompt with `(with grounding)`
- Generation: greedy, maximum 256 new tokens
- Coordinates: released 0-1000 `Grounded Operation` boxes; bbox center used
- Parser: explicit predicted `CLICK`/`TYPE`/`SELECT` only; no GT action repair

The legacy Transformers 4.36.2 dependencies are isolated in the uv overlay.
PyTorch 2.8.0+cu128 and torchvision 0.23.0+cu128 are supplied by the verified
workspace CUDA runtime through `run_python.sh`.

## Coverage And Parsing

- Four shards: exactly 520 rows each.
- Ordered merge: exactly indices 0 through 2,079.
- Unique identities: 2,080; episodes: 252.
- Parsed: 1,958/2,080.
- Parsed actions: `CLICK=1,687`, `TYPE=218`, `SELECT=53`.
- Strict failures: missing bbox 63, missing marker 30, unsupported/implicit 29.
- No response was reinterpreted using the ground-truth action.

## Audit

The complete audit passed model/tokenizer revisions, identity/order,
image/bbox/answer provenance, prompt hashes, generation configuration, score
coverage, and artifact hashes.

- Predictions SHA-256:
  `baa3738fa9982a5d19d7582753e268cd59bafabaafc1d1b8d65de2baf66ffc04`
- Score SHA-256:
  `4b1717a05a29b25a2743d3bb965a3196446a533d790ee54ed7f16d7606fc98e4`
- Metadata SHA-256:
  `e3dbd288037f14849ca713f92468adaef67f859a3f2f520fffc2b190a82054ef`

Artifacts: `artifacts/merged/predictions.jsonl`, `score.json`, and `audit.json`.

## Boundary

The public release does not include the paper's split-specific Mind2Web
prompt, history serializer, action converter, or evaluator. This result is a
controlled public-checkpoint transfer and must remain separate from the paper
anchor.