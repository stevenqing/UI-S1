# UI-TARS AndroidControl Preflight

Status: `FOUR_PUBLIC_CHECKPOINT_TRANSFERS_COMPLETE_AUDIT_PASS`

## Targets

The UI-TARS paper reports the following AndroidControl percentages in
`Type / Grounding / Step SR` order:

| Model | Low | High |
| --- | --- | --- |
| UI-TARS-2B-SFT | 98.1 / 87.3 / 89.3 | 81.2 / 78.4 / 68.9 |
| UI-TARS-7B-SFT | 98.0 / 89.3 / 90.8 | 83.7 / 80.5 / 72.5 |

## Pinned Public Checkpoints

- `ByteDance-Seed/UI-TARS-2B-SFT`
  revision `f366a1db3e7f29635f5b236d6a71dea367a0a700`.
- `ByteDance-Seed/UI-TARS-7B-SFT`
  revision `3434901a9dd04dd3625617d839a5724fe5e2db20`.

These are the annealing-stage SFT model family named by the paper. The public
release does not include its AndroidControl evaluator or exact split-specific
prompt serializer, so results from this runner are controlled transfers rather
than anchor reproductions.

## Data And Prompt Contract

- OS-Atlas prepared lane: exactly 7,708 ordered steps and 1,412 episodes.
- Low: high-level goal plus the current public low-level instruction and prior
  action history.
- High: high-level goal plus prior action history; no current low-level
  instruction.
- Generic official UI-TARS mobile prompt and function-call grammar.
- The global prompt adds `wait()` because WAIT is one of the seven actions in
  this AndroidControl lane; this is a benchmark-level schema declaration, not a
  per-row GT hint.
- Greedy generation, frequency penalty 1, maximum 128 new tokens, seed 0.
- Released 0-1000 relative point coordinates.

## GT-Independent Mapping

- `click` -> `CLICK`
- `long_press` -> `LONG_PRESS`
- `type` -> `TYPE`
- `scroll` -> `SCROLL`
- `open_app` -> `OPEN_APP`
- `press_back` -> `PRESS_BACK`
- `wait` -> `WAIT`

Unsupported functions are strict parse failures. No output is reinterpreted
using the reference action, text, or coordinate.

## Scoring

The scorer uses the existing AndroidControl contract: exact action type;
Euclidean coordinate distance at most 140 in 0-1000 space for click/long press;
exact or token F1 above 0.5 for TYPE/OPEN_APP; exact direction/action for other
functions. Click-only grounding is reported as the main grounding metric, with
coordinate grounding including long press retained as a diagnostic.

## Runtime

vLLM 0.11.0 uses one model replica per GPU and a fixed 4 GiB KV cache. The
fixed cache avoids invalid free-memory profiling when unrelated shared GPU
processes allocate or release memory. Four resumable shards write each row with
`fsync` before strict ordered merge and complete audit.
