# OS-Atlas Mind2Web Scope Check

Status: `NOT_A_PAPER_BENCHMARK_ROW`

## Conclusion

The OS-Atlas paper does not report a Multimodal-Mind2Web evaluation row for
OS-Atlas-4B or OS-Atlas-7B. Mind2Web is used during grounding pre-training and
action fine-tuning; it is not one of the held-out evaluation benchmarks.

The paper's evaluated agent benchmarks are GUI-Act-Web, OmniAct-Web,
OmniAct-Desktop, AndroidControl-Low/High, and GUI-Odyssey. Grounding is evaluated
on ScreenSpot and ScreenSpot-V2.

## Checkpoint Boundary

- `OS-Copilot/OS-Atlas-Base-4B` and `OS-Atlas-Base-7B` are public grounding
  models. Their paper results are ScreenSpot grounding results, not Mind2Web
  action metrics.
- The original OS-Atlas-4B/7B action models are evaluated on the six agent
  benchmark lanes above. The AndroidControl zero-shot OOD checkpoint remains
  unpublished.
- `OS-Copilot/OS-Atlas-Pro-4B/7B` are all-dataset successor models. The 7B
  AndroidControl result is already reported separately and cannot be relabeled
  as a Mind2Web result.

## Evidence

OS-Atlas Section 5.1 explicitly lists five agent benchmark families and omits
Mind2Web from evaluation. Appendix D lists the test samples and action spaces for
those benchmarks. Mind2Web appears in Sections 3.2 and 5.1 only as training data.

Source: `arXiv:2410.23218v1`; official repository `OS-Copilot/OS-Atlas`.

## Matrix Decision

Remove OS-Atlas-4B/7B from the Mind2Web pending-run matrix. No GPU run is
appropriate because there is no paper Mind2Web row to reproduce. Any optional
public-grounding transfer would be a new experiment, not a baseline completion.
