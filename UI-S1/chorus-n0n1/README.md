# CHORUS N0/N1

Executable scaffold for the N0 headroom and N1 reader-disagreement study.

The workflow is gate-driven. Run Phase A first and stop at the generated Gate A
report before attempting N0 or N1.

## Phase A Preflight

```bash
cd /home/aiscuser/UI-S1/UI-S1
python chorus-n0n1/src/phase_a.py \
  --config chorus-n0n1/configs/android_control_hiconagent.yaml \
  --config chorus-n0n1/configs/gui_odyssey_hiconagent.yaml \
  --config chorus-n0n1/configs/android_control_har_gui.yaml \
  --config chorus-n0n1/configs/gui_odyssey_har_gui.yaml \
  --preflight-only
```

The command writes:

- `chorus-n0n1/runs/<run_id>/preflight.json`
- `chorus-n0n1/REPORT_GATE_A.md`

Phase A cannot run real baseline reproduction until the benchmark screenshots,
GUI-Odyssey data, model checkpoints, and official/local-official repositories are
configured. HiconAgent remains the primary target; HAR-GUI-3B is configured only
as the explicit secondary public-checkpoint baseline.

## Local Assets

The HiconAgent repo is expected at:

```bash
chorus-n0n1/data/external/CVPR26-HiconAgent
```

The public HAR-GUI-3B checkpoint is expected at:

```bash
chorus-n0n1/data/models/HAR-GUI-3B
```

Start the HAR-GUI-3B secondary baseline server on GPUs 4-7 with:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_USE_V1=0 NCCL_DEBUG=WARN \
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name har-gui-3b \
  --model chorus-n0n1/data/models/HAR-GUI-3B \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.60 \
  --max-model-len 8192 \
  --limit-mm-per-prompt image=4 \
  --enforce-eager
```

The V0/eager settings avoid a vLLM V1 memory-profiling assertion seen during
startup on this node.

GUI-Odyssey random-split test annotations can be fetched with the helper script.
The full hflqf screenshot archive is still required because public single-image
mirrors do not cover every random-split test screenshot.

```bash
python chorus-n0n1/scripts/download_gui_odyssey_subset.py \
  --data-dir datasets/GUI-Odyssey \
  --split random_split \
  --subset test

python - <<'PY'
from huggingface_hub import hf_hub_download
files = [f'screenshots/screenshots.z{i:02d}' for i in range(1, 9)] + ['screenshots/screenshots.zip']
for filename in files:
    print(hf_hub_download('hflqf88888/GUIOdyssey', filename, repo_type='dataset', local_dir='datasets/GUI-Odyssey'))
PY

cd datasets/GUI-Odyssey/screenshots
zip -s 0 screenshots.zip --out merged_screenshots.zip
unzip merged_screenshots.zip -d ../data/screenshots
cd -

python gui_odyssey_eval/convert_to_eval_format.py \
  --data_dir datasets/GUI-Odyssey \
  --split random_split \
  --subset test
```

## Non-Negotiables

- All model calls must go through `src/infer/wrapper.py`.
- Every generation logs `finish_reason` and `truncated`.
- Any report section with more than 1% truncated generations is invalid.
- Benchmark matching must call official/local-official matching code rather than
  reimplementing paper prose.

## N0/N1 Offline Prep

While the HAR GUI-Odyssey full run is still writing its resumable JSONL, build
offline preview inputs without making additional model calls:

```bash
python chorus-n0n1/scripts/prepare_n0n1_inputs.py \
  --results_jsonl related_work/har/outputs/gui_odyssey_paper/full_har_gui_odyssey_20260610.jsonl \
  --output_dir chorus-n0n1/runs/n0n1_inputs/har_gui_odyssey_latest
```

The command writes a step-level table, N0 headroom probe queue, N1 disagreement
reader queue, and `manifest.json`. Before 833 / 1666 episodes it reports
`WAIT_FOR_HALF`; at or after 833 episodes it reports
`READY_FOR_N0N1_OFFLINE_PREP_HALF` if the truncation gate is still valid. It does
not run teacher probes or reader models.
