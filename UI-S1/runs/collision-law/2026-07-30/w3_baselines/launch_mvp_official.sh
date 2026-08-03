#!/usr/bin/env bash
set -euo pipefail
run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$run_dir/../../../.." && pwd)"
python="$workspace/runs/mind2web-tongui/2026-07-28/.venv/bin/python"
asset_root="$run_dir/../w3_assets"
source_root="$asset_root/MVP"
overlay="$asset_root/mvp-overlay"
model="$asset_root/GTA1-7B"
data="$asset_root/ScreenSpot-Pro"
cell="$run_dir/../w3_artifacts/mvp_official_gta1_screenspot_pro"
runtime_patch="$run_dir/mvp_collective_timeout.patch"
[[ -x "$python" && -d "$source_root/.git" && -d "$overlay/transformers" ]] || { echo "missing MVP runtime" >&2; exit 2; }
if grep -Fq 'timeout=timedelta(hours=1)' "$source_root/mvp_sspro.py"; then
  :
elif git -C "$source_root" apply --check "$runtime_patch"; then
  git -C "$source_root" apply "$runtime_patch"
else
  echo "MVP source does not match the pinned collective-timeout patch" >&2
  exit 4
fi
mkdir -p "$cell"
export PYTHONPATH="$source_root:$overlay${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -f "$cell/source_result.json" ]]; then
  pushd "$cell" >/dev/null
  "$python" -m torch.distributed.run --standalone --nproc_per_node=4 "$source_root/mvp_sspro.py" \
    --attn_layer 20 --target_token_id ',' --max_inferences 4 --batch_size 1 --num_workers 0 \
    --json_file_dir "$data/annotations" --base_image_dir "$data/images" --model_path "$model"
  mapfile -t outputs < <(find sspro_rst -maxdepth 1 -name '*.json' -type f)
  [[ "${#outputs[@]}" -eq 1 ]] || { echo "expected one official MVP output" >&2; exit 3; }
  mv "${outputs[0]}" source_result.json
  popd >/dev/null
fi
PYTHONPATH="$run_dir:$PYTHONPATH" "$python" "$run_dir/score_mvp_official.py" \
  --input "$cell/source_result.json" --output "$cell/score.json"