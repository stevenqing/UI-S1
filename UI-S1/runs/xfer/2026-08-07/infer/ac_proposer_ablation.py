import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[2]
MVP_ROOT = ROOT / "runs/collision-law/2026-07-30/w3_assets/MVP"
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(MVP_ROOT))

from ac_common import COORDINATE_ACTIONS, fold_mapping, image_sha256, load_paired_sample, load_prompt_templates, prompt_text
from mvp_sspro import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from proposer_ablation import generate_multilayer, sha256_file


def completed_ids(path):
    if not path.exists():
        return set()
    ids = [json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate AC proposer ablation ids")
    return set(ids)


def point_hit(region, point):
    return region[0] <= point[0] <= region[2] and region[1] <= point[1] <= region[3]


def summarize(rows, layers):
    report = {}
    for layer in layers:
        by_rank = defaultdict(list)
        for row in rows:
            for rank, value in enumerate(row["layers"][str(layer)]):
                by_rank[rank].append(point_hit(value["region"], row["gt_point"]))
        rates = [sum(by_rank[rank]) / len(by_rank[rank]) for rank in range(12)]
        report[str(layer)] = {
            "rank0_gt_point_containment": rates[0],
            "mean_rank0_to_rank11_gt_point_containment": float(np.mean(rates)),
            "gt_point_containment_by_rank": rates,
        }
    selected = max(
        layers,
        key=lambda layer: (
            report[str(layer)]["rank0_gt_point_containment"],
            report[str(layer)]["mean_rank0_to_rank11_gt_point_containment"],
            -layer,
        ),
    )
    return report, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    roster = yaml.safe_load((RUN_DIR / "configs/xfer_roster.yaml").read_text())
    proposer = roster["androidcontrol"]["proposer"]
    if proposer["selection_status"] != "PENDING_DEV_ABLATION":
        raise ValueError("AC proposer selection is not pending")
    layers = list(proposer["layer_candidates"])
    folds = fold_mapping("low")
    rows = [
        row for row in load_paired_sample()
        if folds[row["episode_id"]] == proposer["dev_fold"]
        and row["gt_action"] in COORDINATE_ACTIONS
    ]
    if len(rows) != 238:
        raise ValueError(f"expected 238 AC dev grounding rows, found {len(rows)}")
    if args.limit is not None:
        rows = rows[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    completed = completed_ids(args.output) if args.resume else set()
    model_spec = next(model for model in roster["androidcontrol"]["models"] if model["id"] == proposer["model"])
    model_dir = ROOT / model_spec["local_path"]
    config = Qwen2_5_VLConfig.from_pretrained(model_dir)
    config.target_token_id = ","
    config.target_layer_idx = 0
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, config=config, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map="cuda:0",
    ).eval()
    processor = Qwen2_5_VLProcessor.from_pretrained(
        model_dir, min_pixels=256 * 28 * 28, max_pixels=1344 * 28 * 28,
    )
    prompts = load_prompt_templates()
    with args.output.open("a", buffering=1) as output:
        for source in rows:
            if source["id"] in completed:
                continue
            sample = source["low"]
            image = Image.open(BytesIO(sample["image"]["bytes"])).convert("RGB")
            response, resized_size, layer_regions = generate_multilayer(
                image, prompt_text("GUI-R1-7B", sample, prompts),
                processor, model, layers, proposer,
            )
            artifact = {
                "id": source["id"],
                "stable_index": source["stable_index"],
                "episode_id": source["episode_id"],
                "setting": "low",
                "source_index": source["low_index"],
                "source_sha256": source["source_low_sha256"],
                "image_sha256": image_sha256(sample),
                "gt_action": sample["gt_action"],
                "gt_point": list(map(float, sample["gt_bbox"][:2])),
                "response": response,
                "resized_size": resized_size,
                "layers": layer_regions,
            }
            output.write(json.dumps(artifact, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
    all_rows = [json.loads(line) for line in args.output.read_text().splitlines() if line.strip()]
    if args.limit is None and len(all_rows) == 238:
        report, selected = summarize(all_rows, layers)
        summary_path = args.output.with_suffix(".summary.json")
        summary_path.write_text(json.dumps({
            "schema_version": 1,
            "status": "PASS",
            "benchmark": "androidcontrol",
            "setting": "low",
            "dev_fold": proposer["dev_fold"],
            "selection_scope": proposer["selection_scope"],
            "rows": len(all_rows),
            "layers": report,
            "selected_layer": selected,
            "query_token": proposer["query_token_candidates"][0],
            "model_revision": model_spec["revision"],
            "model_index_sha256": sha256_file(model_dir / "model.safetensors.index.json"),
            "trace_sha256": sha256_file(args.output),
        }, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "PASS", "selected_layer": selected, "summary": str(summary_path)}, indent=2))
    else:
        print(json.dumps({"status": "PARTIAL", "rows": len(all_rows)}, indent=2))


if __name__ == "__main__":
    main()