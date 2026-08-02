import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
MVP_ROOT = ROOT / "runs/collision-law/2026-07-30/w3_assets/MVP"
DATA_ROOT = ROOT / "runs/collision-law/2026-07-30/w3_assets/ScreenSpot-Pro"
CONFIG_PATH = RUN_DIR / "configs/g2_protocol.yaml"
ROSTER_PATH = RUN_DIR / "configs/g1_roster.yaml"
INPUT_SHA256 = "0e6b4387f704b94ec071c8fdb6a381c3293f2bfe8b9ae846b613529c476061b8"
sys.path.insert(0, str(MVP_ROOT))
from mvp_sspro import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    get_top_attention_regions,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def perturbed_indices(regions, seeds, stable_index):
    coverage = np.asarray([region["coverage"] for region in regions], dtype=float)
    output = {}
    for seed in seeds:
        rng = np.random.default_rng(np.random.SeedSequence([seed, stable_index]))
        score = np.log(coverage + 1) + 0.25 * rng.gumbel(size=len(regions))
        order = sorted(range(len(regions)), key=lambda index: (-score[index], index))
        output[str(seed)] = [index + 1 for index in order[:3]]
    return output


def completed_ids(path):
    if not path.exists():
        return set()
    ids = [json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("G2 duplicate resumed region identities")
    return set(ids)


def required_region_indices(perturbations, model, p1_budget):
    perturb_union = {index for selected in perturbations.values() for index in selected}
    base = range(p1_budget) if model == "GTA1-72B" else range(4)
    return sorted(set(base) | perturb_union)


def normalize_existing_manifest(path, p1_budget):
    if not path.exists() or path.stat().st_size == 0:
        return 0
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    changed = 0
    for row in rows:
        expected = {
            model: required_region_indices(row["perturbed_region_indices"], model, p1_budget)
            for model in ("GTA1-72B", "UI-Venus-Ground-72B", "Qwen3.5-122B-A10B")
        }
        if row["required_region_indices_by_model"] != expected:
            row["required_region_indices_by_model"] = expected
            changed += 1
    if changed:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    return changed


def clear_singleton_torchrun_environment():
    if os.environ.get("WORLD_SIZE") == "1" and "LOCAL_RANK" not in os.environ:
        os.environ.pop("WORLD_SIZE", None)
        os.environ.pop("RANK", None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if sha256_file(args.inputs) != INPUT_SHA256:
        raise ValueError("G2 label-free input hash mismatch")
    rows = [json.loads(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    if len(rows) != 1581 or any("bbox" in row or "target_bbox" in row for row in rows):
        raise ValueError("G2 proposer requires complete label-free inputs")
    if args.limit is not None:
        rows = rows[:args.limit]
    protocol = yaml.safe_load(CONFIG_PATH.read_text())
    roster = yaml.safe_load(ROSTER_PATH.read_text())
    model_spec = roster["models"]["GTA1-72B"]
    sensitivity = protocol["proposal_sensitivity"]
    p1_budget = protocol["cells"]["P1"]["selected_budget"]
    clear_singleton_torchrun_environment()
    config = Qwen2_5_VLConfig.from_pretrained(args.model_dir, local_files_only=True)
    config.target_token_id = ","
    config.target_layer_idx = 20
    max_memory = {index: "52GiB" for index in range(torch.cuda.device_count())}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        max_memory=max_memory,
        local_files_only=True,
    ).eval()
    processor = Qwen2_5_VLProcessor.from_pretrained(
        args.model_dir,
        min_pixels=model_spec["min_pixels"],
        max_pixels=model_spec["max_pixels"],
        local_files_only=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    normalized_rows = normalize_existing_manifest(args.output, p1_budget) if args.resume else 0
    if normalized_rows:
        print(json.dumps({"normalized_existing_rows": normalized_rows, "P1_budget": p1_budget}), flush=True)
    completed = completed_ids(args.output) if args.resume else set()
    model_index_hash = sha256_file(args.model_dir / "model.safetensors.index.json")
    protocol_hash = canonical_hash(protocol["proposer"] | {"proposal_sensitivity": sensitivity})
    written = 0
    with args.output.open("a", buffering=1) as output:
        for source in rows:
            if source["id"] in completed:
                continue
            image = Image.open(DATA_ROOT / "images" / source["img_filename"]).convert("RGB")
            ranked, resized_size = get_top_attention_regions(
                image,
                source["instruction"],
                processor,
                model,
                model.device,
                max_regions=sensitivity["source_ranked_regions"],
            )
            if len(ranked) < p1_budget - 1:
                raise ValueError(f"G2 P1 N{p1_budget} requires at least {p1_budget - 1} attention regions: {source['id']}")
            regions = [{
                "region_index": 0,
                "region": [0, 0, image.width, image.height],
                "coverage": 0.0,
                "official_rank": 0,
            }]
            regions.extend({
                "region_index": index,
                "region": list(map(int, item["region"])),
                "coverage": float(item["coverage"]),
                "official_rank": index,
            } for index, item in enumerate(ranked, start=1))
            perturbations = perturbed_indices(regions[1:], sensitivity["seeds"], source["stable_index"])
            required = {
                model: required_region_indices(perturbations, model, p1_budget)
                for model in ("GTA1-72B", "UI-Venus-Ground-72B", "Qwen3.5-122B-A10B")
            }
            artifact = {
                **source,
                "proposer_model": "GTA1-72B",
                "proposer_revision": model_spec["revision"],
                "model_index_sha256": model_index_hash,
                "protocol_sha256": protocol_hash,
                "attention_layer": 20,
                "target_token": ",",
                "resized_size": list(resized_size),
                "regions": regions,
                "perturbed_region_indices": perturbations,
                "required_region_indices_by_model": required,
            }
            artifact["regions_sha256"] = canonical_hash(regions)
            output.write(json.dumps(artifact, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
            written += 1
            print(json.dumps({"written": written, "id": source["id"]}), flush=True)
    print(json.dumps({"status": "PASS", "written": written}), flush=True)


if __name__ == "__main__":
    main()
