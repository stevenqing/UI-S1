import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / "runs/allocation-law/2026-08-01"
COLLISION_DIR = ROOT / "runs/collision-law/2026-07-30"
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(COLLISION_DIR))
from allocation_eval import load_gta1, load_manifest
import w1_run


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def region_contains_bbox(region, bbox):
    return region[0] <= bbox[0] and region[1] <= bbox[1] and region[2] >= bbox[2] and region[3] >= bbox[3]


def region_contains_center(region, bbox):
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return region[0] <= center_x <= region[2] and region[1] <= center_y <= region[3]


def nested_has_key(value, key):
    if isinstance(value, dict):
        return key in value or any(nested_has_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(nested_has_key(item, key) for item in value)
    return False


def load_views_module():
    path = COLLISION_DIR / "w2_infer/views.py"
    spec = importlib.util.spec_from_file_location("allocation_l3_w2_views", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def l3_status():
    identities, aligned_models, pivot = w1_run.load_pool("mind2web", "visual")
    deployable = w1_run.deployable_models(identities, aligned_models, pivot)
    views_module, views_path = load_views_module()
    expected_views = {"full", "v1", "v2", "v3", "v4"}
    if views_module.VIEWS != expected_views or views_module.CROP_FRACTIONS != {"v2": 0.50, "v3": 0.75}:
        raise ValueError("Mind2Web W2 view protocol changed")
    w2 = {}
    artifact_root = COLLISION_DIR / "w2_artifacts/mind2web/tongui-7b"
    for view in ("v1", "v2", "v3", "v4"):
        path = artifact_root / view / "predictions.jsonl"
        rows = read_jsonl(path)
        if len(rows) != 2080 or [row["index"] for row in rows] != list(range(2080)):
            raise ValueError(f"Mind2Web W2 {view} coverage mismatch")
        if any("attention_layer" in row or "target_token" in row for row in rows):
            raise ValueError(f"Mind2Web W2 {view} unexpectedly claims attention provenance")
        w2[view] = {"rows": len(rows), "sha256": sha256_file(path)}
    return {
        "schema_version": 1,
        "status": "BLOCKED_PREREGISTRATION_GAP",
        "stage": "discovery",
        "rows": len(identities),
        "mde": 0.011558476230933909,
        "corrected_label_release": "UNAVAILABLE_VERSIONED_RELEASE",
        "v_only": {
            "required_budget": 12,
            "required": "TongUI-7B full plus 11 attention-proposed crops",
            "available_full_views": 1,
            "available_attention_proposed_crop_views": 0,
            "noncompliant_geometry_views": w2,
            "view_protocol_sha256": sha256_file(views_path),
            "status": "UNAVAILABLE_INCOMPLETE_POOL",
        },
        "mixed": {
            "required": "three deployable lineages x four aligned views",
            "aligned_full_view_models": aligned_models,
            "deployable_full_view_models": deployable,
            "deployable_model_count": len(deployable),
            "lineages_with_four_compliant_views": [],
            "status": "UNAVAILABLE_INCOMPLETE_POOL",
        },
        "lineage_only": {
            "required_full_view_models": 12,
            "aligned_full_view_models": len(aligned_models),
            "deployable_full_view_models": len(deployable),
            "status": "UNAVAILABLE_FEWER_THAN_12",
        },
        "prediction": "NOT_EVALUATED",
        "kill_conditions": {"L-K4": "NOT_EVALUATED"},
    }


def l4_result():
    manifest = load_manifest(RUN_DIR / "raw/shared_regions_n12.jsonl")
    gta1 = load_gta1(ROOT / "runs/ccm-h2h/2026-07-31/h1/shards/top18", manifest)
    if {row["attention_layer"] for row in gta1.values()} != {20}:
        raise ValueError("GTA1 attention layer mismatch")
    if {row["target_token"] for row in gta1.values()} != {","}:
        raise ValueError("GTA1 target token mismatch")
    if {row["model_revision"] for row in gta1.values()} != {"701bedc80b447863bd60e3318ae44f6cbbfafd78"}:
        raise ValueError("GTA1 revision mismatch")

    bbox_hits = []
    center_hits = []
    rank_bbox_hits = [[] for _ in range(12)]
    rank_center_hits = [[] for _ in range(12)]
    for row in gta1.values():
        bbox = row["target_bbox"]
        for rank, candidate in enumerate(row["candidates"][:12]):
            bbox_hit = region_contains_bbox(candidate["region"], bbox)
            center_hit = region_contains_center(candidate["region"], bbox)
            bbox_hits.append(bbox_hit)
            center_hits.append(center_hit)
            rank_bbox_hits[rank].append(bbox_hit)
            rank_center_hits[rank].append(center_hit)

    qwen_config_path = ROOT / "runs/ccm-h2h/2026-07-31/h3/models/Qwen3-VL-8B-Instruct/config.json"
    uitars_config_path = ROOT / "runs/mind2web-tongui/2026-07-28/models/UI-TARS-7B-SFT/config.json"
    qwen_config = json.loads(qwen_config_path.read_text())
    uitars_config = json.loads(uitars_config_path.read_text())
    for name, config in (("Qwen3", qwen_config), ("UI-TARS", uitars_config)):
        if nested_has_key(config, "target_token_id") or nested_has_key(config, "target_layer_idx"):
            raise ValueError(f"{name} config unexpectedly exposes released proposer controls")

    l1 = json.loads((RUN_DIR / "L1_RESULTS.json").read_text())
    e1_accuracy = l1["evaluations"]["v_only"]["12"]["accuracy"]
    return {
        "schema_version": 1,
        "status": "PARTIAL_E1_ONLY_E2_UNAVAILABLE",
        "budget": 12,
        "E1_shared_gta1": {
            "status": "AVAILABLE_REUSED_IDENTICAL_L1_POOL",
            "rows": len(gta1),
            "proposer": {
                "model": "GTA1-7B",
                "revision": "701bedc80b447863bd60e3318ae44f6cbbfafd78",
                "attention_layer": 20,
                "target_token": ",",
                "ordering": "official_attention_rank_prefix_N12",
            },
            "diagnostic": {
                "candidate_count": len(bbox_hits),
                "full_bbox_containment_ratio": sum(bbox_hits) / len(bbox_hits),
                "target_center_containment_ratio": sum(center_hits) / len(center_hits),
                "per_rank_full_bbox_containment": [sum(values) / len(values) for values in rank_bbox_hits],
                "per_rank_target_center_containment": [sum(values) / len(values) for values in rank_center_hits],
            },
            "accuracy": e1_accuracy,
        },
        "E2_native_per_model": {
            "status": "UNAVAILABLE",
            "pooled_scoring": "NOT_RUN",
            "models": {
                "Qwen3-VL-8B-Instruct": {
                    "status": "UNAVAILABLE_NO_RELEASED_NATIVE_PROPOSER",
                    "architecture": qwen_config["architectures"][0],
                    "model_type": qwen_config["model_type"],
                    "config_sha256": sha256_file(qwen_config_path),
                    "target_extraction_fields_present": False,
                },
                "UI-TARS-7B-SFT": {
                    "status": "UNAVAILABLE_UNSUPPORTED_ARCHITECTURE_CONVERSION",
                    "architecture": uitars_config["architectures"][0],
                    "model_type": uitars_config["model_type"],
                    "config_sha256": sha256_file(uitars_config_path),
                    "target_extraction_fields_present": False,
                },
            },
        },
        "ablation_comparison": "NOT_AVAILABLE",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=RUN_DIR)
    args = parser.parse_args()
    l3 = l3_status()
    l4 = l4_result()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "L3_STATUS.json").write_text(json.dumps(l3, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "L4_RESULTS.json").write_text(json.dumps(l4, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "L3": {"status": l3["status"], "v_only": l3["v_only"]["status"], "mixed": l3["mixed"]["status"], "lineage_only": l3["lineage_only"]["status"]},
        "L4": {"status": l4["status"], "E1_accuracy": l4["E1_shared_gta1"]["accuracy"], "E1_diagnostic": l4["E1_shared_gta1"]["diagnostic"], "E2": l4["E2_native_per_model"]["status"]},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
