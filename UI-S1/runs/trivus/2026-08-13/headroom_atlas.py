import json
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
PRIOR_DIR = RUN_DIR.parent / "2026-08-12"
VUS_DIR = ROOT / "runs/visual-utility-selector/2026-08-11"
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(PRIOR_DIR))

from benchmark_adaptive_adjudication import FAMILIES, family_comparison
from context_common import atomic_json_file, sha256_file
from finalize_trivus import frozen_baselines, load_configs, load_public, merge_outers


OUTPUT_PATH = RUN_DIR / "HEADROOM_ATLAS.json"
POLICY_NAMES = (
    "JOINT3", "TARGET_ONLY", "JOINT2_NO_ANDROID", "NO_VISUAL",
    "RANDOM_ID_PLACEBO",
)


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_candidate_labels(public):
    labels = {}
    vus_manifest_path = VUS_DIR / "data/private_label_folds.manifest.json"
    vus_manifest = json.loads(vus_manifest_path.read_text())
    for fold, item in vus_manifest["folds"].items():
        path = VUS_DIR / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise PermissionError(f"VUS private-label hash mismatch: {fold}")
        rows = load_jsonl(path)
        if len(rows) != item["records"]:
            raise ValueError(f"VUS private-label count mismatch: {fold}")
        for row in rows:
            if row["sample_key"] in labels:
                raise ValueError("Duplicate candidate label")
            labels[row["sample_key"]] = row["candidate_success"]
    android_manifest_path = PRIOR_DIR / "data/PRIVATE_LABEL_MANIFEST.json"
    android_manifest = json.loads(android_manifest_path.read_text())
    for fold, item in android_manifest["folds"].items():
        path = ROOT / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise PermissionError(f"Android private-label hash mismatch: {fold}")
        rows = load_jsonl(path)
        if len(rows) != item["rows"]:
            raise ValueError(f"Android private-label count mismatch: {fold}")
        for row in rows:
            if row["sample_key"] in labels:
                raise ValueError("Duplicate candidate label")
            labels[row["sample_key"]] = row["candidate_success"]
    if set(labels) != set(public):
        raise ValueError("Candidate-label public coverage mismatch")
    for key, values in labels.items():
        expected = 3 if public[key]["benchmark"] == "androidcontrol" else 12
        if (
            not isinstance(values, list)
            or len(values) != expected
            or any(type(value) is not bool for value in values)
        ):
            raise ValueError(f"Candidate-label schema mismatch: {key}")
    return labels, {
        "vus_manifest_sha256": sha256_file(vus_manifest_path),
        "android_manifest_sha256": sha256_file(android_manifest_path),
    }


def union_policy_outputs(outputs, method, public):
    result = {}
    contributors = {}
    for key, row in public.items():
        available = {
            policy: outputs[policy][method][key]
            for policy in POLICY_NAMES
            if key in outputs[policy][method]
        }
        if not available:
            raise ValueError(f"No policy output for {key}")
        result[key] = any(available.values())
        contributors[key] = tuple(sorted(
            policy for policy, success in available.items() if success
        ))
    return result, contributors


def candidate_oracle(labels):
    return {key: any(values) for key, values in labels.items()}


def atlas(outputs, public, strongest, labels, config):
    safe_union, safe_contributors = union_policy_outputs(outputs, "safe", public)
    direct_union, direct_contributors = union_policy_outputs(outputs, "direct", public)
    full_oracle = candidate_oracle(labels)
    layers = {
        "target_only_safe": outputs["TARGET_ONLY"]["safe"],
        "target_only_direct": outputs["TARGET_ONLY"]["direct"],
        "policy_safe_union_oracle": safe_union,
        "policy_direct_union_oracle": direct_union,
        "full_candidate_oracle": full_oracle,
    }
    result = {}
    for family in FAMILIES:
        comparisons = {
            name: family_comparison(
                public, values, strongest, family, "oracle_strongest", config
            )
            for name, values in layers.items()
        }
        ranking_gap = family_comparison(
            public, full_oracle, direct_union, family, "oracle_strongest", config
        )
        result[family] = {
            "layers_vs_strongest": comparisons,
            "unrecovered_candidate_ranking_gap": ranking_gap,
            "successful_policy_contributor_counts": {
                "safe": {
                    policy: sum(
                        policy in safe_contributors[key]
                        for key, row in public.items() if row["benchmark"] == family
                    )
                    for policy in POLICY_NAMES
                },
                "direct": {
                    policy: sum(
                        policy in direct_contributors[key]
                        for key, row in public.items() if row["benchmark"] == family
                    )
                    for policy in POLICY_NAMES
                },
            },
        }
    return result


def main():
    config, training_config = load_configs()
    public = load_public()
    outputs, _ = merge_outers(public)
    _, strongest = frozen_baselines(public, training_config)
    labels, manifests = load_candidate_labels(public)
    result = {
        "schema_version": 1,
        "status": "PASS_POSTHOC_HEADROOM_ATLAS",
        "outcome": "EXPLORATORY_ONLY_NO_PROMOTION",
        "rows": len(public),
        "manifests": manifests,
        "layers": [
            "target_only_safe", "target_only_direct",
            "policy_safe_union_oracle", "policy_direct_union_oracle",
            "full_candidate_oracle",
        ],
        "atlas": atlas(outputs, public, strongest, labels, config),
        "interpretation": {
            "policy_unions_are_label_dependent_oracles": True,
            "full_candidate_oracle_is_label_dependent": True,
            "ranking_gap_identifies_unrecovered_candidate_space": True,
            "confirmatory": False,
            "promotion_allowed": False,
        },
    }
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    atomic_json_file(OUTPUT_PATH, result)
    print(json.dumps({
        family: {
            "full_candidate_headroom": value["layers_vs_strongest"]
            ["full_candidate_oracle"]["equal_cell_family"],
            "policy_direct_union_headroom": value["layers_vs_strongest"]
            ["policy_direct_union_oracle"]["equal_cell_family"],
            "unrecovered_ranking_gap": value["unrecovered_candidate_ranking_gap"]
            ["equal_cell_family"],
        }
        for family, value in result["atlas"].items()
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()