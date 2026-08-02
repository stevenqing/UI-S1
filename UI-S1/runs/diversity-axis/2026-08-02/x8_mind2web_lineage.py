import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
COLLISION_DIR = ROOT / "runs/collision-law/2026-07-30"
sys.path.insert(0, str(COLLISION_DIR))
import w1_run


FAMILY = {
    "cogagent-18b": "CogAgent",
    "tongui-32b": "TongUI",
    "tongui-3b": "TongUI",
    "tongui-7b": "TongUI",
    "ui-tars-72b": "UI-TARS",
    "ui-tars-7b": "UI-TARS",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identities, models, pivot = w1_run.load_pool("mind2web", "visual")
    deployable = w1_run.deployable_models(identities, models, pivot)
    unknown = set(deployable) - set(FAMILY)
    if unknown:
        raise ValueError(f"X8 missing frozen family mapping: {sorted(unknown)}")
    counts = Counter(FAMILY[model] for model in deployable)
    result = {
        "schema_version": 1,
        "status": "BLOCKED_NO_SAME_LINEAGE_N6",
        "rows": len(identities),
        "aligned_full_view_models": models,
        "aligned_model_count": len(models),
        "deployable_models": deployable,
        "deployable_model_count": len(deployable),
        "deployable_family_counts": dict(sorted(counts.items())),
        "required_budget": 6,
        "same_lineage_maximum": max(counts.values()),
        "cross_lineage_N6_available": len(deployable) >= 6,
        "same_lineage_N6_available": max(counts.values()) >= 6,
        "comparison": "NOT_EVALUATED",
        "original_L3_modified": False,
        "L_K4_modified": False,
        "forbidden_substitutions": ["model_duplication", "non_deployable_model", "view_reuse"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()