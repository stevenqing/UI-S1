import hashlib
import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
OUTPUT_PATH = RUN_DIR / "INPUT_MANIFEST.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_file(output, path, role):
    path = Path(path).resolve()
    relative = path.relative_to(ROOT.resolve()).as_posix()
    if relative in output:
        output[relative]["roles"] = sorted(set(output[relative]["roles"]) | {role})
        return
    if not path.is_file():
        raise FileNotFoundError(path)
    output[relative] = {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "roles": [role],
    }


def add_glob(output, pattern, role, minimum=1):
    paths = sorted(ROOT.glob(pattern))
    if len(paths) < minimum:
        raise FileNotFoundError(f"GRAN input glob is empty: {pattern}")
    for path in paths:
        add_file(output, path, role)
    return paths


def mind2web_inputs(output):
    task_path = ROOT / "runs/xfer/2026-08-07/data/mind2web/mind2web_test_task.jsonl"
    add_file(output, task_path, "mind2web_task_rows")
    add_file(
        output,
        ROOT / "runs/complementarity/2026-07-30/folds.json",
        "mind2web_website_fold_map",
    )
    for stage in ("stage1", "stage1/view1", "stage2"):
        for model in ("tongui", "cogagent", "uitars"):
            add_glob(
                output,
                f"runs/xfer/2026-08-07/raw/{stage}/{model}/shard-*.jsonl",
                f"mind2web_{stage.replace('/', '_')}_{model}",
            )
    rows = [json.loads(line) for line in task_path.read_text().splitlines() if line.strip()]
    if len(rows) != 2080 or len({str(row["id"]) for row in rows}) != 2080:
        raise ValueError("GRAN Mind2Web task identity mismatch")
    image_paths = sorted({str(row["image"]) for row in rows})
    for value in image_paths:
        add_file(output, ROOT / value, "mind2web_image_for_dimensions")
    return {"task_rows": len(rows), "unique_images": len(image_paths)}


def screenspot_inputs(output):
    add_file(
        output,
        ROOT / "runs/allocation-law/2026-08-01/raw/shared_regions_n12.jsonl",
        "screenspot_region_manifest",
    )
    add_file(
        output,
        ROOT / "runs/allocation-law/2026-08-01/configs/l1_pools.yaml",
        "screenspot_pool_units",
    )
    add_glob(
        output,
        "runs/ccm-h2h/2026-07-31/h1/shards/top18/shard-*.jsonl",
        "screenspot_gta1_views_0_15",
    )
    for model, directory, prefix in (
        ("qwen3", "qwen3_views", "qwen3-views-4-11"),
        ("uitars", "uitars_views", "uitars-views-4-11"),
    ):
        add_glob(
            output,
            f"runs/ccm-h2h/2026-07-31/h3/shards/{directory}/shard-*.jsonl",
            f"screenspot_{model}_views_0_3",
        )
        add_glob(
            output,
            f"runs/allocation-law/2026-08-01/shards/{prefix}-*.jsonl",
            f"screenspot_{model}_views_4_11",
        )
    add_file(
        output,
        ROOT / "runs/consolidate/2026-08-06/raw/q1_regions.jsonl",
        "screenspot_four_arm_regions",
    )
    for directory in ("q1-gta1", "q1-qwen3", "q1-uitars"):
        add_glob(
            output,
            f"runs/consolidate/2026-08-06/raw/{directory}/shard-*.jsonl",
            f"screenspot_four_arm_{directory}",
        )


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    files = {}
    mind = mind2web_inputs(files)
    screenspot_inputs(files)
    role_counts = {}
    for item in files.values():
        for role in item["roles"]:
            role_counts[role] = role_counts.get(role, 0) + 1
    manifest = {
        "schema_version": 1,
        "status": "LOCKED_BEFORE_GRAN_LABEL_STATISTICS_AND_TAU_SWEEP",
        "round": "gran",
        "gpu_used": False,
        "scorer_or_evaluator_imported": False,
        "label_statistics_computed": False,
        "tau_sweep_started": False,
        "mind2web": mind,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files.values()),
        "role_counts": dict(sorted(role_counts.items())),
        "files": dict(sorted(files.items())),
    }
    OUTPUT_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if json.loads(OUTPUT_PATH.read_text()) != manifest:
        raise ValueError("GRAN input manifest readback mismatch")
    print(json.dumps({
        "status": manifest["status"],
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "mind2web": manifest["mind2web"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()