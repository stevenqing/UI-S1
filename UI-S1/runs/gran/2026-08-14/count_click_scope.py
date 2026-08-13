import json
from collections import Counter
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
TASK_PATH = ROOT / "runs/xfer/2026-08-07/data/mind2web/mind2web_test_task.jsonl"
MANIFEST_PATH = RUN_DIR / "INPUT_MANIFEST.json"
OUTPUT_PATH = RUN_DIR / "CLICK_SCOPE.json"


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text())
    task_relative = TASK_PATH.relative_to(ROOT).as_posix()
    if (
        manifest.get("status") != "LOCKED_BEFORE_GRAN_LABEL_STATISTICS_AND_TAU_SWEEP"
        or manifest.get("label_statistics_computed") is not False
        or manifest.get("tau_sweep_started") is not False
        or task_relative not in manifest.get("files", {})
    ):
        raise PermissionError("GRAN input manifest is not locked")
    rows = [json.loads(line) for line in TASK_PATH.read_text().splitlines() if line.strip()]
    if len(rows) != 2080 or len({str(row["id"]) for row in rows}) != 2080:
        raise ValueError("GRAN Mind2Web row identity mismatch")
    action_counts = Counter(str(row["step"]["operation"]["op"]) for row in rows)
    click_ids = sorted(str(row["id"]) for row in rows if row["step"]["operation"]["op"] == "CLICK")
    default_bins = 4
    minimum_rows = 400
    bins = default_bins if len(click_ids) // default_bins >= minimum_rows else 3
    projected = [len(click_ids) // bins] * bins
    for index in range(len(click_ids) % bins):
        projected[index] += 1
    if min(projected) < minimum_rows:
        raise ValueError("GRAN CLICK scope cannot satisfy G-K5 after preregistered fallback")
    result = {
        "schema_version": 1,
        "status": "CLICK_SCOPE_LOCKED_BEFORE_P_HAT_OR_MARGIN_ACCESS",
        "input_manifest_sha256": __import__("hashlib").sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        "task_rows": len(rows),
        "action_counts": dict(sorted(action_counts.items())),
        "click_rows": len(click_ids),
        "click_row_ids_sha256": __import__("hashlib").sha256(
            ("\n".join(click_ids) + "\n").encode()
        ).hexdigest(),
        "minimum_rows_per_stratum": minimum_rows,
        "default_strata": default_bins,
        "selected_strata": bins,
        "selected_quantiles": [0.25, 0.5, 0.75] if bins == 4 else [1 / 3, 2 / 3],
        "projected_stratum_rows_before_ties": projected,
        "p_hat_accessed": False,
        "candidate_success_accessed": False,
        "margin_computed": False,
        "tau_sweep_started": False
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()