import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from xf_mind2web import score_prediction
from xfer_common import aggregate


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]


def perturbation_order(regions, seed, stable_index, selected=15):
    coverage = np.asarray([region["coverage"] for region in regions], dtype=np.float64)
    rng = np.random.default_rng(np.random.SeedSequence([seed, stable_index]))
    scores = np.log(coverage + 1) + 0.25 * rng.gumbel(size=len(regions))
    return sorted(range(len(regions)), key=lambda index: (-scores[index], index))[:selected]


def estimate_mde(rows, regions_by_id, full_predictions, view_predictions, seeds):
    details = []
    for seed in seeds:
        seed_success = {}
        for row in rows:
            source = regions_by_id[row["id"]]
            selected = perturbation_order(source["regions"], seed, row["stable_index"])
            candidates = [{**full_predictions[row["id"]], "model": "TongUI-7B"}] + [
                {**view_predictions[region_index + 1][row["id"]], "model": "TongUI-7B"}
                for region_index in selected
            ]
            image_size = Image.open(ROOT / row["image"]).size
            prediction = aggregate(candidates, ["TongUI-7B"], image_size)
            seed_success[row["id"]] = score_prediction(row, prediction, image_size)
        micro = float(np.mean(list(seed_success.values())))
        episodes = {}
        for row in rows:
            episodes.setdefault(row["episode_id"], []).append(seed_success[row["id"]])
        episode_macro = float(np.mean([np.mean(values) for values in episodes.values()]))
        details.append({
            "seed": seed,
            "micro_step_sr": micro,
            "episode_macro_step_sr": episode_macro,
        })
    micro_sd = statistics.stdev(value["micro_step_sr"] for value in details)
    macro_sd = statistics.stdev(value["episode_macro_step_sr"] for value in details)
    return {
        "seeds": details,
        "micro_sample_sd": micro_sd,
        "micro_mde": 2 * micro_sd,
        "episode_macro_sample_sd": macro_sd,
        "episode_macro_mde": 2 * macro_sd,
    }


def load_unique(directory):
    rows = {}
    for path in sorted(directory.glob("shard-*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError(f"duplicate row: {row['id']}")
            rows[row["id"]] = row
    if len(rows) != 2080:
        raise ValueError(f"expected 2,080 rows in {directory}, found {len(rows)}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposer-regions", type=Path, required=True)
    parser.add_argument("--full-lane", type=Path, required=True)
    parser.add_argument("--view-lane", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load((RUN_DIR / "configs/mde.yaml").read_text())
    if config["status"] != "RESULT_BLIND_BEFORE_MDE_SCORING":
        raise ValueError("MDE protocol is not frozen")
    rows = [json.loads(line) for line in (RUN_DIR / "data/mind2web/mind2web_test_task.jsonl").read_text().splitlines() if line.strip()]
    regions = load_unique(args.proposer_regions)
    full_rows = load_unique(args.full_lane)
    view_rows = load_unique(args.view_lane)
    full_predictions = {row_id: value["prediction"] for row_id, value in full_rows.items()}
    view_predictions = {}
    for view_index in range(1, 17):
        set_name = f"view{view_index}"
        values = {}
        for row_id, value in view_rows.items():
            if value["source_hashes"][set_name] != regions[row_id]["regions_sha256"]:
                raise ValueError(f"MDE proposer provenance mismatch: {row_id}/{set_name}")
            values[row_id] = value["predictions"][set_name][0]["prediction"]
        view_predictions[view_index] = values
    if set(regions) != {row["id"] for row in rows} or set(full_predictions) != set(regions):
        raise ValueError("MDE identity mismatch")
    if any(set(values) != set(regions) for values in view_predictions.values()):
        raise ValueError("MDE view-prediction coverage mismatch")
    result = {
        "schema_version": 1,
        "status": "PASS",
        "protocol": "configs/mde.yaml",
        "benchmark": "mind2web",
        "rows": len(rows),
        **estimate_mde(rows, regions, full_predictions, view_predictions, config["perturbation"]["seeds"]),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def test_contracts():
    assert (ROOT / "runs/xfer/2026-08-07").resolve() == RUN_DIR.resolve()
    regions = [{"coverage": value} for value in range(16, 0, -1)]
    first = perturbation_order(regions, 1, 0)
    second = perturbation_order(regions, 1, 0)
    assert first == second and len(first) == 15 and len(set(first)) == 15
    assert all(0 <= index < 16 for index in first)


if __name__ == "__main__":
    test_contracts()
    if len(__import__("sys").argv) > 1:
        main()
    else:
        print(json.dumps({"status": "PASS"}))