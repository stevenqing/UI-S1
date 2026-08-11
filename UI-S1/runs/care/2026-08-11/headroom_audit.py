import json
from pathlib import Path

import numpy as np
from PIL import Image


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
VUS = ROOT / "runs/visual-utility-selector/2026-08-11"
ARMS = ("C_uni", "C_cond", "C_rand", "C_self")
BENCHMARKS = ("mind2web", "screenspot_pro")


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rate(values):
    return sum(values) / len(values)


def row_summary(rows):
    passed = sum(row[0] for row in rows)
    direct = sum(row[1] for row in rows)
    safe = sum(row[2] for row in rows)
    return {
        "arm_rows": len(rows),
        "pass_rate": passed / len(rows),
        "direct_accuracy": direct / len(rows),
        "safe_accuracy": safe / len(rows),
        "direct_recall_given_any_correct": direct / passed if passed else None,
        "ranking_failure_rate_given_coverage": (passed - direct) / passed if passed else None,
    }


def main():
    public = {row["sample_key"]: row for row in load_jsonl(VUS / "data/public_records.jsonl")}
    labels = {}
    for fold in range(5):
        labels.update({
            row["sample_key"]: row["candidate_success"]
            for row in load_jsonl(VUS / f"data/private_labels_fold-{fold}.jsonl")
        })
    result = json.loads((VUS / "set_ranker_adjudication.json").read_text())

    mind_rows = {
        row["id"]: row
        for row in load_jsonl(ROOT / "runs/xfer/2026-08-07/data/mind2web/mind2web_test_task.jsonl")
    }
    screen_rows = {
        row["id"]: row
        for row in load_jsonl(ROOT / "runs/allocation-law/2026-08-01/raw/shared_regions_n12.jsonl")
    }
    target_area = {benchmark: {} for benchmark in BENCHMARKS}
    for row_id, row in mind_rows.items():
        width, height = Image.open(ROOT / row["image"]).size
        bbox = row["step"]["bbox"]
        target_area["mind2web"][row_id] = bbox["width"] * bbox["height"] / (width * height)
    for row_id, row in screen_rows.items():
        width, height = row["img_size"]
        left, top, right, bottom = row["target_bbox"]
        target_area["screenspot_pro"][row_id] = (right - left) * (bottom - top) / (width * height)

    output = {"schema_version": 1, "status": "PASS_DIAGNOSTIC", "benchmarks": {}}
    for benchmark in BENCHMARKS:
        benchmark_result = {"arms": {}}
        row_ids = sorted(result["outputs"][benchmark]["C_uni"]["safe"])
        shared_first_six = []
        for row_id in row_ids:
            blocks = [public[f"{benchmark}/{arm}/{row_id}"]["candidates"][:6] for arm in ARMS]
            shared_first_six.append(all(block == blocks[0] for block in blocks[1:]))

        stage1_pass = []
        union_pass = []
        safe_by_arm = {}
        pass_by_arm = {}
        direct_by_arm = {}
        fallback_by_arm = {}
        area_rows = [[] for _ in range(4)]
        correct_count_rows = {"1": [], "2-3": [], "4-6": [], "7-12": []}
        cuts = np.quantile(list(target_area[benchmark].values()), [0.25, 0.5, 0.75])

        for arm in ARMS:
            methods = result["outputs"][benchmark][arm]
            safe_by_arm[arm] = [bool(methods["safe"][row_id]) for row_id in row_ids]
            direct_by_arm[arm] = [bool(methods["direct"][row_id]) for row_id in row_ids]
            fallback_by_arm[arm] = [bool(methods["fallback"][row_id]) for row_id in row_ids]
            pass_by_arm[arm] = []
            for index, row_id in enumerate(row_ids):
                success = labels[f"{benchmark}/{arm}/{row_id}"]
                passed = any(success)
                pass_by_arm[arm].append(passed)
                area_bucket = int(np.searchsorted(cuts, target_area[benchmark][row_id], side="right"))
                area_rows[area_bucket].append((passed, direct_by_arm[arm][index], safe_by_arm[arm][index]))
                correct = sum(success)
                key = "1" if correct == 1 else "2-3" if correct <= 3 else "4-6" if correct <= 6 else "7-12"
                correct_count_rows[key].append((passed, direct_by_arm[arm][index], safe_by_arm[arm][index]))

        for index, row_id in enumerate(row_ids):
            stage1_pass.append(any(labels[f"{benchmark}/C_uni/{row_id}"][:6]))
            union_pass.append(any(pass_by_arm[arm][index] for arm in ARMS))

        for arm in ARMS:
            candidate_oracle = pass_by_arm[arm]
            pair_oracle = [
                fallback_by_arm[arm][index] or direct_by_arm[arm][index]
                for index in range(len(row_ids))
            ]
            benchmark_result["arms"][arm] = {
                "fallback": rate(fallback_by_arm[arm]),
                "direct": rate(direct_by_arm[arm]),
                "safe": rate(safe_by_arm[arm]),
                "pass_at_12": rate(candidate_oracle),
                "pair_oracle": rate(pair_oracle),
                "candidate_ranking_gap": rate(candidate_oracle) - rate(pair_oracle),
                "gate_gap": rate(pair_oracle) - rate(safe_by_arm[arm]),
            }

        keys = ("fallback", "direct", "safe", "pass_at_12", "pair_oracle", "candidate_ranking_gap", "gate_gap")
        benchmark_result["equal_arm"] = {
            key: float(np.mean([benchmark_result["arms"][arm][key] for arm in ARMS]))
            for key in keys
        }
        benchmark_result["acquisition"] = {
            "shared_first6_fraction": rate(shared_first_six),
            "stage1_pass_at_6": rate(stage1_pass),
            "pass_at_12_per_arm": {arm: rate(pass_by_arm[arm]) for arm in ARMS},
            "best_static_pass_at_12": max(rate(pass_by_arm[arm]) for arm in ARMS),
            "union_pass_at_48": rate(union_pass),
            "oracle_route_coverage_gain_over_best_arm": rate(union_pass) - max(rate(pass_by_arm[arm]) for arm in ARMS),
        }
        benchmark_result["evidence"] = {
            "target_area_quantile_cuts": cuts.tolist(),
            "by_target_area_quartile": {str(index): row_summary(rows) for index, rows in enumerate(area_rows)},
            "by_correct_candidate_count": {key: row_summary(rows) for key, rows in correct_count_rows.items()},
        }
        output["benchmarks"][benchmark] = benchmark_result

    mind = output["benchmarks"]["mind2web"]
    screen = output["benchmarks"]["screenspot_pro"]
    output["diagnostic_gates"] = {
        "D1_candidate_ranking_gap_above_MDE_both": (
            mind["equal_arm"]["candidate_ranking_gap"] > 0.006106589385659482
            and screen["equal_arm"]["candidate_ranking_gap"] > 0.007
        ),
        "D2_first_six_shared_all_rows": (
            mind["acquisition"]["shared_first6_fraction"] == 1.0
            and screen["acquisition"]["shared_first6_fraction"] == 1.0
        ),
        "D3_oracle_route_coverage_gain_above_MDE_both": (
            mind["acquisition"]["oracle_route_coverage_gain_over_best_arm"] > 0.006106589385659482
            and screen["acquisition"]["oracle_route_coverage_gain_over_best_arm"] > 0.007
        ),
        "D4_small_target_failure_exceeds_large_by_10pp_both": (
            mind["evidence"]["by_target_area_quartile"]["0"]["ranking_failure_rate_given_coverage"]
            - mind["evidence"]["by_target_area_quartile"]["3"]["ranking_failure_rate_given_coverage"] > 0.10
            and screen["evidence"]["by_target_area_quartile"]["0"]["ranking_failure_rate_given_coverage"]
            - screen["evidence"]["by_target_area_quartile"]["3"]["ranking_failure_rate_given_coverage"] > 0.10
        ),
        "D5_unique_correct_direct_recall_below_half_both": (
            mind["evidence"]["by_correct_candidate_count"]["1"]["direct_recall_given_any_correct"] < 0.5
            and screen["evidence"]["by_correct_candidate_count"]["1"]["direct_recall_given_any_correct"] < 0.5
        ),
    }
    if not all(output["diagnostic_gates"].values()):
        raise ValueError(f"CARE diagnostic gate failed: {output['diagnostic_gates']}")
    (RUN_DIR / "headroom_audit.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": output["status"], "diagnostic_gates": output["diagnostic_gates"], "equal_arm": {benchmark: output["benchmarks"][benchmark]["equal_arm"] for benchmark in BENCHMARKS}, "acquisition": {benchmark: output["benchmarks"][benchmark]["acquisition"] for benchmark in BENCHMARKS}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
