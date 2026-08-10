import json
from pathlib import Path

import numpy as np
import yaml

from lsa_common import load_rows, reliability_statistics, fallback_index


RUN_DIR = Path(__file__).resolve().parent


def main():
    config = yaml.safe_load((RUN_DIR / "configs/lsa_prereg.yaml").read_text())
    if config["status"] != "FROZEN_BEFORE_LSA_RESULTS":
        raise ValueError("LSA preregistration is not frozen")
    banks = load_rows()
    result = {"schema_version": 1, "status": "PASS", "benchmarks": {}}
    for benchmark, rows in banks.items():
        ids = sorted(rows)
        sums, counts = reliability_statistics(rows, ids)
        positives = [sum(candidate.success for candidate in rows[row_id].candidates) for row_id in ids]
        fallback = [rows[row_id].candidates[fallback_index(rows[row_id], sums, counts)].success for row_id in ids]
        result["benchmarks"][benchmark] = {
            "rows": len(ids),
            "candidates_per_row": sorted({len(rows[row_id].candidates) for row_id in ids}),
            "oracle_pass_at_12": float(np.mean([value > 0 for value in positives])),
            "fallback_accuracy_in_sample_diagnostic": float(np.mean(fallback)),
            "mixed_label_rows": sum(0 < value < len(rows[ids[index]].candidates) for index, value in enumerate(positives)),
            "all_negative_rows": sum(value == 0 for value in positives),
            "all_positive_rows": sum(value == len(rows[ids[index]].candidates) for index, value in enumerate(positives)),
            "mean_successful_candidates": float(np.mean(positives)),
        }
    result["LSA_K1"] = any(value["mixed_label_rows"] == 0 or value["candidates_per_row"] != [12] for value in result["benchmarks"].values())
    (RUN_DIR / "lsa_oracle.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()