import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2", type=Path, required=True)
    parser.add_argument("--w1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    e2 = json.loads(args.e2.read_text())
    w1 = json.loads(args.w1.read_text())
    result = {
        "status": "PASS",
        "contract": {
            "e2_scope": "disagreement_pool_nested_grouped_test_folds",
            "absolute_full_pool_delta": "pooled routed gain steps / full clean rows",
            "a3_scope": "W1 deployable grouped folds",
            "warning": "reranker runs all models and is compared against equal-compute A3, not single-model compute",
        },
        "pools": {},
    }
    for pool, values in e2["pools"].items():
        t2 = values["aggregate"]["T2"]["pooled"]
        a3 = w1["scopes"]["deployable"][pool]["aggregate"]["A3_pka_joint"]
        best = w1["scopes"]["deployable"][pool]["aggregate"]["A0_heldout_best"]
        result["pools"][pool] = {
            "e2_t2_headroom_capture": t2["headroom_capture"],
            "e2_routed_gain_steps": t2["routed_gain_steps"],
            "e2_projected_full_pool_delta": t2["projected_full_pool_delta"],
            "w1_a3_step_sr": a3,
            "w1_heldout_best_step_sr": best,
            "w1_a3_delta": a3 - best,
        }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["pools"], indent=2))


if __name__ == "__main__":
    main()