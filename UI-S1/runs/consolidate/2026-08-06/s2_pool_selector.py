import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(RUN_DIR))

from common import MODELS, SEED, geometry_features, load_context, point_in_bbox, write_json

SOURCEBIAS_DIR = ROOT / "runs/sourcebias/2026-08-03"
sys.path.insert(0, str(SOURCEBIAS_DIR))
from sourcebias_common import b3_select_index, split_ids


ACTION_FOLDS = 3
OUTER_FOLDS = 5
TOP_FRACTION = 0.10
BOOTSTRAPS = 10000


def action_fold(action):
    return action[1] % ACTION_FOLDS


def b3_accuracy(context, actions, row_ids):
    correct = 0
    for row_id in row_ids:
        candidates = [context["bank"][action][row_id] for action in actions]
        selected, _ = b3_select_index(candidates)
        correct += int(point_in_bbox(candidates[selected]["point"], context["metadata"][row_id]["target_bbox"]))
    return correct / len(row_ids)


def action_reliability(context, actions, row_ids):
    output = {}
    for action in actions:
        output[action] = float(np.mean([
            point_in_bbox(context["bank"][action][row_id]["point"], context["metadata"][row_id]["target_bbox"])
            for row_id in row_ids
        ]))
    return output


def feature_vector(context, pool, dev_ids, reliability):
    actions = tuple(tuple(action) for action in pool["actions"])
    geometry = geometry_features(context, actions, dev_ids)
    qualities = np.asarray([reliability[action] for action in actions], dtype=np.float64)
    counts = Counter(action[0] for action in actions)
    return np.asarray([
        geometry["pair_mean"], geometry["pair_std"], geometry["pair_median"], geometry["pair_q90"],
        geometry["largest_cluster_share"], geometry["cross_lineage_mean"], geometry["within_lineage_mean"],
        geometry["cross_within_ratio"], geometry["lineage_count"], geometry["pool_size"],
        *[counts[model] for model in MODELS],
        float(np.mean(qualities)), float(np.min(qualities)), float(np.max(qualities)),
        float(np.max(qualities) - np.partition(qualities, -2)[-2]),
    ], dtype=np.float64)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main():
    d1_path = ROOT / "runs/dominance/2026-08-06/d1_dominance_law.json"
    d1 = json.loads(d1_path.read_text())
    pools = d1["screen_spot"]["pools"]
    context = load_context()
    all_actions = sorted({tuple(action) for pool in pools for action in pool["actions"]})
    records = []
    block_deltas = {"primary": [], "geometry_only": [], "quality_only": []}

    for outer in range(OUTER_FOLDS):
        dev_ids, test_ids = split_ids(context, outer)
        reliability = action_reliability(context, all_actions, dev_ids)
        cache = {}
        for pool in pools:
            actions = tuple(tuple(action) for action in pool["actions"])
            cache[pool["pool_id"]] = {
                "features": feature_vector(context, pool, dev_ids, reliability),
                "dev_target": b3_accuracy(context, actions, dev_ids),
                "test_target": b3_accuracy(context, actions, test_ids),
                "quality_only": float(np.mean([reliability[action] for action in actions])),
                "actions": actions,
            }
        for heldout_action_fold in range(ACTION_FOLDS):
            train = [
                pool for pool in pools
                if all(action_fold(tuple(action)) != heldout_action_fold for action in pool["actions"])
            ]
            test = [
                pool for pool in pools
                if all(action_fold(tuple(action)) == heldout_action_fold for action in pool["actions"])
            ]
            if not train or not test:
                raise ValueError("S2 empty action-disjoint split")
            train_actions = {tuple(action) for pool in train for action in pool["actions"]}
            test_actions = {tuple(action) for pool in test for action in pool["actions"]}
            if train_actions & test_actions:
                raise ValueError("S2 action leakage")

            x_train = np.asarray([cache[pool["pool_id"]]["features"] for pool in train])
            y_train = np.asarray([cache[pool["pool_id"]]["dev_target"] for pool in train])
            x_test = np.asarray([cache[pool["pool_id"]]["features"] for pool in test])
            y_test = np.asarray([cache[pool["pool_id"]]["test_target"] for pool in test])
            model = RandomForestRegressor(
                n_estimators=300, min_samples_leaf=8, max_features=0.8,
                random_state=SEED + 10 * outer + heldout_action_fold, n_jobs=-1,
            )
            model.fit(x_train, y_train)
            primary = model.predict(x_test)
            geometry_columns = 13
            geometry_model = RandomForestRegressor(
                n_estimators=300, min_samples_leaf=8, max_features=0.8,
                random_state=SEED + 100 + 10 * outer + heldout_action_fold, n_jobs=-1,
            )
            geometry_model.fit(x_train[:, :geometry_columns], y_train)
            geometry_only = geometry_model.predict(x_test[:, :geometry_columns])
            quality_only = np.asarray([cache[pool["pool_id"]]["quality_only"] for pool in test])
            top_count = max(1, math.ceil(len(test) * TOP_FRACTION))
            random_mean = float(np.mean(y_test))
            scores = {"primary": primary, "geometry_only": geometry_only, "quality_only": quality_only}
            block = {
                "outer_fold": outer,
                "heldout_action_fold": heldout_action_fold,
                "train_pools": len(train),
                "test_pools": len(test),
                "train_actions": len(train_actions),
                "test_actions": len(test_actions),
                "top_count": top_count,
                "random_pool_mean_B3": random_mean,
                "selectors": {},
            }
            for name, score in scores.items():
                top_indices = np.argsort(score, kind="mergesort")[-top_count:]
                top_mean = float(np.mean(y_test[top_indices]))
                rho = float(spearmanr(score, y_test).statistic)
                delta = top_mean - random_mean
                block_deltas[name].append(delta)
                block["selectors"][name] = {
                    "spearman": rho,
                    "top10_mean_B3": top_mean,
                    "delta_over_random_pool_mean": delta,
                }
            records.append(block)
            print(json.dumps({"outer": outer, "action_fold": heldout_action_fold, "test_pools": len(test)}), flush=True)

    pooled = {
        name: {
            "heldout_spearman_mean": float(np.mean([block["selectors"][name]["spearman"] for block in records])),
            "heldout_spearman_median": float(np.median([block["selectors"][name]["spearman"] for block in records])),
            "top10_delta_summary": summarize(values),
        }
        for name, values in block_deltas.items()
    }
    rng = np.random.default_rng(SEED)
    for name, values in block_deltas.items():
        values = np.asarray(values, dtype=np.float64)
        samples = np.mean(rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True), axis=1)
        pooled[name]["top10_delta_bootstrap"] = {
            "point": float(np.mean(values)),
            "ci_99": [float(np.quantile(samples, 0.005)), float(np.quantile(samples, 0.995))],
            "resamples": BOOTSTRAPS,
            "unit": "outer_fold_x_action_fold_block",
            "seed": SEED,
        }
    success = pooled["primary"]["heldout_spearman_mean"] > 0.7 and pooled["primary"]["top10_delta_bootstrap"]["ci_99"][0] > 0
    result = {
        "schema_version": 1,
        "status": "PASS",
        "split": {
            "outer_application_folds": OUTER_FOLDS,
            "action_folds": ACTION_FOLDS,
            "action_fold_rule": "view_index modulo 3",
            "no_shared_action_between_train_and_test": True,
        },
        "primary_features": "geometry + lineage composition + development member reliability",
        "controls": ["geometry_only", "quality_only", "random_pool_mean"],
        "blocks": records,
        "summary": pooled,
        "criterion": "mean held-out Spearman > 0.7 and top-10% delta 99% CI lower > 0",
        "success": success,
        "claim": "UNLABELED_POOL_SELECTOR_SUPPORTED" if success else "POOL_SELECTOR_NOT_SUPPORTED",
    }
    write_json(RUN_DIR / "s2_pool_selector.json", result)
    print(json.dumps({"success": success, "summary": pooled}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()