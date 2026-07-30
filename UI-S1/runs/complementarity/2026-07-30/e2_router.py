import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from threadpoolctl import threadpool_limits

from common import auc_roc, fold_for, load_rows, micro, pivot_rows, split_identities


POOLS = (("androidcontrol", "low"), ("androidcontrol", "high"), ("mind2web", "visual"))


class ConstantHead:
    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, values):
        positive = np.full(len(values), self.probability)
        return np.column_stack([1 - positive, positive])


def hash_text(values, features):
    vectorizer = HashingVectorizer(
        n_features=features, analyzer="char_wb", ngram_range=(3, 5),
        alternate_sign=True, norm="l2",
    )
    return vectorizer.transform(values).toarray().astype(np.float32)


def agreement_features(model_rows, selected_models):
    valid = [model for model in selected_models if model_rows[model]["parse_ok"] and model_rows[model]["pred_action"]]
    if not valid:
        return [0.0, 10.0, 0.0, 0.0, 0.0]
    counts = Counter(model_rows[model]["pred_action"] for model in valid)
    votes = sorted(counts.values(), reverse=True)
    margin = (votes[0] - (votes[1] if len(votes) > 1 else 0)) / len(valid)
    winning = sorted(counts, key=lambda action: (-counts[action], action))[0]
    coordinates = [
        (model_rows[model]["pred_x"], model_rows[model]["pred_y"])
        for model in valid if model_rows[model]["pred_action"] == winning
        and not math.isnan(model_rows[model]["pred_x"]) and not math.isnan(model_rows[model]["pred_y"])
    ]
    distances = [math.dist(left, right) for index, left in enumerate(coordinates) for right in coordinates[index + 1:]]
    dispersion = float(np.median(distances)) / 0.14 if distances else 0.0
    agreements = [
        float(model_rows[left]["pred_action"] == model_rows[right]["pred_action"])
        for index, left in enumerate(selected_models) for right in selected_models[index + 1:]
    ]
    return [
        margin, dispersion, float(np.mean(agreements)) if agreements else 1.0,
        float(np.std(agreements)) if agreements else 0.0, len(counts) / len(valid),
    ]


def build_features(identities, models, pivot, tier, selected_models):
    references = [next(iter(pivot[row_id].values())) for row_id in identities]
    instruction_hash = hash_text([row["instruction"] for row in references], 64)
    input_numeric = np.asarray([
        [
            row["history_len"], len(row["instruction"]), len(row["history"]),
            row["image_width"], row["image_height"], row["image_gray_mean"],
            row["image_gray_std"], row["image_entropy"], row["image_edge_density"],
        ]
        for row in references
    ], dtype=np.float32)
    arrays = [input_numeric, instruction_hash]
    names = [
        "input/history_len", "input/instruction_len", "input/history_chars", "input/image_width",
        "input/image_height", "input/image_gray_mean", "input/image_gray_std", "input/image_entropy",
        "input/image_edge_density",
    ] + [f"instruction_hash/{index}" for index in range(64)]
    blocks = {"input_numeric": list(range(0, 9)), "instruction_embedding": list(range(9, 73))}
    if tier in {"T1", "T2"}:
        action_vocab = sorted({pivot[row_id][model]["pred_action"] for row_id in identities for model in models})
        output_start = sum(array.shape[1] for array in arrays)
        for model in models:
            numeric = []
            raw_values = []
            for row_id in identities:
                row = pivot[row_id][model]
                one_hot = [float(row["pred_action"] == action) for action in action_vocab]
                numeric.append(one_hot + [
                    float(row["parse_ok"]),
                    -1.0 if math.isnan(row["pred_x"]) else row["pred_x"],
                    -1.0 if math.isnan(row["pred_y"]) else row["pred_y"],
                    len(row["pred_param"]), len(row["pred_raw"]),
                ])
                raw_values.append(row["pred_raw"])
            numeric_array = np.asarray(numeric, dtype=np.float32)
            raw_hash = hash_text(raw_values, 16)
            arrays.extend([numeric_array, raw_hash])
            names.extend(
                [f"outputs/{model}/action={action}" for action in action_vocab]
                + [f"outputs/{model}/parse_ok", f"outputs/{model}/x", f"outputs/{model}/y",
                   f"outputs/{model}/param_len", f"outputs/{model}/raw_len"]
                + [f"outputs/{model}/raw_hash_{index}" for index in range(16)]
            )
        blocks["model_outputs"] = list(range(output_start, sum(array.shape[1] for array in arrays)))
    if tier == "T2":
        consistency_start = sum(array.shape[1] for array in arrays)
        consistency = np.asarray([
            agreement_features(pivot[row_id], selected_models) for row_id in identities
        ], dtype=np.float32)
        arrays.append(consistency)
        names.extend([
            "consistency/vote_margin", "consistency/geometric_dispersion",
            "consistency/pairwise_action_agreement_mean", "consistency/pairwise_action_agreement_std",
            "consistency/distinct_action_rate",
        ])
        blocks["model_consistency"] = list(range(consistency_start, consistency_start + 5))
    values = np.concatenate(arrays, axis=1)
    values = np.nan_to_num(values, nan=-1.0, posinf=1e6, neginf=-1e6)
    return values, names, blocks


def train_heads(train_x, train_ids, models, pivot):
    heads = {}
    for model in models:
        labels = np.asarray([pivot[row_id][model]["success"] for row_id in train_ids], dtype=np.int8)
        if len(set(labels)) == 1:
            heads[model] = ConstantHead(float(labels[0]))
        else:
            head = HistGradientBoostingClassifier(
                learning_rate=0.1, max_iter=20, max_leaf_nodes=15,
                min_samples_leaf=20, l2_regularization=1.0,
                early_stopping=False, random_state=20260730,
            )
            with threadpool_limits(limits=1):
                head.fit(train_x, labels)
            heads[model] = head
    return heads


def predict_matrix(heads, values, models):
    return np.column_stack([heads[model].predict_proba(values)[:, 1] for model in models])


def routed_success(probabilities, identities, models, pivot, best_single, threshold=None):
    choices = np.argmax(probabilities, axis=1)
    successes = []
    for index, row_id in enumerate(identities):
        model = models[choices[index]]
        if threshold is not None and probabilities[index, choices[index]] < threshold:
            model = best_single
        successes.append(bool(pivot[row_id][model]["success"]))
    return successes


def tune_threshold(probabilities, identities, models, pivot, best_single):
    candidates = []
    for threshold in np.linspace(0, 1, 101):
        score = micro(routed_success(probabilities, identities, models, pivot, best_single, threshold))
        candidates.append((score, -threshold, threshold))
    return max(candidates)[2]


def permutation_importance_by_block(head, baseline_auc, values, labels, blocks, seed):
    if baseline_auc is None or isinstance(head, ConstantHead):
        return {block: None for block in blocks}
    rng = np.random.default_rng(seed)
    output = {}
    for block, indices in blocks.items():
        permuted = values.copy()
        order = rng.permutation(len(values))
        permuted[:, indices] = permuted[order][:, indices]
        permuted_auc = auc_roc(labels, head.predict_proba(permuted)[:, 1])
        output[block] = baseline_auc - permuted_auc if permuted_auc is not None else None
    return output


def run_pool(bench, setting, e1):
    rows = load_rows(bench, setting)
    identities, models, pivot = pivot_rows(rows)
    disagreement = [
        row_id for row_id in identities
        if 0 < sum(pivot[row_id][model]["success"] for model in models) < len(models)
    ]
    pool = f"{bench}/{setting}"
    e1_folds = {fold["fold"]: fold for fold in e1["pools"][pool]["folds"]}
    folds = []
    for test_fold in range(5):
        outer_train_ids, test_ids = split_identities(pool, disagreement, pivot, test_fold)
        dev_fold = (test_fold + 1) % 5
        dev_ids = [
            row_id for row_id in outer_train_ids
            if fold_for(pool, next(iter(pivot[row_id].values()))["group_key"]) == dev_fold
        ]
        dev_set = set(dev_ids)
        train_ids = [row_id for row_id in outer_train_ids if row_id not in dev_set]
        if not train_ids or not dev_ids or not test_ids:
            raise ValueError(f"empty nested split for {pool}/fold-{test_fold}")
        best_single = max(models, key=lambda model: (sum(pivot[row_id][model]["success"] for row_id in dev_ids), model))
        oracle_test = micro(any(pivot[row_id][model]["success"] for model in models) for row_id in test_ids)
        if oracle_test != 1.0:
            raise ValueError("disagreement-pool oracle must be one")
        fold_result = {
            "fold": test_fold, "train_folds": sorted(set(range(5)) - {test_fold, dev_fold}),
            "dev_fold": dev_fold, "test_fold": test_fold,
            "train_rows": len(train_ids), "dev_rows": len(dev_ids), "test_rows": len(test_ids),
            "best_single_selected_on_dev": best_single, "tiers": {},
        }
        for tier in ("T0", "T1", "T2"):
            selected = e1_folds[test_fold]["selected_models"]
            all_ids = train_ids + dev_ids + test_ids
            values, feature_names, blocks = build_features(all_ids, models, pivot, tier, selected)
            train_end = len(train_ids)
            dev_end = train_end + len(dev_ids)
            train_x, dev_x, test_x = values[:train_end], values[train_end:dev_end], values[dev_end:]
            heads = train_heads(train_x, train_ids, models, pivot)
            dev_probabilities = predict_matrix(heads, dev_x, models)
            test_probabilities = predict_matrix(heads, test_x, models)
            threshold = tune_threshold(dev_probabilities, dev_ids, models, pivot, best_single)
            plain = routed_success(test_probabilities, test_ids, models, pivot, best_single)
            abstain = routed_success(test_probabilities, test_ids, models, pivot, best_single, threshold)
            best_score = micro(pivot[row_id][best_single]["success"] for row_id in test_ids)
            headroom = oracle_test - best_score
            heads_report = {}
            for model_index, model in enumerate(models):
                labels = [pivot[row_id][model]["success"] for row_id in test_ids]
                auc = auc_roc(labels, test_probabilities[:, model_index])
                heads_report[model] = {
                    "auroc": auc,
                    "feature_block_permutation_auc_drop": permutation_importance_by_block(
                        heads[model], auc, test_x, labels, blocks, 20260730 + test_fold + model_index
                    ),
                }
            plain_score, abstain_score = micro(plain), micro(abstain)
            fold_result["tiers"][tier] = {
                "features": len(feature_names), "feature_blocks": {key: len(value) for key, value in blocks.items()},
                "best_single_step_sr": best_score, "oracle_step_sr": oracle_test,
                "routed_step_sr": plain_score, "abstain_step_sr": abstain_score,
                "headroom_capture": (plain_score - best_score) / headroom if headroom else 0.0,
                "abstain_headroom_capture": (abstain_score - best_score) / headroom if headroom else 0.0,
                "abstain_threshold": threshold, "heads": heads_report,
            }
            print(f"completed {pool} fold={test_fold} tier={tier}", flush=True)
        folds.append(fold_result)
    aggregate = {}
    for tier in ("T0", "T1", "T2"):
        aggregate[tier] = {}
        for metric in ("best_single_step_sr", "oracle_step_sr", "routed_step_sr", "abstain_step_sr", "headroom_capture", "abstain_headroom_capture"):
            values = [fold["tiers"][tier][metric] for fold in folds]
            aggregate[tier][metric] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "folds": values}
    return {"disagreement_rows": len(disagreement), "models": models, "folds": folds, "aggregate": aggregate}


def add_pooled_metrics(result):
    for pool, value in result["pools"].items():
        total_pool_rows = 7650 if pool.startswith("androidcontrol/") else 2080
        for tier in ("T0", "T1", "T2"):
            folds = value["folds"]
            headroom = sum(
                (fold["tiers"][tier]["oracle_step_sr"] - fold["tiers"][tier]["best_single_step_sr"])
                * fold["test_rows"]
                for fold in folds
            )
            routed_gain = sum(
                (fold["tiers"][tier]["routed_step_sr"] - fold["tiers"][tier]["best_single_step_sr"])
                * fold["test_rows"]
                for fold in folds
            )
            abstain_gain = sum(
                (fold["tiers"][tier]["abstain_step_sr"] - fold["tiers"][tier]["best_single_step_sr"])
                * fold["test_rows"]
                for fold in folds
            )
            value["aggregate"][tier]["pooled"] = {
                "headroom_steps": round(headroom),
                "routed_gain_steps": round(routed_gain),
                "abstain_gain_steps": round(abstain_gain),
                "headroom_capture": routed_gain / headroom if headroom else 0.0,
                "abstain_headroom_capture": abstain_gain / headroom if headroom else 0.0,
                "projected_full_pool_delta": routed_gain / total_pool_rows,
                "projected_abstain_full_pool_delta": abstain_gain / total_pool_rows,
                "full_pool_rows": total_pool_rows,
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1", type=Path)
    parser.add_argument("--finalize-existing", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.finalize_existing is not None:
        result = add_pooled_metrics(json.loads(args.finalize_existing.read_text()))
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({pool: value["aggregate"]["T2"]["pooled"] for pool, value in result["pools"].items()}, indent=2))
        return
    if args.e1 is None:
        parser.error("--e1 is required unless --finalize-existing is used")
    e1 = json.loads(args.e1.read_text())
    result = {
        "status": "PASS",
        "contract": {
            "purpose": "routing feasibility upper bound, not deployment recipe",
            "pool": "disagreement rows only",
            "T0": "instruction hashing embedding, history length, image low-dimensional features",
            "T1": "T0 plus all released-parser outputs and per-model raw hashing embeddings",
            "T2": "T1 plus E1 selected-subset vote margin/dispersion and pairwise-action summaries",
            "prior_audit_comparator": {"union": 0.3059, "structural_router": 0.2397},
        },
        "pools": {},
    }
    for bench, setting in POOLS:
        result["pools"][f"{bench}/{setting}"] = run_pool(bench, setting, e1)
    result = add_pooled_metrics(result)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({pool: {
        tier: {
            "capture": values["aggregate"][tier]["headroom_capture"]["mean"],
            "abstain_capture": values["aggregate"][tier]["abstain_headroom_capture"]["mean"],
        } for tier in ("T0", "T1", "T2")
    } for pool, values in result["pools"].items()}, indent=2))


if __name__ == "__main__":
    main()