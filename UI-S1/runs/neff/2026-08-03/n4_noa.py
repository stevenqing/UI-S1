import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CALA_DIR = ROOT / "runs/cala/2026-08-03"
H1_DIR = ROOT / "runs/ccm-h2h/2026-07-31/h1"
H3_DIR = ROOT / "runs/ccm-h2h/2026-07-31/h3"
CLOSING_DIR = ROOT / "runs/closing/2026-08-02"
sys.path.insert(0, str(CALA_DIR))
sys.path.insert(0, str(H1_DIR))
sys.path.insert(0, str(H3_DIR))
sys.path.insert(0, str(CLOSING_DIR))
from cala_common import BUDGETS, SHARED_ACTIONS, UNIFORM_SEQUENCE, action_name, build_rows, correctness, load_bank, split_ids
from cala_static import evaluate_fold
from aggregators_coord import mvp_official
from h3_eval import ccm_select, fit_ccm, point_in_bbox
from f1_paired_bootstrap import paired_bootstrap


SEED = 20260803
MDE = 0.007043345177520599
ACTION_INDEX = {action: index for index, action in enumerate(SHARED_ACTIONS)}


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohen_kappa(left, right):
    left = np.asarray(left, dtype=np.bool_); right = np.asarray(right, dtype=np.bool_)
    observed = float(np.mean(left == right)); left_rate = float(np.mean(left)); right_rate = float(np.mean(right))
    expected = left_rate * right_rate + (1-left_rate)*(1-right_rate)
    return 1.0 if math.isclose(expected, 1.0) else (observed-expected)/(1-expected)


def generalized_neff(actions, correlations):
    count = len(actions)
    denominator = float(sum(correlations[left, right] for left in actions for right in actions))
    if denominator <= 0:
        raise ValueError(f"NOA nonpositive generalized-N_eff denominator: {denominator}")
    return count * count / denominator


def development_statistics(context, dev_ids):
    correct = {action: correctness(context, action, dev_ids) for action in SHARED_ACTIONS}
    accuracy = {action: float(values.mean()) for action, values in correct.items()}
    correlations = {}
    for left in SHARED_ACTIONS:
        for right in SHARED_ACTIONS:
            if left == right:
                correlations[left, right] = 1.0
            elif (right, left) in correlations:
                correlations[left, right] = correlations[right, left]
            else:
                correlations[left, right] = cohen_kappa(~correct[left], ~correct[right])
    return correct, accuracy, correlations


def noa_sequence(accuracy, correlations, length=16):
    selected = []
    records = []
    current = 0.0
    while len(selected) < length:
        choices = []
        for action in SHARED_ACTIONS:
            if action in selected: continue
            value = generalized_neff([*selected, action], correlations)
            marginal = value - current
            choices.append(((marginal, accuracy[action], -ACTION_INDEX[action]), action, value, marginal))
        _, action, value, marginal = max(choices, key=lambda item: item[0])
        selected.append(action); current = value
        records.append({"step": len(selected), "action": action_name(action), "development_N_eff": value, "marginal_N_eff": marginal, "individual_development_accuracy": accuracy[action]})
    return tuple(selected), records


def realized_neff(points):
    count = len(points)
    denominator = count
    for left in range(count):
        for right in range(left + 1, count):
            similarity = math.exp(-math.dist(points[left], points[right])**2/(2*14.0**2))
            denominator += 2*similarity
    return count*count/denominator


def row_prefix_data(context, row_ids, sequence):
    correctness_by_prefix = np.zeros((len(row_ids), 13), dtype=np.bool_)
    marginals = np.zeros((len(row_ids), 16), dtype=np.float64)
    for row_index, row_id in enumerate(row_ids):
        candidates = []
        points = []
        previous = 0.0
        bbox = context["metadata"][row_id]["target_bbox"]
        for action_index, action in enumerate(sequence):
            candidate = context["bank"][action][row_id]
            candidates.append(candidate); points.append(candidate["point"])
            value = realized_neff(points); marginals[row_index, action_index] = value-previous; previous=value
            if action_index + 1 >= 4:
                pseudo = [{"coverage": item.get("coverage",0), "region":item["region"]} for item in candidates]
                prediction = mvp_official(points, pseudo)
                correctness_by_prefix[row_index, action_index+1-4] = point_in_bbox(prediction, bbox)
    return marginals, correctness_by_prefix


def stopping_counts(marginals, threshold):
    counts = np.full(len(marginals), 16, dtype=np.int64)
    for index in range(3, 16):
        stop = (counts == 16) & (marginals[:, index] < threshold)
        counts[stop] = index+1
    return counts


def select_threshold(marginals, prefix_correct):
    thresholds = np.unique(np.concatenate(([-np.inf], marginals[:,3:].ravel(), [np.inf])))
    choices = []
    rows = np.arange(len(marginals))
    for threshold in thresholds:
        counts = stopping_counts(marginals, threshold)
        mean_forwards = float(counts.mean())
        if mean_forwards > 8:
            continue
        accuracy = float(prefix_correct[rows, counts-4].mean())
        choices.append(((accuracy, -mean_forwards, threshold), threshold, accuracy, mean_forwards))
    if not choices:
        raise ValueError("NOA-stop has no threshold with mean forwards <=8")
    _, threshold, accuracy, mean_forwards = max(choices, key=lambda item:item[0])
    return threshold, {"development_B3":accuracy,"development_mean_forwards":mean_forwards,"eligible_thresholds":len(choices),"all_thresholds":len(thresholds)}


def variable_rows(context, row_ids, sequence, counts):
    return [{"id":row_id,"application":context["metadata"][row_id]["application"],"target_bbox":context["metadata"][row_id]["target_bbox"],"candidates":[context["bank"][action][row_id] for action in sequence[:int(count)]]} for row_id,count in zip(row_ids,counts)]


def evaluate_variable(context, dev_ids, test_ids, sequence, dev_counts, test_counts):
    dev_rows = variable_rows(context,dev_ids,sequence,dev_counts); test_rows = variable_rows(context,test_ids,sequence,test_counts)
    tables,priors = fit_ccm(dev_rows); outputs={rule:{} for rule in ("B3_mvp","M1_ccm","pass_at_n")}
    for row in test_rows:
        candidates=row["candidates"]; points=[item["point"] for item in candidates]; pseudo=[{"coverage":item.get("coverage",0),"region":item["region"]} for item in candidates]
        outputs["B3_mvp"][row["id"]]=point_in_bbox(mvp_official(points,pseudo),row["target_bbox"])
        outputs["M1_ccm"][row["id"]]=point_in_bbox(candidates[ccm_select(row,tables,priors)]["point"],row["target_bbox"])
        outputs["pass_at_n"][row["id"]]=any(point_in_bbox(point,row["target_bbox"]) for point in points)
    return outputs


def merge(target,source):
    for rule,values in source.items():
        if set(target[rule])&set(values): raise ValueError("NOA duplicate held-out output")
        target[rule].update(values)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    context=load_bank(); static=json.loads((CALA_DIR/"cala_static_results.json").read_text())
    noa_outputs={budget:{rule:{} for rule in ("B3_mvp","M1_ccm","pass_at_n")} for budget in BUDGETS}; uniform_outputs={budget:{rule:{} for rule in ("B3_mvp","M1_ccm","pass_at_n")} for budget in BUDGETS}; stop_outputs={rule:{} for rule in ("B3_mvp","M1_ccm","pass_at_n")}
    fold_reports={}; all_test_counts=[]
    for fold in range(5):
        dev_ids,test_ids=split_ids(context,fold); _,accuracy,correlations=development_statistics(context,dev_ids); sequence,steps=noa_sequence(accuracy,correlations)
        for budget in BUDGETS:
            merge(noa_outputs[budget],evaluate_fold(context,dev_ids,test_ids,sequence[:budget]))
            merge(uniform_outputs[budget],evaluate_fold(context,dev_ids,test_ids,UNIFORM_SEQUENCE[:budget]))
        dev_marginals,dev_prefix=row_prefix_data(context,dev_ids,sequence); threshold,threshold_report=select_threshold(dev_marginals,dev_prefix)
        test_marginals,_=row_prefix_data(context,test_ids,sequence); dev_counts=stopping_counts(dev_marginals,threshold); test_counts=stopping_counts(test_marginals,threshold); all_test_counts.extend(test_counts.tolist())
        merge(stop_outputs,evaluate_variable(context,dev_ids,test_ids,sequence,dev_counts,test_counts))
        fold_reports[str(fold)]={"dev_rows":len(dev_ids),"test_rows":len(test_ids),"sequence":[action_name(action) for action in sequence],"steps":steps,"threshold":threshold,"threshold_selection":threshold_report,"test_mean_forwards":float(test_counts.mean()),"test_histogram":{str(k):v for k,v in sorted(Counter(test_counts).items())}}
    accuracy={"NOA_static":{str(b):{r:sum(v.values())/1581 for r,v in noa_outputs[b].items()} for b in BUDGETS},"Uniform_Mixed":{str(b):{r:sum(v.values())/1581 for r,v in uniform_outputs[b].items()} for b in BUDGETS},"NOA_stop":{r:sum(v.values())/1581 for r,v in stop_outputs.items()},"existing_baselines":static["accuracy"]}
    rows=[context["metadata"][row_id] for row_id in context["row_ids"]]; comparisons={}
    for budget in BUDGETS:
        for rule in ("B3_mvp","M1_ccm","pass_at_n"):
            record=paired_bootstrap(rows,noa_outputs[budget][rule],uniform_outputs[budget][rule],resamples=10000,seed=SEED); record.update({"left":f"NOA_static/N{budget}/{rule}","right":f"Uniform_Mixed/N{budget}/{rule}","left_accuracy":accuracy["NOA_static"][str(budget)][rule],"right_accuracy":accuracy["Uniform_Mixed"][str(budget)][rule]}); comparisons[f"NOA_static_N{budget}_{rule}_vs_Uniform"]=record
    for rule in ("B3_mvp","M1_ccm","pass_at_n"):
        record=paired_bootstrap(rows,stop_outputs[rule],uniform_outputs[12][rule],resamples=10000,seed=SEED); record.update({"left":f"NOA_stop/{rule}","right":f"Uniform_Mixed/N12/{rule}","left_accuracy":accuracy["NOA_stop"][rule],"right_accuracy":accuracy["Uniform_Mixed"]["12"][rule]}); comparisons[f"NOA_stop_{rule}_vs_Uniform_N12"]=record
    counts=np.asarray(all_test_counts); static_primary=comparisons["NOA_static_N12_B3_mvp_vs_Uniform"]; stop_primary=comparisons["NOA_stop_B3_mvp_vs_Uniform_N12"]
    result={"schema_version":1,"status":"PASS","rows":1581,"accuracy":accuracy,"comparisons":comparisons,"folds":fold_reports,"NOA_static_adjudication":{"minimum_success":static_primary["point_delta"]>=0,"strong_success":static_primary["ci_99"][0]>0},"NOA_stop_compute":{"mean_forwards":float(counts.mean()),"median_forwards":float(np.median(counts)),"p10":float(np.quantile(counts,.1)),"p90":float(np.quantile(counts,.9)),"histogram":{str(k):v for k,v in sorted(Counter(counts).items())}},"NOA_stop_adjudication":{"mean_forwards_at_most_8":float(counts.mean())<=8,"accuracy_within_MDE_of_Uniform_N12":stop_primary["point_delta"]>=-MDE,"strong_success":float(counts.mean())<=8 and stop_primary["point_delta"]>=-MDE},"sources":{"CALA_static_sha256":sha256_file(CALA_DIR/"cala_static_results.json"),"N5_sha256":sha256_file(RUN_DIR/"n5_stopping_gate.json"),"stop_operations_sha256":sha256_file(RUN_DIR/"AMENDMENT_001_NOA_STOP_OPERATIONS.md")}}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps({"accuracy":accuracy,"static_primary":static_primary,"stop_primary":stop_primary,"stop_compute":result["NOA_stop_compute"],"adjudication":{"static":result["NOA_static_adjudication"],"stop":result["NOA_stop_adjudication"]}},indent=2,sort_keys=True))


if __name__=="__main__": main()