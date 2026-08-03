import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from b1_source_bias import test_distribution
from sourcebias_common import b3_select_index, fixed_rows, load_pools, point_in_bbox, rule_outputs, split_ids, split_72


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
SEED = 20260803


def stratified_group_bootstrap(rows, values, resamples=10000):
    by_fold_group = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_fold_group[row["outer_fold"]][row["application"]].append(row["id"])
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(resamples):
        selected = []
        for fold in sorted(by_fold_group):
            groups = sorted(by_fold_group[fold])
            for group in rng.choice(groups, size=len(groups), replace=True):
                selected.extend(by_fold_group[fold][group])
        samples.append(float(np.mean([values[row_id] for row_id in selected])))
    return {"resamples":resamples,"seed":SEED,"point":float(np.mean(list(values.values()))),"bootstrap_mean":float(np.mean(samples)),"ci_99":[float(np.quantile(samples,.005)),float(np.quantile(samples,.995))],"p_le_zero_plus_one":float((1+sum(value<=0 for value in samples))/(resamples+1))}


def b3_outputs(rows):
    outputs = {}
    for row in rows:
        selected, _ = b3_select_index(row["candidates"])
        outputs[row["id"]] = {"selected_model":row["candidates"][selected]["model"],"correct":bool(point_in_bbox(row["candidates"][selected]["point"],row["target_bbox"]))}
    return outputs


def view_pool_report(context, actions, splitter):
    rows = fixed_rows(context, actions, splitter); outputs = b3_outputs(rows)
    return {stratum:test_distribution(rows,outputs,"B3_mvp",stratum) for stratum in ("all","correct","incorrect")}


def distance_report(context, actions, splitter):
    rows=fixed_rows(context,actions,splitter); models=[]
    for action in actions:
        if action[0] not in models: models.append(action[0])
    per_model={model:{} for model in models}
    for row in rows:
        diagonal=math.hypot(*row["img_size"])
        for model in models:
            points=[candidate["point"] for candidate in row["candidates"] if candidate["model"]==model]
            distances=[math.dist(points[left],points[right])/diagonal for left in range(len(points)) for right in range(left+1,len(points))]
            per_model[model][row["id"]]=float(np.median(distances))
    output={"lineage_median_pair_distance":{model:float(np.median(list(values.values()))) for model,values in per_model.items()},"paired_differences":{}}
    gta=models[0]
    for other in models[1:]:
        delta={row["id"]:per_model[gta][row["id"]]-per_model[other][row["id"]] for row in rows}
        output["paired_differences"][f"{gta}_minus_{other}"]=stratified_group_bootstrap(rows,delta)
    output["GTA_lower_than_both_99CI"]=all(value["ci_99"][1]<0 for value in output["paired_differences"].values())
    return output


def balance_pool(context, rows, splitter):
    balanced=[]
    for row in rows:
        counts=defaultdict(int)
        for action in row["actions"]: counts[action[0]]+=1
        target=min(counts.values()); retained=[]; seen=defaultdict(int)
        for action in row["actions"]:
            if seen[action[0]]<target: retained.append(action); seen[action[0]]+=1
        candidates=[context["bank"][action][row["id"]] for action in retained]
        balanced.append({**row,"actions":tuple(retained),"candidates":candidates})
    before=b3_outputs(rows); after=b3_outputs(balanced)
    return {"original_candidate_count":len(rows[0]["actions"]),"balanced_candidate_count":len(balanced[0]["actions"]),"original_accuracy":sum(value["correct"] for value in before.values())/len(rows),"balanced_accuracy":sum(value["correct"] for value in after.values())/len(rows),"delta":sum(after[row_id]["correct"]-before[row_id]["correct"] for row_id in before)/len(rows),"balanced_incorrect_source_bias":test_distribution(balanced,after,"B3_mvp","incorrect")}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    contexts,pools=load_pools(); context7,context72=contexts["7B"],contexts["72B"]
    model7=("GTA1-7B","Qwen3-VL-8B-Instruct","UI-TARS-7B-SFT"); model72=("GTA1-72B","UI-Venus-Ground-72B","Qwen3.5-122B-A10B")
    view_reports={
        "7B":{"view0":view_pool_report(context7,tuple((model,0) for model in model7),split_ids),"views1_3":view_pool_report(context7,tuple((model,view) for view in range(1,4) for model in model7),split_ids)},
        "72B":{"view0":view_pool_report(context72,tuple((model,0) for model in model72),split_72),"views1_3":view_pool_report(context72,tuple((model,view) for view in range(1,4) for model in model72),split_72)},
    }
    distances={"7B":distance_report(context7,tuple((model,view) for view in range(4) for model in model7),split_ids),"72B":distance_report(context72,tuple((model,view) for view in range(4) for model in model72),split_72)}
    balanced={}
    for name,rows in pools.items():
        if "Uniform_Mixed" not in name: continue
        context=context7 if name.startswith("7B") else context72; splitter=split_ids if name.startswith("7B") else split_72
        counts=defaultdict(int)
        for action in rows[0]["actions"]: counts[action[0]]+=1
        if max(counts.values())-min(counts.values())<=1: balanced[name]=balance_pool(context,rows,splitter)
    overrepresentation={}
    for scale,gta in (("7B","GTA1-7B"),("72B","GTA1-72B")):
        view0=view_reports[scale]["view0"]["incorrect"]["standardized_residuals"][gta]; crops=view_reports[scale]["views1_3"]["incorrect"]["standardized_residuals"][gta]
        overrepresentation[scale]={"view0_GTA_standardized_residual":view0,"views1_3_GTA_standardized_residual":crops,"weaker_on_view0":view0<crops}
    attribution_supported=all(overrepresentation[scale]["weaker_on_view0"] and distances[scale]["GTA_lower_than_both_99CI"] for scale in ("7B","72B"))
    result={"schema_version":1,"status":"PASS","view_source_bias":view_reports,"within_lineage_geometry":distances,"count_balancing":balanced,"proposal_source_attribution":{"by_scale":overrepresentation,"supported_at_both_scales":attribution_supported,"interpretation":"proposal_source_attribution" if attribution_supported else "heterogeneous_pool_aggregation_effect"}}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps({"proposal_source_attribution":result["proposal_source_attribution"],"geometry":distances,"count_balancing":{name:{key:value[key] for key in ("original_candidate_count","balanced_candidate_count","original_accuracy","balanced_accuracy","delta")} for name,value in balanced.items()}},indent=2,sort_keys=True))


if __name__=="__main__": main()