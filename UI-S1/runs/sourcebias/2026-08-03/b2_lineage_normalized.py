import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from sourcebias_common import b3_select_index, load_pools, official_groups, point_in_bbox, rule_outputs, split_ids, split_72


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
SEED = 20260803
THRESHOLD = 14.0


def fit_stats(rows):
    action_counts = defaultdict(lambda: [0, 0])
    model_counts = defaultdict(lambda: [0, 0])
    for row in rows:
        for candidate in row["candidates"]:
            action = (candidate["model"], candidate["view_index"])
            correct = int(point_in_bbox(candidate["point"], row["target_bbox"]))
            action_counts[action][0] += correct; action_counts[action][1] += 1
            model_counts[candidate["model"]][0] += correct; model_counts[candidate["model"]][1] += 1
    return {
        "action_accuracy": {action: values[0] / values[1] for action, values in action_counts.items()},
        "model_reliability": {model: values[0] / values[1] for model, values in model_counts.items()},
    }


def graph_components(points):
    adjacency = {index:set() for index in range(len(points))}
    for left in range(len(points)):
        for right in range(left+1,len(points)):
            if math.dist(points[left],points[right])<=THRESHOLD:
                adjacency[left].add(right); adjacency[right].add(left)
    components=[]; unvisited=set(range(len(points)))
    while unvisited:
        seed=min(unvisited); unvisited.remove(seed); stack=[seed]; component=[]
        while stack:
            node=stack.pop(); component.append(node)
            for neighbor in sorted(adjacency[node],reverse=True):
                if neighbor in unvisited: unvisited.remove(neighbor); stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return components


def centroid(points, indices, weights=None):
    values=np.asarray([points[index] for index in indices],dtype=np.float64)
    if weights is None: return values.mean(axis=0).tolist()
    weights=np.asarray(weights,dtype=np.float64)
    return ((values*weights[:,None]).sum(axis=0)/weights.sum()).tolist() if weights.sum()>0 else values.mean(axis=0).tolist()


def geometric_median(points):
    values=np.asarray(points,dtype=np.float64); estimate=values.mean(axis=0)
    for _ in range(100):
        distances=np.linalg.norm(values-estimate,axis=1)
        exact=np.where(distances<1e-12)[0]
        if len(exact): return values[int(exact[0])].tolist()
        updated=(values/distances[:,None]).sum(axis=0)/(1/distances).sum()
        if np.linalg.norm(updated-estimate)<=1e-6: return updated.tolist()
        estimate=updated
    return estimate.tolist()


def reduce_lineage(candidates, reduction, stats):
    points=[candidate["point"] for candidate in candidates]
    if reduction=="R1":
        group=official_groups(points)[0]; return centroid(points,group)
    if reduction=="R2": return geometric_median(points)
    if reduction=="R3":
        index=min(range(len(points)),key=lambda left:(sum(math.dist(points[left],point) for point in points),candidates[left]["view_index"],left)); return list(points[index])
    if reduction=="R4":
        component=max(graph_components(points),key=lambda values:(len(values),-min(candidates[index]["view_index"] for index in values),-min(values))); return centroid(points,component)
    if reduction=="R5":
        index=max(range(len(candidates)),key=lambda value:(stats["action_accuracy"][(candidates[value]["model"],candidates[value]["view_index"])],-candidates[value]["view_index"],-value)); return list(points[index])
    if reduction=="R6":
        index=min(range(len(candidates)),key=lambda value:(abs(candidates[value]["view_index"]),value)); return list(points[index])
    if reduction=="R7":
        group=official_groups(points)[0]; weights=[stats["action_accuracy"][(candidates[index]["model"],candidates[index]["view_index"])] for index in group]; return centroid(points,group,weights)
    raise ValueError(reduction)


def decide(representatives, models, decision, stats):
    if decision=="D1":
        component=max(graph_components(representatives),key=lambda values:(len(values),-min(values))); return centroid(representatives,component)
    reliabilities=stats["model_reliability"]
    if decision=="D2":
        winner=max(range(len(models)),key=lambda index:(sum(reliabilities[models[other]] for other in range(len(models)) if math.dist(representatives[index],representatives[other])<=THRESHOLD),reliabilities[models[index]],-index)); return list(representatives[winner])
    if decision=="D3":
        distances=[math.dist(representatives[left],representatives[right]) for left in range(len(models)) for right in range(left+1,len(models))]
        if distances and all(value>THRESHOLD for value in distances):
            winner=max(range(len(models)),key=lambda index:(reliabilities[models[index]],-index)); return list(representatives[winner])
        component=max(graph_components(representatives),key=lambda values:(len(values),-min(values))); return centroid(representatives,component)
    raise ValueError(decision)


def predict_r0(row, variant, stats):
    candidates = row["candidates"]
    points = [candidate["point"] for candidate in candidates]
    _, group = b3_select_index(candidates)
    if variant == "R0a":
        return centroid(points, group)
    by_model = defaultdict(list)
    models = []
    for index in group:
        model = candidates[index]["model"]
        if model not in by_model:
            models.append(model)
        by_model[model].append(index)
    lineage_centroids = [centroid(points, by_model[model]) for model in models]
    if variant == "R0b":
        return centroid(lineage_centroids, range(len(lineage_centroids)))
    if variant == "R0c":
        weights = [stats["model_reliability"][model] for model in models]
        return centroid(lineage_centroids, range(len(lineage_centroids)), weights)
    raise ValueError(variant)


def predict(row, variant, stats):
    if variant.startswith("R0"):
        return predict_r0(row, variant, stats)
    reduction,decision=variant.split("_"); by_model=defaultdict(list); models=[]
    for candidate in row["candidates"]:
        if candidate["model"] not in by_model: models.append(candidate["model"])
        by_model[candidate["model"]].append(candidate)
    representatives=[reduce_lineage(by_model[model],reduction,stats) for model in models]
    return decide(representatives,models,decision,stats)


def evaluate(rows, variant, stats):
    return {row["id"]:bool(point_in_bbox(predict(row,variant,stats),row["target_bbox"])) for row in rows}


def bootstrap(rows, left, right, resamples=10000):
    by_fold_group=defaultdict(lambda:defaultdict(list))
    for row in rows: by_fold_group[row["outer_fold"]][row["application"]].append(row["id"])
    rng=np.random.default_rng(SEED); samples=[]
    for _ in range(resamples):
        selected=[]
        for fold in sorted(by_fold_group):
            groups=sorted(by_fold_group[fold])
            for group in rng.choice(groups,size=len(groups),replace=True): selected.extend(by_fold_group[fold][group])
        samples.append(float(np.mean([left[row_id]-right[row_id] for row_id in selected])))
    point=float(np.mean([left[row["id"]]-right[row["id"]] for row in rows]))
    return {"left_accuracy":sum(left.values())/len(left),"right_accuracy":sum(right.values())/len(right),"point_delta":point,"resamples":resamples,"seed":SEED,"ci_99":[float(np.quantile(samples,.005)),float(np.quantile(samples,.995))],"p_one_sided_delta_le_zero":float((1+sum(value<=0 for value in samples))/(resamples+1))}


def run_scale(scale, context, rows, splitter, variants, best_model, reported_best_single):
    baselines,_=rule_outputs(context,rows,splitter)
    baseline_outputs={rule:{row_id:value["correct"] for row_id,value in values.items()} for rule,values in baselines.items()}
    best_single={row["id"]:bool(point_in_bbox(next(candidate["point"] for candidate in row["candidates"] if candidate["model"]==best_model and candidate["view_index"]==0),row["target_bbox"])) for row in rows}
    nested={}; selections={}; grid={variant:{} for variant in variants}
    for outer in range(5):
        test=[row for row in rows if row["outer_fold"]==outer]; outer_dev=[row for row in rows if row["outer_fold"]!=outer]
        inner_val_fold=(outer+1)%5; inner_train=[row for row in rows if row["outer_fold"] not in (outer,inner_val_fold)]; inner_val=[row for row in rows if row["outer_fold"]==inner_val_fold]
        inner_stats=fit_stats(inner_train); scores=[]
        for order,variant in enumerate(variants):
            outputs=evaluate(inner_val,variant,inner_stats); scores.append((sum(outputs.values())/len(outputs),-order,variant))
        selected=max(scores)[2]; refit=fit_stats(outer_dev); fold_output=evaluate(test,selected,refit); nested.update(fold_output)
        selections[str(outer)]={"inner_validation_fold":inner_val_fold,"inner_train_rows":len(inner_train),"inner_validation_rows":len(inner_val),"selected_variant":selected,"inner_validation_accuracy":max(scores)[0],"outer_dev_rows":len(outer_dev),"outer_test_rows":len(test)}
        for variant in variants: grid[variant].update(evaluate(test,variant,refit))
    matched_best_accuracy=sum(best_single.values())/len(rows)
    accuracy={"nested_LN":sum(nested.values())/len(rows),"B3_mvp":sum(baseline_outputs["B3_mvp"].values())/len(rows),"M1_ccm":sum(baseline_outputs["M1_ccm"].values())/len(rows),"best_single_reported":reported_best_single,"best_single_matched_bank_view0":matched_best_accuracy}
    grid_accuracy={variant:sum(values.values())/len(rows) for variant,values in grid.items()}
    comparisons={name:bootstrap(rows,nested,values) for name,values in (("vs_B3",baseline_outputs["B3_mvp"]),("vs_M1",baseline_outputs["M1_ccm"]),("vs_best_single_matched_bank_view0",best_single))}
    comparisons["vs_best_single_reported"]={"nested_accuracy":accuracy["nested_LN"],"reported_best_single_accuracy":reported_best_single,"point_delta":accuracy["nested_LN"]-reported_best_single,"paired_inference_available":math.isclose(matched_best_accuracy,reported_best_single,abs_tol=1e-15)}
    grid_values = np.asarray(list(grid_accuracy.values()), dtype=np.float64)
    return {"accuracy":accuracy,"outer_selections":selections,"selection_frequency":dict(Counter(item["selected_variant"] for item in selections.values())),"descriptive_crossfit_grid":grid_accuracy,"descriptive_best_variant":max(variants,key=lambda variant:(grid_accuracy[variant],-variants.index(variant))),"descriptive_summary":{"min":float(np.min(grid_values)),"median":float(np.median(grid_values)),"max":float(np.max(grid_values)),"nested_percentile_leq":float(np.mean(grid_values<=accuracy["nested_LN"]))},"comparisons":comparisons,"outputs":{"nested_LN":nested,"B3_mvp":baseline_outputs["B3_mvp"],"M1_ccm":baseline_outputs["M1_ccm"],"best_single":best_single}}


def validate_frozen_baselines(reports, config, allow_recovered_baseline_drift=False):
    expected = {
        scale: {key: config["baselines"][scale][key] for key in ("B3", "M1")}
        for scale in ("7B", "72B")
    }
    actual = {
        scale: {"B3": reports[scale]["accuracy"]["B3_mvp"], "M1": reports[scale]["accuracy"]["M1_ccm"]}
        for scale in ("7B", "72B")
    }
    matches = all(
        math.isclose(actual[scale][key], expected[scale][key], abs_tol=1e-15)
        for scale in expected for key in expected[scale]
    )
    if not matches and not allow_recovered_baseline_drift:
        raise ValueError(f"B2 frozen baseline mismatch: {actual}")
    return {
        "matches": matches,
        "mode": "FROZEN" if matches else "RECOVERY_DRIFT_ACCEPTED",
        "expected": expected,
        "actual": actual,
        "delta": {
            scale: {key: actual[scale][key] - expected[scale][key] for key in expected[scale]}
            for scale in expected
        },
    }


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--allow-recovered-baseline-drift",action="store_true"); args=parser.parse_args()
    config=yaml.safe_load((RUN_DIR/"configs/b2_variants.yaml").read_text()); variants=config["combined_method_order"]; r0_variants=config["r0_only_method_order"]; contexts,pools=load_pools()
    if len(variants)!=24 or len(set(variants))!=24 or variants[:3]!=r0_variants:
        raise ValueError("B2 frozen combined method order mismatch")
    reports={
        "7B":run_scale("7B",contexts["7B"],pools["7B_Uniform_Mixed_N12"],split_ids,variants,"Qwen3-VL-8B-Instruct",config["baselines"]["7B"]["best_single"]["accuracy"]),
        "72B":run_scale("72B",contexts["72B"],pools["72B_Uniform_Mixed_N8"],split_72,variants,"Qwen3.5-122B-A10B",config["baselines"]["72B"]["best_single"]["accuracy"]),
    }
    r0_only_reports={
        "7B":run_scale("7B",contexts["7B"],pools["7B_Uniform_Mixed_N12"],split_ids,r0_variants,"Qwen3-VL-8B-Instruct",config["baselines"]["7B"]["best_single"]["accuracy"]),
        "72B":run_scale("72B",contexts["72B"],pools["72B_Uniform_Mixed_N8"],split_72,r0_variants,"Qwen3.5-122B-A10B",config["baselines"]["72B"]["best_single"]["accuracy"]),
    }
    baseline_validation = validate_frozen_baselines(reports, config, args.allow_recovered_baseline_drift)
    for scale in ("7B", "72B"):
        if not math.isclose(reports[scale]["accuracy"]["best_single_reported"], config["baselines"][scale]["best_single"]["accuracy"], abs_tol=1e-15):
            raise ValueError(f"B2 reported best-single mismatch: {scale}")
    success={scale:reports[scale]["comparisons"]["vs_B3"]["point_delta"]>config["MDE"] and reports[scale]["comparisons"]["vs_B3"]["ci_99"][0]>0 for scale in reports}
    r0_success={scale:r0_only_reports[scale]["comparisons"]["vs_B3"]["point_delta"]>config["MDE"] and r0_only_reports[scale]["comparisons"]["vs_B3"]["ci_99"][0]>0 for scale in r0_only_reports}
    combined_success=all(success.values())
    result={"schema_version":1,"status":"PASS","frozen_baseline_validation":baseline_validation,"reports":reports,"r0_only_reports":r0_only_reports,"primary_success_by_scale":success,"r0_only_success_by_scale":r0_success,"B2_primary_success":combined_success,"R0_only_primary_success":all(r0_success.values()),"B_K4":reports["72B"]["accuracy"]["nested_LN"]<reports["72B"]["accuracy"]["best_single_reported"],"B_K5":combined_success and not all(r0_success.values()),"B3x_action":"RUN" if combined_success else "CANCEL","variant_count":len(variants),"r0_only_variant_count":len(r0_variants)}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({scale:{"accuracy":value["accuracy"],"selections":Counter(item["selected_variant"] for item in value["outer_selections"].values()),"comparisons":value["comparisons"]} for scale,value in reports.items()}|{"decision":{"primary_success":result["B2_primary_success"],"B_K4":result["B_K4"],"B3x":result["B3x_action"]}},indent=2,sort_keys=True,default=dict))


if __name__=="__main__": main()