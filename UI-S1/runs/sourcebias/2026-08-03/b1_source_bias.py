import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import chisquare

from sourcebias_common import load_pools, rule_outputs, split_ids, split_72


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]


def test_distribution(rows, outputs, rule, stratum):
    selected_rows = [row for row in rows if stratum == "all" or outputs[row["id"]]["correct"] == (stratum == "correct")]
    models = sorted({candidate["model"] for row in selected_rows for candidate in row["candidates"]})
    observed_counts = Counter(outputs[row["id"]]["selected_model"] for row in selected_rows)
    candidate_counts = Counter(candidate["model"] for row in selected_rows for candidate in row["candidates"])
    observed = np.asarray([observed_counts[model] for model in models], dtype=np.float64)
    candidate = np.asarray([candidate_counts[model] for model in models], dtype=np.float64)
    proportions = candidate / candidate.sum()
    expected = proportions * observed.sum()
    test = chisquare(observed, expected)
    residual = (observed - expected) / np.sqrt(expected * (1 - proportions))
    cramers_v = math.sqrt(float(test.statistic) / (observed.sum() * (len(models) - 1))) if observed.sum() and len(models) > 1 else 0.0
    return {
        "rows": len(selected_rows), "models": models,
        "observed_winners": {model: int(observed_counts[model]) for model in models},
        "candidate_members": {model: int(candidate_counts[model]) for model in models},
        "expected_winners": {model: float(expected[index]) for index, model in enumerate(models)},
        "standardized_residuals": {model: float(residual[index]) for index, model in enumerate(models)},
        "chi_square": float(test.statistic), "p_value": float(test.pvalue), "cramers_V": cramers_v,
    }


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--figure",type=Path,required=True); args=parser.parse_args()
    config=yaml.safe_load((RUN_DIR/"configs/b1_pools.yaml").read_text()); contexts,pools=load_pools(); reports={}; compositions={}
    for pool_name, rows in pools.items():
        scale=config["pools"][pool_name]["scale"]; context=contexts[scale]; splitter=split_ids if scale=="7B" else split_72
        outputs, components=rule_outputs(context,rows,splitter); reports[pool_name]={}
        for rule in config["rules"]:
            reports[pool_name][rule]={stratum:test_distribution(rows,outputs[rule],rule,stratum) for stratum in config["strata"]}
        compositions[pool_name]={rule:{stratum:dict(sorted(Counter(model for row in rows if stratum=="all" or outputs[rule][row["id"]]["correct"]==(stratum=="correct") for model in components[rule][row["id"]]).items())) for stratum in config["strata"]} for rule in components}
    seven=reports["7B_Uniform_Mixed_N12"]["B3_mvp"]["incorrect"]; seventy=reports["72B_Uniform_Mixed_N8"]["B3_mvp"]["incorrect"]
    pass7=seven["p_value"]<.001 and seven["standardized_residuals"]["GTA1-7B"]>0
    pass72=seventy["p_value"]<.001 and seventy["standardized_residuals"]["GTA1-72B"]>0
    result={"schema_version":1,"status":"PASS","reports":reports,"winner_component_composition":compositions,"gate":{"7B_bias_pass":pass7,"72B_bias_pass":pass72,"B1_pass":pass7,"B2_action":"RUN_BOTH_SCALES" if pass7 and pass72 else "RUN_72B_ONLY" if pass72 else "CANCEL_B2","B_K1":not(pass7 or pass72)},"source_config":str((RUN_DIR/"configs/b1_pools.yaml").relative_to(ROOT))}
    figure,axes=plt.subplots(1,2,figsize=(10,4.2))
    for axis,pool,title in zip(axes,("7B_Uniform_Mixed_N12","72B_Uniform_Mixed_N8"),("7B incorrect B3 winners","72B incorrect B3 winners")):
        report=reports[pool]["B3_mvp"]["incorrect"]; models=report["models"]; axis.bar(range(len(models)),[report["observed_winners"][m] for m in models],label="Observed"); axis.plot(range(len(models)),[report["expected_winners"][m] for m in models],"o-",color="#C84B31",label="Expected by candidates"); axis.set_xticks(range(len(models)),[m.split("-")[0] for m in models],rotation=15); axis.set_title(title); axis.legend(); axis.grid(axis="y",alpha=.2)
    figure.tight_layout(); args.figure.parent.mkdir(parents=True,exist_ok=True); figure.savefig(args.figure); plt.close(figure); result["figure"]=str(args.figure.resolve().relative_to(ROOT)); args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps({"7B":seven,"72B":seventy,"gate":result["gate"]},indent=2,sort_keys=True))


if __name__=="__main__": main()