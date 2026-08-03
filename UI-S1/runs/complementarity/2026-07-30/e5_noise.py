import argparse
import json
import math
from pathlib import Path

import numpy as np


PROMPTS = ("original", "please_carry_out", "your_objective")
PROFILES = {"androidcontrol": (12800, 768), "mind2web": (1344, 768)}
TRANSITIONS = (
    ("AndroidControl GUI-R1 High 3B -> 7B", "androidcontrol", 504 / 7708),
    ("AndroidControl UI-AGILE High 3B -> 7B", "androidcontrol", 149 / 7708),
    ("AndroidControl GUI-R1 Low 3B -> 7B", "androidcontrol", 102 / 7708),
    ("AndroidControl UI-AGILE Low 3B -> 7B", "androidcontrol", -118 / 7708),
    ("Mind2Web TongUI 3B -> 7B", "mind2web", 82 / 2080),
    ("Mind2Web TongUI 7B -> 32B", "mind2web", -19 / 2080),
    ("Mind2Web UI-TARS 2B -> 7B", "mind2web", 308 / 2080),
    ("Mind2Web UI-TARS 7B -> 72B", "mind2web", 131 / 2080),
)


def score_path(artifact_root, benchmark, prompt, tokens):
    return artifact_root / benchmark / f"{prompt}_{tokens}" / "score.json"


def baseline_score(root, benchmark):
    if benchmark == "androidcontrol":
        path = root / "runs/androidcontrol-rft/2026-07-29/artifacts/gui-r1-7b/high/score.json"
        return json.loads(path.read_text())["metrics"]["step_success"]["accuracy"], str(path.relative_to(root))
    path = root / "runs/mind2web-tongui/2026-07-28/artifacts/tongui-7b/merged/score.json"
    return json.loads(path.read_text())["step_success_micro"], str(path.relative_to(root))


def read_cell(path, benchmark):
    value = json.loads(path.read_text())
    if benchmark == "androidcontrol":
        if value["rows"] != 7708 or value["coverage"] != "COMPLETE":
            raise ValueError(f"incomplete E5 AndroidControl score: {path}")
        return value["metrics"]["step_success"]["accuracy"]
    if value["rows"] != 2080 or value["coverage"] != "COMPLETE":
        raise ValueError(f"incomplete E5 Mind2Web score: {path}")
    return value["step_success_micro"]


def render_table(result):
    lines = [
        "# E5 MDE-aware transition table", "",
        "MDE is twice the sample standard deviation across the 3 prompt x 2 preprocessing cells. Greedy decoding has no seed dimension.", "",
        "| Transition | Delta | Benchmark MDE | Distinguishable |",
        "|---|---:|---:|---|",
    ]
    for item in result.get("transitions", []):
        lines.append(
            f"| {item['name']} | {item['delta']:+.2%} | {item['mde']:.2%} | {'YES' if item['distinguishable'] else 'NO'} |"
        )
    if not result.get("transitions"):
        lines.extend(["", "Pending complete E5 inference cells."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    result = {
        "status": "PASS",
        "contract": {
            "prompt_variants": list(PROMPTS),
            "visual_profiles": {
                "androidcontrol": {"original_max_visual_tokens": 12800, "deployment_max_visual_tokens": 768},
                "mind2web": {"original_max_visual_tokens": 1344, "deployment_max_visual_tokens": 768},
            },
            "generation": "greedy; seed dimension skipped",
            "sd": "sample standard deviation, ddof=1",
            "mde": "2 * sample SD",
        },
        "benchmarks": {},
        "missing_cells": [],
    }
    for benchmark, profiles in PROFILES.items():
        values = []
        cells = []
        baseline, source = baseline_score(root, benchmark)
        for prompt in PROMPTS:
            for tokens in profiles:
                if prompt == "original" and tokens == profiles[0]:
                    score, score_source = baseline, source
                else:
                    path = score_path(args.artifact_root, benchmark, prompt, tokens)
                    if not path.exists():
                        result["missing_cells"].append(f"{benchmark}/{prompt}_{tokens}")
                        continue
                    score, score_source = read_cell(path, benchmark), str(path)
                values.append(score)
                cells.append({"prompt": prompt, "max_visual_tokens": tokens, "step_micro": score, "source": score_source})
        summary = {"cells": cells, "complete_cells": len(cells), "expected_cells": 6}
        if len(values) == 6:
            summary.update({
                "mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)),
                "mde": float(2 * np.std(values, ddof=1)), "minimum": min(values), "maximum": max(values),
            })
        result["benchmarks"][benchmark] = summary
    if result["missing_cells"]:
        result["status"] = "PENDING_INFERENCE"
    else:
        result["transitions"] = []
        for name, benchmark, delta in TRANSITIONS:
            mde = result["benchmarks"][benchmark]["mde"]
            result["transitions"].append({
                "name": name, "benchmark": benchmark, "delta": delta, "mde": mde,
                "distinguishable": abs(delta) >= mde,
            })
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.table.write_text(render_table(result))
    print(json.dumps({"status": result["status"], "missing_cells": result["missing_cells"]}, indent=2))


if __name__ == "__main__":
    main()