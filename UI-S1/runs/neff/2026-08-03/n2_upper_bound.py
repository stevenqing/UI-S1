import argparse
import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
RHO_VIEW = 0.895
NEFF_LIMIT = 1 / RHO_VIEW


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--n1", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    n1 = json.loads(args.n1.read_text())
    collapse = n1["primary_panel"]["collapse_any_estimator"]
    diagnostics = {}
    for estimator, fits in n1["primary_panel"]["fits"].items():
        fit = fits["N_eff"]
        diagnostics[estimator] = {
            "N_eff_limit": NEFF_LIMIT,
            "predicted_accuracy": fit["intercept"] + fit["coefficient"] * NEFF_LIMIT,
            "fit_residual_sd": fit["residual_sd"],
            "fit_collapse_success": fits["collapse_success"],
        }
    observed = {
        "V_only_N16_B3": 0.5831752055660974,
        "H1_official_B3_N10": 0.6046805819101835,
        "H1_graph_centroid_N10": 0.6179633143580012,
        "paper_only_GRPO_selector": 0.628,
    }
    result = {
        "schema_version": 1,
        "status": "ELIGIBLE" if collapse else "BLOCKED_N1_COLLAPSE",
        "claim_eligible": collapse,
        "rho_view_external_reference": RHO_VIEW,
        "asymptotic_N_eff": NEFF_LIMIT,
        "diagnostic_extrapolations": diagnostics,
        "observed_references": observed,
        "adjudication": "No impossibility upper-bound claim is made because no preregistered one-factor collapse passed." if not collapse else "Evaluate winning estimator against local observations.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
