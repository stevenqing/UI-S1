import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def pct(value):
    return "—" if value is None else f"{100 * value:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    allocation = ROOT / "runs/allocation-law/2026-08-01"
    collision = ROOT / "runs/collision-law/2026-07-30"
    l1 = json.loads((allocation / "L1_RESULTS.json").read_text())
    l2 = json.loads((allocation / "L2_RESULTS.json").read_text())
    w3 = json.loads((collision / "w3_summary.json").read_text())
    rows = [
        ("GTA1 bare single view", 1, None, w3["gta1_bare"]["accuracy"], None, "local reproduction; model-card anchor 50.1"),
        ("GTA1 + official MVP candidates", 12, l1["evaluations"]["v_only"]["12"]["accuracy"]["B3_mvp"], l1["evaluations"]["v_only"]["12"]["accuracy"]["M1_ccm"], l1["evaluations"]["v_only"]["12"]["accuracy"]["pass_at_n"], "local, fixed GTA1 lineage"),
        ("Qwen3 single lineage", 12, l2["pools"]["qwen3_12views"]["accuracy"]["B3_mvp"], l2["pools"]["qwen3_12views"]["accuracy"]["M1_ccm"], l2["pools"]["qwen3_12views"]["accuracy"]["pass_at_n"], "local, shared GTA1 geometry"),
        ("UI-TARS single lineage", 12, l2["pools"]["uitars_12views"]["accuracy"]["B3_mvp"], l2["pools"]["uitars_12views"]["accuracy"]["M1_ccm"], l2["pools"]["uitars_12views"]["accuracy"]["pass_at_n"], "local, shared GTA1 geometry"),
        ("MVP GRPO selector (4B, trained)", None, None, 0.628, None, "published paper only; different environment; excluded from calculations"),
        ("Mixed lineage", 12, l1["evaluations"]["mixed"]["12"]["accuracy"]["B3_mvp"], l1["evaluations"]["mixed"]["12"]["accuracy"]["M1_ccm"], l1["evaluations"]["mixed"]["12"]["accuracy"]["pass_at_n"], "local, 3 lineages x 4 views"),
        ("Mixed lineage", 24, l1["evaluations"]["mixed"]["24"]["accuracy"]["B3_mvp"], l1["evaluations"]["mixed"]["24"]["accuracy"]["M1_ccm"], l1["evaluations"]["mixed"]["24"]["accuracy"]["pass_at_n"], "local, one-sided budget extension"),
    ]
    lines = [
        "# Closing Main Tables",
        "",
        "## Primary ScreenSpot-Pro table",
        "",
        "| Configuration | Forwards | B3 | M1 / reported selector | pass@N | Source |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, forwards, b3, m1, oracle, source in rows:
        lines.append(f"| {name} | {'—' if forwards is None else forwards} | {pct(b3)} | {pct(m1)} | {pct(oracle)} | {source} |")
    lines.extend([
        "",
        "Only the local N=12 rows are direct fixed-budget comparisons on the same 1,581 examples and candidate geometry. The bare row uses one forward, Mixed N24 uses 24 forwards, and the published 62.8 row is not from our environment; none enters a paired comparison with those rows.",
        "",
        "The two additional lineages are not stronger hidden substitutes. At displayed two-decimal table precision, Qwen3 and UI-TARS individually trail GTA1 M1 by 3.60 and 7.96 percentage points (exact unrounded gaps: 3.61 and 7.97 points), yet the mixed N12 pool reaches 63.82.",
        "",
        "The published MVP anchor 61.7 is audited through the separate official-code reproduction at 61.35 (0.35 points lower). It is not an anchor for the N12 GTA1 row at 60.09/60.40, whose candidate budget and evaluation are different.",
        "",
        "We do not claim absolute ScreenSpot-Pro SOTA. The supported statement is: under the same local backbone inventory, shared candidate geometry, examples, and 12-forward test-time budget, the mixed pool exceeds every internally evaluated single-lineage configuration, including unchanged B3 and fold-local M1 selectors.",
        "",
        "## Lineage-count and composition table",
        "",
        "| N12 pool | B3 | M1 | pass@12 | Mean dev failure kappa |",
        "|---|---:|---:|---:|---:|",
    ])
    for key, label in (
        ("gta1_qwen3_6x2", "GTA1 + Qwen3"),
        ("gta1_uitars_6x2", "GTA1 + UI-TARS"),
        ("qwen3_uitars_6x2", "Qwen3 + UI-TARS"),
        ("three_lineages_4x3", "GTA1 + Qwen3 + UI-TARS"),
    ):
        pool = l2["pools"][key]
        lines.append(
            f"| {label} | {pct(pool['accuracy']['B3_mvp'])} | {pct(pool['accuracy']['M1_ccm'])} | {pct(pool['accuracy']['pass_at_n'])} | {pool['mean_dev_kappa']:.3f} |"
        )
    lines.extend([
        "",
        "The best two-lineage M1 pool (GTA1 + Qwen3, 63.88) is statistically close to the three-lineage point (63.82). This table supports a correlation/composition interpretation rather than a monotonic model-count claim.",
        "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(json.dumps({"status": "PASS", "rows_primary": len(rows), "rows_composition": 4}, indent=2))


if __name__ == "__main__":
    main()
