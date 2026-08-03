import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
ALLOCATION_DIR = ROOT / "runs/allocation-law/2026-08-01"
DIVERSITY_DIR = ROOT / "runs/diversity-axis/2026-08-02"
sys.path.insert(0, str(DIVERSITY_DIR))
sys.path.insert(0, str(ALLOCATION_DIR))
from x3_curve_stats import load_sources
from allocation_eval import build_pool, compact_evaluation, l2_units


def load_closing_pools():
    gta1, generated, l1_units = load_sources()
    pools = {}
    units = {}
    units["v_only_N12"] = [("GTA1-7B", view) for view in range(12)]
    units["mixed_N12"] = [
        tuple(unit.rsplit("/view", 1)) for unit in l1_units[12]
    ]
    units["mixed_N12"] = [(model, int(view)) for model, view in units["mixed_N12"]]
    units["v_only_N16"] = [("GTA1-7B", view) for view in range(16)]
    units["mixed_N16"] = [
        tuple(unit.rsplit("/view", 1)) for unit in l1_units[16]
    ]
    units["mixed_N16"] = [(model, int(view)) for model, view in units["mixed_N16"]]
    l2 = l2_units(ALLOCATION_DIR / "configs/l2_pools.yaml")
    units["qwen3_N12"] = l2["qwen3_12views"]
    units["uitars_N12"] = l2["uitars_12views"]
    for name, selected in units.items():
        rows = build_pool(gta1, generated, selected)
        pools[name] = {"rows": rows, "evaluation": compact_evaluation(rows)}
    return gta1, pools