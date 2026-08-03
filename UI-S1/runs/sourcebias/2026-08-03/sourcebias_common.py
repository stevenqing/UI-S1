import json
import math
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CALA = ROOT / "runs/cala/2026-08-03"
H1 = ROOT / "runs/ccm-h2h/2026-07-31/h1"
H3 = ROOT / "runs/ccm-h2h/2026-07-31/h3"
sys.path.insert(0, str(CALA))
sys.path.insert(0, str(H1))
sys.path.insert(0, str(H3))
from cala_common import UNIFORM_SEQUENCE, load_bank, split_ids
from cala_transfer_72b import GTA_N8, UNIFORM_N8, load_context as load_72, split_ids as split_72
from aggregators_coord import official_groups
from h3_eval import ccm_select, fit_ccm, point_in_bbox


def parse_view_action(name):
    model, view = name.rsplit("/view", 1)
    return model, int(view)


def parse_region_action(name):
    model, region = name.rsplit("/region", 1)
    return model, int(region)


def policy_rows(context, fold_actions, split_function):
    output = []
    for fold in range(5):
        _, test_ids = split_function(context, fold)
        actions = fold_actions[fold]
        for row_id in test_ids:
            output.append({
                "id": row_id,
                "application": context["metadata"][row_id]["application"],
                "target_bbox": context["metadata"][row_id]["target_bbox"],
                "img_size": context["metadata"][row_id]["img_size"],
                "outer_fold": fold,
                "actions": actions,
                "candidates": [context["bank"][action][row_id] for action in actions],
            })
    output.sort(key=lambda row: row["id"])
    if len(output) != 1581:
        raise ValueError("source-bias policy row coverage mismatch")
    return output


def fixed_rows(context, actions, split_function):
    return policy_rows(context, {fold: tuple(actions) for fold in range(5)}, split_function)


def load_pools():
    context7 = load_bank()
    static = json.loads((CALA / "cala_static_results.json").read_text())
    pools = {}
    for budget in (4, 8, 12, 16, 24):
        pools[f"7B_Uniform_Mixed_N{budget}"] = fixed_rows(context7, UNIFORM_SEQUENCE[:budget], split_ids)
    for method in ("CALA_S", "Quality_Only"):
        fold_actions = {
            fold: tuple(parse_view_action(name) for name in static["fold_sequences"][str(fold)]["sequences"][method][:12])
            for fold in range(5)
        }
        pools[f"7B_{method}_N12"] = policy_rows(context7, fold_actions, split_ids)
    context72 = load_72()
    transfer = json.loads((CALA / "cala_transfer_72b_results.json").read_text())
    pools["72B_Uniform_Mixed_N8"] = fixed_rows(context72, UNIFORM_N8, split_72)
    fold_actions72 = {fold: tuple(parse_region_action(name) for name in transfer["folds"][str(fold)]["CALA_S"]) for fold in range(5)}
    pools["72B_CALA_S_N8"] = policy_rows(context72, fold_actions72, split_72)
    contexts = {"7B": context7, "72B": context72}
    return contexts, pools


def b3_select_index(candidates):
    points = [candidate["point"] for candidate in candidates]
    groups = official_groups(points)
    scored = []
    for group_index, group in enumerate(groups):
        coverage = sum(candidates[index].get("coverage", 0) for index in group) / len(group)
        scored.append((len(group) + coverage / 1000, -group_index, group))
    winner = max(scored)[2]
    selected = max(winner, key=lambda index: (candidates[index].get("coverage", 0), -index))
    return selected, winner


def graph_component(candidates):
    points = [candidate["point"] for candidate in candidates]
    adjacency = {index: set() for index in range(len(points))}
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            if math.dist(points[left], points[right]) <= 14:
                adjacency[left].add(right); adjacency[right].add(left)
    components = []
    unvisited = set(range(len(points)))
    while unvisited:
        seed = min(unvisited); stack = [seed]; component = []; unvisited.remove(seed)
        while stack:
            node = stack.pop(); component.append(node)
            for neighbor in sorted(adjacency[node], reverse=True):
                if neighbor in unvisited:
                    unvisited.remove(neighbor); stack.append(neighbor)
        components.append(tuple(sorted(component)))
    winner = max(components, key=lambda values: (len(values), -min(values)))
    centroid = [sum(points[index][axis] for index in winner) / len(winner) for axis in (0, 1)]
    selected = min(range(len(points)), key=lambda index: (math.dist(points[index], centroid), index))
    return selected, winner, centroid


def rule_outputs(context, rows, split_function):
    output = {rule: {} for rule in ("B3_mvp", "M1_ccm", "graph_centroid")}
    component_models = {rule: {} for rule in ("B3_mvp", "graph_centroid")}
    for fold in range(5):
        fold_rows = [row for row in rows if row["outer_fold"] == fold]
        if not fold_rows:
            raise ValueError(f"source-bias empty fold: {fold}")
        actions = fold_rows[0]["actions"]
        if any(row["actions"] != actions for row in fold_rows):
            raise ValueError("source-bias action policy varies within fold")
        dev_ids, _ = split_function(context, fold)
        dev_rows = [{"id": row_id, "application": context["metadata"][row_id]["application"], "target_bbox": context["metadata"][row_id]["target_bbox"], "candidates": [context["bank"][action][row_id] for action in actions]} for row_id in dev_ids]
        tables, priors = fit_ccm(dev_rows)
        for row in fold_rows:
            candidates = row["candidates"]
            b3_index, b3_group = b3_select_index(candidates)
            m1_index = ccm_select(row, tables, priors)
            graph_index, graph_group, graph_point = graph_component(candidates)
            values = {
                "B3_mvp": (b3_index, candidates[b3_index]["point"]),
                "M1_ccm": (m1_index, candidates[m1_index]["point"]),
                "graph_centroid": (graph_index, graph_point),
            }
            for rule, (selected, point) in values.items():
                output[rule][row["id"]] = {"selected_index": selected, "selected_model": candidates[selected]["model"], "correct": bool(point_in_bbox(point, row["target_bbox"])), "point": list(map(float, point))}
            component_models["B3_mvp"][row["id"]] = [candidates[index]["model"] for index in b3_group]
            component_models["graph_centroid"][row["id"]] = [candidates[index]["model"] for index in graph_group]
    return output, component_models
