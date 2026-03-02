#!/usr/bin/env python3
"""
Task 4: Validate bottleneck identification results.

Runs five analysis modules on the per-app eigenfunction outputs from Task 3,
produces structured reports (JSON + Markdown) and visualizations, and emits
a Go/No-Go verdict to decide whether to proceed to Task 5.

Usage:
    python scripts/validate_bottlenecks.py                    # all 3 apps
    python scripts/validate_bottlenecks.py --app excel        # single app
    python scripts/validate_bottlenecks.py --percentile-k 10  # stricter threshold

Inputs (from Tasks 2 & 3):
    outputs/fnet/gui360/{app}/f_values.npz
    outputs/fnet/gui360/{app}/bottlenecks_described.json
    outputs/fnet/gui360/{app}/results.json
    outputs/transitions/gui360_full/transitions.jsonl
    outputs/transitions/gui360_full/adjacency.json
    outputs/transitions/gui360_full/state_registry.json

Outputs:
    outputs/bottleneck_validation/report.json
    outputs/bottleneck_validation/report.md
    outputs/bottleneck_validation/{app}/*.png
    outputs/bottleneck_validation/aggregate/*.png
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

APPS = ["excel", "word", "ppt"]

FNET_DIR = PROJECT_ROOT / "outputs" / "fnet" / "gui360"
TRANSITIONS_DIR = PROJECT_ROOT / "outputs" / "transitions" / "gui360_full"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "bottleneck_validation"

# ============================================================
# Expected bottleneck categories per app
# ============================================================
EXPECTED_CATEGORIES = {
    "excel": {
        "ribbon_nav_chain": "Data-only view (missing Home tab)",
        "dialog_chain": "Dialog-heavy states (Format Cells, Excel Options, etc.)",
        "localized_ui": "Non-English / Afrikaans UI paths",
    },
    "word": {
        "table_layout_mode": "Dual Layout tabs (table editing context)",
        "dialog_chain": "Dialog-heavy states (Word Options, etc.)",
        "developer_mode": "Developer tab active with specialized views",
    },
    "ppt": {
        "embedded_object_edit": "Embedded object editing (Data/Formulas tabs present)",
        "non_standard_view": "Non-standard views (missing Home tab)",
        "dialog_chain": "Dialog-heavy states (PPT Options, Insert Picture, etc.)",
    },
}


# ============================================================
# Module 1: Category classification engine
# ============================================================

def _has_tab(sig: str, tab: str) -> bool:
    """Check if a tab name appears in a comma-separated signature."""
    return tab in [t.strip() for t in sig.split(",")]


def _missing_tab(sig: str, tab: str) -> bool:
    """Check if a tab name is missing from a comma-separated signature."""
    return tab not in [t.strip() for t in sig.split(",")]


def _is_localized(dialog: str) -> bool:
    """Heuristic: dialog name contains non-ASCII or Afrikaans-looking words."""
    if dialog == "none":
        return False
    # Afrikaans keywords commonly seen in the data
    afrikaans_markers = [
        "waarskuwing", "opsies", "selle", "skakel", "outostoor",
        "spesiaal", "makro", "neem", "verwyder", "duplikate",
        "voeg", "hiperskakel", "gestoor", "hierdie", "rekenaar",
        "formateer", "ordenings",
    ]
    lower = dialog.lower()
    for marker in afrikaans_markers:
        if marker in lower:
            return True
    # Non-ASCII characters (excluding special Unicode marks)
    if any(ord(c) > 127 for c in dialog) and "bing_search" not in lower:
        return True
    return False


DIALOG_CHAIN_KEYWORDS = [
    "options", "format cells", "format", "insert picture", "insert chart",
    "insert hyperlink", "insert object", "insert video", "insert icon",
    "insert stock", "insert table", "insert function", "insert merge",
    "save as", "publish", "export", "font", "paragraph", "page setup",
    "borders", "word count", "signature", "table of contents",
    "smartart", "record macro", "remove duplicates", "sort warning",
    "sort", "import data", "power query", "document inspector",
    "add-ins", "compress", "custom views", "go to", "group",
    "word options", "excel options", "powerpoint options",
    "chart in microsoft", "object", "set up show", "slide size",
    "colors", "choose theme", "header and footer", "caption",
    "content control", "field", "envelopes", "language", "tabs",
    "share", "open", "type your text", "microsoft excel",
    "microsoft word", "microsoft powerpoint",
    "how do i turn on autosave", "compatibility",
]


def classify_bottleneck(state: dict, app: str) -> str | None:
    """Classify a single bottleneck state into an expected category.

    Returns the category key or None if unclassified.
    """
    sig = state.get("active_tab_signature", "")
    dialog = state.get("dialog_state", "none")
    tabs = [t.strip() for t in sig.split(",") if t.strip()]

    if app == "excel":
        # Localized UI
        if _is_localized(dialog):
            return "localized_ui"
        # Dialog chain
        if dialog != "none":
            lower = dialog.lower()
            for kw in DIALOG_CHAIN_KEYWORDS:
                if kw in lower:
                    return "dialog_chain"
            # Any non-none dialog is likely a dialog chain state
            return "dialog_chain"
        # Ribbon nav chain: Data-only or missing Home
        if _missing_tab(sig, "Home") and len(tabs) > 0:
            return "ribbon_nav_chain"
        return None

    elif app == "word":
        # Dialog chain
        if dialog != "none":
            lower = dialog.lower()
            for kw in DIALOG_CHAIN_KEYWORDS:
                if kw in lower:
                    return "dialog_chain"
            return "dialog_chain"
        # Dual layout tabs
        if tabs.count("Layout") >= 2:
            return "table_layout_mode"
        # Developer mode with non-standard view
        if _has_tab(sig, "Developer"):
            # Check if it's a reduced/specialized view
            if _missing_tab(sig, "Home") or _missing_tab(sig, "Insert"):
                return "developer_mode"
            # Developer + Layout,Layout already caught above
            # Developer with full ribbon but no Draw (older views)
            if _missing_tab(sig, "Draw") and _missing_tab(sig, "Mailings"):
                return "developer_mode"
        # Non-standard views missing Home
        if _missing_tab(sig, "Home") and len(tabs) > 0:
            return "developer_mode"
        return None

    elif app == "ppt":
        # Dialog chain
        if dialog != "none":
            lower = dialog.lower()
            for kw in DIALOG_CHAIN_KEYWORDS:
                if kw in lower:
                    return "dialog_chain"
            return "dialog_chain"
        # Embedded object editing: Data/Formulas tabs present in PPT
        if _has_tab(sig, "Data") or _has_tab(sig, "Formulas"):
            return "embedded_object_edit"
        # Non-standard view: missing Home
        if _missing_tab(sig, "Home") and len(tabs) > 0:
            return "non_standard_view"
        return None

    return None


def run_category_classification(bottlenecks: list[dict], app: str,
                                 all_states: dict) -> dict:
    """Classify bottlenecks and compute precision/recall per category."""
    classifications = {}
    category_counts = Counter()
    for b in bottlenecks:
        cat = classify_bottleneck(b, app)
        classifications[b["hash"]] = cat
        if cat is not None:
            category_counts[cat] += 1

    n_classified = sum(1 for v in classifications.values() if v is not None)
    precision = n_classified / len(bottlenecks) if bottlenecks else 0.0

    # Recall: for each category, how many states in the full registry
    # that match the category pattern are actually in bottlenecks
    bottleneck_hashes = {b["hash"] for b in bottlenecks}
    app_states = {h: s for h, s in all_states.items()
                  if s.get("app_domain") == app}

    per_category = {}
    for cat_key in EXPECTED_CATEGORIES[app]:
        cat_total = 0
        cat_in_bottleneck = 0
        for h, s in app_states.items():
            c = classify_bottleneck(s, app)
            if c == cat_key:
                cat_total += 1
                if h in bottleneck_hashes:
                    cat_in_bottleneck += 1
        recall = cat_in_bottleneck / cat_total if cat_total > 0 else 0.0
        per_category[cat_key] = {
            "description": EXPECTED_CATEGORIES[app][cat_key],
            "bottleneck_count": category_counts.get(cat_key, 0),
            "total_matching_states": cat_total,
            "recall": round(recall, 4),
        }

    categories_with_bottlenecks = sum(
        1 for c in per_category.values() if c["bottleneck_count"] > 0
    )

    return {
        "classifications": classifications,
        "precision": round(precision, 4),
        "per_category": per_category,
        "categories_identified": categories_with_bottlenecks,
        "category_counts": dict(category_counts),
    }


# ============================================================
# Module 2: Bottleneck clustering
# ============================================================

def _tab_category(sig: str) -> str:
    """Reduce a tab signature to a human-readable category."""
    tabs = [t.strip() for t in sig.split(",") if t.strip()]
    if not tabs:
        return "empty_ribbon"
    has_home = "Home" in tabs
    has_data = "Data" in tabs
    has_formulas = "Formulas" in tabs
    has_developer = "Developer" in tabs
    n_layout = tabs.count("Layout")
    if n_layout >= 2:
        return "dual_layout"
    if has_data and has_formulas and not has_home:
        return "data_only_view"
    if not has_home and has_data:
        return "reduced_ribbon_data"
    if not has_home:
        return "reduced_ribbon"
    if has_developer:
        return "full_ribbon_developer"
    return "standard_ribbon"


def run_bottleneck_clustering(bottlenecks: list[dict]) -> dict:
    """Cluster bottlenecks by (tab_category, dialog_state)."""
    clusters = defaultdict(list)
    for b in bottlenecks:
        tc = _tab_category(b.get("active_tab_signature", ""))
        ds = b.get("dialog_state", "none")
        key = f"{tc}|{ds}"
        clusters[key].append(b["hash"])

    # Merge small clusters (< 3 members) into "other"
    MIN_CLUSTER = 3
    merged = {}
    other = []
    for key, members in clusters.items():
        if len(members) >= MIN_CLUSTER:
            merged[key] = members
        else:
            other.extend(members)
    if other:
        merged["other|misc"] = other

    # Build summary
    summary = []
    for key, members in sorted(merged.items(), key=lambda x: -len(x[1])):
        tc, ds = key.split("|", 1)
        summary.append({
            "cluster_label": key,
            "tab_category": tc,
            "dialog_state": ds,
            "count": len(members),
        })

    return {
        "clusters": {k: v for k, v in merged.items()},
        "cluster_summary": summary,
        "num_clusters": len(merged),
    }


# ============================================================
# Module 3: Graph structure validation
# ============================================================

def run_graph_validation(adjacency: dict, f_values_map: dict,
                          threshold: float, app: str,
                          state_registry: dict) -> dict:
    """Compute cut-ratio, normalized cut, and conductance on the app subgraph."""
    # Build app-specific subgraph
    app_states = {h for h, s in state_registry.items()
                  if s.get("app_domain") == app}

    # Partition: S = bottleneck (f < threshold), Sbar = non-bottleneck
    S = set()
    Sbar = set()
    for h in app_states:
        if h in f_values_map:
            if f_values_map[h] < threshold:
                S.add(h)
            else:
                Sbar.add(h)

    if not S or not Sbar:
        return {
            "cut_ratio": float("nan"),
            "normalized_cut": float("nan"),
            "conductance": float("nan"),
            "num_bottleneck_states": len(S),
            "num_non_bottleneck_states": len(Sbar),
        }

    # Count edges
    cut_edges = 0
    total_edges = 0
    vol_S = 0   # sum of degrees in S
    vol_Sbar = 0

    for src, neighbors in adjacency.items():
        if src not in app_states:
            continue
        for dst, weight in neighbors.items():
            if dst not in app_states:
                continue
            w = int(weight)
            total_edges += w
            if src in S:
                vol_S += w
            else:
                vol_Sbar += w
            # Cut edge: crosses the partition
            if (src in S and dst in Sbar) or (src in Sbar and dst in S):
                cut_edges += w

    cut_ratio = cut_edges / total_edges if total_edges > 0 else float("nan")

    # Normalized cut: cut(S, Sbar) * (1/vol(S) + 1/vol(Sbar))
    if vol_S > 0 and vol_Sbar > 0:
        ncut = cut_edges * (1.0 / vol_S + 1.0 / vol_Sbar)
    else:
        ncut = float("nan")

    # Conductance: cut(S, Sbar) / min(vol(S), vol(Sbar))
    min_vol = min(vol_S, vol_Sbar)
    conductance = cut_edges / min_vol if min_vol > 0 else float("nan")

    return {
        "cut_ratio": round(cut_ratio, 6),
        "normalized_cut": round(ncut, 6),
        "conductance": round(conductance, 6),
        "cut_edges": cut_edges,
        "total_edges": total_edges,
        "vol_S": vol_S,
        "vol_Sbar": vol_Sbar,
        "num_bottleneck_states": len(S),
        "num_non_bottleneck_states": len(Sbar),
    }


# ============================================================
# Module 4: Trajectory-level analysis
# ============================================================

def run_trajectory_analysis(transitions_path: Path,
                             f_values_map: dict,
                             app: str) -> dict:
    """Analyze per-trajectory f-value crossing behavior."""
    # Group transitions by execution_id for this app
    trajectories = defaultdict(list)
    with open(transitions_path) as f:
        for line in f:
            t = json.loads(line)
            eid = t["execution_id"]
            if not eid.startswith(app):
                continue
            trajectories[eid].append(t)

    # Sort each trajectory by step_id
    for eid in trajectories:
        trajectories[eid].sort(key=lambda x: x["step_id"])

    crossing_thresholds = [1.0, 2.0, 3.0]
    crossing_counts = {t: 0 for t in crossing_thresholds}
    total_traj = len(trajectories)

    max_jumps = []       # max single-step f-value jump per trajectory (all)
    max_jumps_crossing = []  # max jumps for crossing trajectories only
    traj_ranges = []     # f-value range per trajectory
    traj_lengths_crossing = []
    traj_lengths_non_crossing = []

    for eid, steps in trajectories.items():
        # Build f-value sequence for this trajectory
        hashes = [steps[0]["state_hash"]]
        for s in steps:
            hashes.append(s["next_state_hash"])

        fvals = []
        for h in hashes:
            if h in f_values_map:
                fvals.append(f_values_map[h])

        if len(fvals) < 2:
            continue

        frange = max(fvals) - min(fvals)
        traj_ranges.append(frange)

        # Max single-step jump
        jumps = [abs(fvals[i+1] - fvals[i]) for i in range(len(fvals)-1)]
        max_jump = max(jumps) if jumps else 0.0
        max_jumps.append(max_jump)

        for thresh in crossing_thresholds:
            if frange > thresh:
                crossing_counts[thresh] += 1

        if frange > 1.0:
            traj_lengths_crossing.append(len(steps))
            max_jumps_crossing.append(max_jump)
        else:
            traj_lengths_non_crossing.append(len(steps))

    crossing_rates = {
        str(t): round(crossing_counts[t] / total_traj, 4) if total_traj > 0 else 0.0
        for t in crossing_thresholds
    }

    avg_max_jump = float(np.mean(max_jumps)) if max_jumps else 0.0
    median_max_jump = float(np.median(max_jumps)) if max_jumps else 0.0
    avg_max_jump_crossing = float(np.mean(max_jumps_crossing)) if max_jumps_crossing else 0.0

    avg_len_crossing = float(np.mean(traj_lengths_crossing)) if traj_lengths_crossing else 0.0
    avg_len_non_crossing = float(np.mean(traj_lengths_non_crossing)) if traj_lengths_non_crossing else 0.0

    return {
        "total_trajectories": total_traj,
        "crossing_rates": crossing_rates,
        "avg_max_single_step_jump": round(avg_max_jump, 4),
        "avg_max_single_step_jump_crossing": round(avg_max_jump_crossing, 4),
        "median_max_single_step_jump": round(median_max_jump, 4),
        "avg_length_crossing": round(avg_len_crossing, 2),
        "avg_length_non_crossing": round(avg_len_non_crossing, 2),
        "num_crossing_trajectories": len(traj_lengths_crossing),
        "num_non_crossing_trajectories": len(traj_lengths_non_crossing),
        "traj_ranges": traj_ranges,
        "max_jumps": max_jumps,
    }


# ============================================================
# Module 5: Correlation and distribution analysis
# ============================================================

def _bimodality_coefficient(values: np.ndarray) -> float:
    """Compute Sarle's bimodality coefficient: BC = (skew^2 + 1) / kurtosis."""
    n = len(values)
    if n < 4:
        return float("nan")
    skew = float(stats.skew(values))
    kurt = float(stats.kurtosis(values, fisher=False))  # excess=False → Pearson
    if kurt == 0:
        return float("nan")
    return (skew ** 2 + 1) / kurt


def _gap_ratio(values: np.ndarray, n_bins: int = 50) -> float:
    """Ratio of the deepest valley to the highest peak in the histogram."""
    counts, _ = np.histogram(values, bins=n_bins)
    if counts.max() == 0:
        return float("nan")
    peak = counts.max()
    # Find the deepest valley between two peaks
    valleys = []
    for i in range(1, len(counts) - 1):
        if counts[i] < counts[i-1] and counts[i] < counts[i+1]:
            valleys.append(counts[i])
    if not valleys:
        return 0.0
    deepest_valley = min(valleys)
    return round(1.0 - deepest_valley / peak, 4)


def run_correlation_analysis(f_values_map: dict, adjacency: dict,
                              state_registry: dict, app: str) -> dict:
    """Compute f-value vs degree correlations and distribution statistics."""
    app_states = {h for h, s in state_registry.items()
                  if s.get("app_domain") == app}

    # Compute degree for each state
    out_degree = defaultdict(int)
    in_degree = defaultdict(int)
    visit_freq = defaultdict(int)

    for src, neighbors in adjacency.items():
        if src not in app_states:
            continue
        for dst, weight in neighbors.items():
            if dst not in app_states:
                continue
            w = int(weight)
            out_degree[src] += w
            in_degree[dst] += w
            visit_freq[src] += w

    # Align arrays
    hashes_common = [h for h in app_states if h in f_values_map and h in out_degree]
    if len(hashes_common) < 10:
        return {
            "spearman_vs_out_degree": {"r": float("nan"), "p": float("nan")},
            "spearman_vs_in_degree": {"r": float("nan"), "p": float("nan")},
            "spearman_vs_visit_freq": {"r": float("nan"), "p": float("nan")},
            "bimodality_coefficient": float("nan"),
            "gap_ratio": float("nan"),
        }

    fv = np.array([f_values_map[h] for h in hashes_common])
    od = np.array([out_degree[h] for h in hashes_common])
    id_ = np.array([in_degree[h] for h in hashes_common])
    vf = np.array([visit_freq[h] for h in hashes_common])

    r_out, p_out = stats.spearmanr(fv, od)
    r_in, p_in = stats.spearmanr(fv, id_)
    r_vf, p_vf = stats.spearmanr(fv, vf)

    # Distribution stats on all f-values for this app
    all_fv = np.array([f_values_map[h] for h in app_states if h in f_values_map])
    bc = _bimodality_coefficient(all_fv)
    gr = _gap_ratio(all_fv)

    return {
        "spearman_vs_out_degree": {"r": round(float(r_out), 4), "p": round(float(p_out), 6)},
        "spearman_vs_in_degree": {"r": round(float(r_in), 4), "p": round(float(p_in), 6)},
        "spearman_vs_visit_freq": {"r": round(float(r_vf), 4), "p": round(float(p_vf), 6)},
        "bimodality_coefficient": round(bc, 4) if not np.isnan(bc) else float("nan"),
        "gap_ratio": gr,
        "num_states_analyzed": len(hashes_common),
    }


# ============================================================
# Visualization
# ============================================================

def _set_style():
    """Configure matplotlib for clean publication-style plots."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.figsize": (8, 5),
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def plot_f_value_histogram(f_values: np.ndarray, threshold: float,
                            app: str, out_path: Path):
    """F-value distribution histogram with bottleneck zone highlighted."""
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(f_values, bins=60, color="#4c72b0",
                                edgecolor="white", alpha=0.85)
    # Color bottleneck bins red
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < threshold:
            patch.set_facecolor("#c44e52")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2,
               label=f"threshold = {threshold:.3f}")
    ax.set_xlabel("f-value")
    ax.set_ylabel("Count")
    ax.set_title(f"{app.upper()} — f-value Distribution")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def plot_f_by_dialog(bottlenecks: list[dict], app: str, out_path: Path):
    """Box plot of f-values grouped by dialog state."""
    dialog_groups = defaultdict(list)
    for b in bottlenecks:
        ds = b.get("dialog_state", "none")
        dialog_groups[ds].append(b["f_value"])

    # Keep top 15 by count
    top = sorted(dialog_groups.items(), key=lambda x: -len(x[1]))[:15]
    if not top:
        return
    labels = [k[:30] for k, _ in top]
    data = [v for _, v in top]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, vert=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#8172b2")
        patch.set_alpha(0.7)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("f-value")
    ax.set_title(f"{app.upper()} — f-value by Dialog State (top 15)")
    fig.savefig(out_path)
    plt.close(fig)


def plot_f_by_tab_category(bottlenecks: list[dict], app: str, out_path: Path):
    """Box plot of f-values grouped by tab category."""
    tc_groups = defaultdict(list)
    for b in bottlenecks:
        tc = _tab_category(b.get("active_tab_signature", ""))
        tc_groups[tc].append(b["f_value"])

    top = sorted(tc_groups.items(), key=lambda x: -len(x[1]))
    if not top:
        return
    labels = [k for k, _ in top]
    data = [v for _, v in top]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, vert=True, patch_artist=True)
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2", "#937860"]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.7)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("f-value")
    ax.set_title(f"{app.upper()} — f-value by Tab Category")
    fig.savefig(out_path)
    plt.close(fig)


def plot_cluster_summary(cluster_summary: list[dict], app: str, out_path: Path):
    """Bar chart of bottleneck clusters."""
    if not cluster_summary:
        return
    top = cluster_summary[:15]
    labels = [c["cluster_label"][:40] for c in top]
    counts = [c["count"] for c in top]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(labels)), counts, color="#4c72b0", alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Count")
    ax.set_title(f"{app.upper()} — Bottleneck Cluster Summary")
    ax.invert_yaxis()
    fig.savefig(out_path)
    plt.close(fig)


def plot_cross_app_kde(all_f_values: dict[str, np.ndarray], out_path: Path):
    """KDE overlay of f-value distributions for all apps."""
    fig, ax = plt.subplots()
    colors = {"excel": "#4c72b0", "word": "#dd8452", "ppt": "#55a868"}
    for app_name, fv in all_f_values.items():
        if len(fv) < 2:
            continue
        kde = stats.gaussian_kde(fv)
        x = np.linspace(fv.min() - 0.5, fv.max() + 0.5, 300)
        ax.plot(x, kde(x), label=app_name.upper(), color=colors.get(app_name, "gray"),
                linewidth=2)
        ax.fill_between(x, kde(x), alpha=0.15, color=colors.get(app_name, "gray"))
    ax.set_xlabel("f-value")
    ax.set_ylabel("Density")
    ax.set_title("Cross-App f-value Distribution (KDE)")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def plot_crossing_trajectories(traj_results: dict[str, dict], out_path: Path):
    """Scatter plot: trajectory length vs f-value range."""
    fig, axes = plt.subplots(1, len(traj_results), figsize=(5*len(traj_results), 5),
                              squeeze=False)
    colors = {"excel": "#4c72b0", "word": "#dd8452", "ppt": "#55a868"}
    for idx, (app_name, res) in enumerate(traj_results.items()):
        ax = axes[0, idx]
        ranges = res.get("traj_ranges", [])
        jumps = res.get("max_jumps", [])
        if ranges and jumps:
            ax.scatter(ranges, jumps, alpha=0.3, s=15,
                       color=colors.get(app_name, "gray"))
            ax.set_xlabel("f-value range")
            ax.set_ylabel("Max single-step jump")
            ax.set_title(f"{app_name.upper()}")
    fig.suptitle("Trajectory f-value Range vs Max Jump", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_expected_vs_identified(cat_results: dict[str, dict], out_path: Path):
    """Grouped bar chart: expected categories vs identified bottleneck counts."""
    fig, axes = plt.subplots(1, len(cat_results), figsize=(5*len(cat_results), 5),
                              squeeze=False)
    for idx, (app_name, res) in enumerate(cat_results.items()):
        ax = axes[0, idx]
        cats = list(res["per_category"].keys())
        total = [res["per_category"][c]["total_matching_states"] for c in cats]
        bn = [res["per_category"][c]["bottleneck_count"] for c in cats]

        x = np.arange(len(cats))
        w = 0.35
        ax.bar(x - w/2, total, w, label="All matching", color="#8da0cb", alpha=0.8)
        ax.bar(x + w/2, bn, w, label="Bottleneck", color="#c44e52", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=8)
        ax.set_ylabel("Count")
        ax.set_title(f"{app_name.upper()}")
        ax.legend(fontsize=8)
    fig.suptitle("Expected Categories vs Identified Bottlenecks", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_coverage_heatmap(cat_results: dict[str, dict], out_path: Path):
    """Heatmap of recall per (app, category)."""
    apps = list(cat_results.keys())
    all_cats = []
    for app_name in apps:
        for c in cat_results[app_name]["per_category"]:
            if c not in all_cats:
                all_cats.append(c)

    matrix = np.full((len(apps), len(all_cats)), np.nan)
    for i, app_name in enumerate(apps):
        for j, cat in enumerate(all_cats):
            if cat in cat_results[app_name]["per_category"]:
                matrix[i, j] = cat_results[app_name]["per_category"][cat]["recall"]

    fig, ax = plt.subplots(figsize=(max(8, len(all_cats)*1.5), 4))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(len(all_cats)))
    ax.set_xticklabels([c.replace("_", "\n") for c in all_cats],
                        fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(apps)))
    ax.set_yticklabels([a.upper() for a in apps])

    # Annotate cells
    for i in range(len(apps)):
        for j in range(len(all_cats)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color="black" if val < 0.5 else "white")

    fig.colorbar(im, ax=ax, label="Recall")
    ax.set_title("Category Coverage (Recall) Heatmap")
    fig.savefig(out_path)
    plt.close(fig)


# ============================================================
# Go/No-Go evaluation
# ============================================================

def evaluate_go_no_go(app_results: dict[str, dict]) -> dict:
    """Apply the 5 criteria across all apps."""
    criteria = []

    # Criterion 1: Each app has >= 2 expected categories identified
    c1_details = {}
    c1_pass = True
    for app_name, res in app_results.items():
        n = res["category"]["categories_identified"]
        c1_details[app_name] = n
        if n < 2:
            c1_pass = False
    criteria.append({
        "id": 1,
        "description": "Each app has >= 2 expected categories identified",
        "threshold": ">= 2",
        "passed": c1_pass,
        "details": c1_details,
    })

    # Criterion 2: >= 40% of bottleneck states belong to expected categories
    c2_details = {}
    c2_pass = True
    for app_name, res in app_results.items():
        p = res["category"]["precision"]
        c2_details[app_name] = p
        if p < 0.40:
            c2_pass = False
    criteria.append({
        "id": 2,
        "description": ">= 40% of bottleneck states belong to expected categories",
        "threshold": ">= 0.40",
        "passed": c2_pass,
        "details": c2_details,
    })

    # Criterion 3: |Spearman r| between f-value and node degree < 0.3
    c3_details = {}
    c3_pass = True
    for app_name, res in app_results.items():
        r = abs(res["correlation"]["spearman_vs_out_degree"]["r"])
        c3_details[app_name] = round(r, 4)
        if r >= 0.3:
            c3_pass = False
    criteria.append({
        "id": 3,
        "description": "|Spearman r| between f-value and node degree < 0.3",
        "threshold": "< 0.3",
        "passed": c3_pass,
        "details": c3_details,
    })

    # Criterion 4: Avg max single-step f-value jump > 0.5 for crossing trajectories
    c4_details = {}
    c4_pass = True
    for app_name, res in app_results.items():
        j = res["trajectory"]["avg_max_single_step_jump_crossing"]
        c4_details[app_name] = j
        if j <= 0.5:
            c4_pass = False
    criteria.append({
        "id": 4,
        "description": "Avg max single-step f-value jump > 0.5",
        "threshold": "> 0.5",
        "passed": c4_pass,
        "details": c4_details,
    })

    # Criterion 5: Normalized cut < 0.5 for each app
    c5_details = {}
    c5_pass = True
    for app_name, res in app_results.items():
        nc = res["graph"]["normalized_cut"]
        c5_details[app_name] = nc
        if np.isnan(nc) or nc >= 0.5:
            c5_pass = False
    criteria.append({
        "id": 5,
        "description": "Normalized cut < 0.5 for each app",
        "threshold": "< 0.5",
        "passed": c5_pass,
        "details": c5_details,
    })

    all_pass = all(c["passed"] for c in criteria)
    return {
        "verdict": "GO" if all_pass else "NO-GO",
        "criteria": criteria,
        "passed_count": sum(1 for c in criteria if c["passed"]),
        "total_count": len(criteria),
    }


# ============================================================
# Report generation
# ============================================================

def generate_markdown_report(report: dict) -> str:
    """Generate a human-readable Markdown report."""
    lines = []
    lines.append("# Bottleneck Validation Report")
    lines.append("")
    lines.append(f"**Generated:** {report['metadata']['timestamp']}")
    lines.append(f"**Apps analyzed:** {', '.join(report['metadata']['apps'])}")
    lines.append("")

    # Go/No-Go
    gng = report["go_no_go"]
    verdict = gng["verdict"]
    emoji = "PASS" if verdict == "GO" else "FAIL"
    lines.append(f"## Go/No-Go Verdict: **{verdict}** ({gng['passed_count']}/{gng['total_count']} criteria passed)")
    lines.append("")
    lines.append("| # | Criterion | Threshold | Passed | Details |")
    lines.append("|---|-----------|-----------|--------|---------|")
    for c in gng["criteria"]:
        passed = "YES" if c["passed"] else "NO"
        details_str = ", ".join(f"{k}: {v}" for k, v in c["details"].items())
        lines.append(f"| {c['id']} | {c['description']} | {c['threshold']} | {passed} | {details_str} |")
    lines.append("")

    # Per-app results
    for app_name, res in report["per_app"].items():
        lines.append(f"## {app_name.upper()}")
        lines.append("")

        # Category classification
        cat = res["category"]
        lines.append(f"### Category Classification (precision: {cat['precision']:.2%})")
        lines.append("")
        lines.append(f"Categories identified: {cat['categories_identified']}/{len(cat['per_category'])}")
        lines.append("")
        lines.append("| Category | Bottleneck Count | Total Matching | Recall |")
        lines.append("|----------|-----------------|----------------|--------|")
        for cname, cdata in cat["per_category"].items():
            lines.append(
                f"| {cname} | {cdata['bottleneck_count']} | "
                f"{cdata['total_matching_states']} | {cdata['recall']:.2%} |"
            )
        lines.append("")

        # Clustering
        clust = res["clustering"]
        lines.append(f"### Clustering ({clust['num_clusters']} clusters)")
        lines.append("")
        lines.append("| Cluster | Tab Category | Dialog | Count |")
        lines.append("|---------|-------------|--------|-------|")
        for cs in clust["cluster_summary"][:10]:
            lines.append(
                f"| {cs['cluster_label'][:50]} | {cs['tab_category']} | "
                f"{cs['dialog_state'][:30]} | {cs['count']} |"
            )
        lines.append("")

        # Graph validation
        gv = res["graph"]
        lines.append("### Graph Structure")
        lines.append("")
        lines.append(f"- Cut ratio: {gv['cut_ratio']:.4f}")
        lines.append(f"- Normalized cut: {gv['normalized_cut']:.4f}")
        lines.append(f"- Conductance: {gv['conductance']:.4f}")
        lines.append(f"- Bottleneck states: {gv['num_bottleneck_states']}, "
                      f"Non-bottleneck: {gv['num_non_bottleneck_states']}")
        lines.append("")

        # Trajectory analysis
        tj = res["trajectory"]
        lines.append("### Trajectory Analysis")
        lines.append("")
        lines.append(f"- Total trajectories: {tj['total_trajectories']}")
        lines.append(f"- Crossing rates: {tj['crossing_rates']}")
        lines.append(f"- Avg max single-step jump (all): {tj['avg_max_single_step_jump']:.4f}")
        lines.append(f"- Avg max single-step jump (crossing only): {tj['avg_max_single_step_jump_crossing']:.4f}")
        lines.append(f"- Median max single-step jump: {tj['median_max_single_step_jump']:.4f}")
        lines.append(f"- Avg trajectory length (crossing): {tj['avg_length_crossing']:.1f}")
        lines.append(f"- Avg trajectory length (non-crossing): {tj['avg_length_non_crossing']:.1f}")
        lines.append("")

        # Correlation
        corr = res["correlation"]
        lines.append("### Correlation Analysis")
        lines.append("")
        lines.append(f"- Spearman vs out-degree: r={corr['spearman_vs_out_degree']['r']:.4f}, "
                      f"p={corr['spearman_vs_out_degree']['p']:.6f}")
        lines.append(f"- Spearman vs in-degree: r={corr['spearman_vs_in_degree']['r']:.4f}, "
                      f"p={corr['spearman_vs_in_degree']['p']:.6f}")
        lines.append(f"- Spearman vs visit frequency: r={corr['spearman_vs_visit_freq']['r']:.4f}, "
                      f"p={corr['spearman_vs_visit_freq']['p']:.6f}")
        lines.append(f"- Bimodality coefficient (Sarle's BC): {corr['bimodality_coefficient']:.4f}")
        lines.append(f"- Gap ratio: {corr['gap_ratio']:.4f}")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def load_data(apps: list[str]) -> dict:
    """Load all input data files."""
    data = {
        "f_values": {},      # app -> {hash: f_value}
        "f_arrays": {},      # app -> np.ndarray
        "bottlenecks": {},   # app -> list[dict]
        "results": {},       # app -> dict
        "adjacency": None,
        "state_registry": None,
        "transitions_path": TRANSITIONS_DIR / "transitions.jsonl",
    }

    for app in apps:
        app_dir = FNET_DIR / app

        # f_values.npz
        npz = np.load(app_dir / "f_values.npz", allow_pickle=True)
        hashes = npz["hashes"]
        fvals = npz["f_values"]
        data["f_values"][app] = {h: float(v) for h, v in zip(hashes, fvals)}
        data["f_arrays"][app] = fvals

        # bottlenecks_described.json
        with open(app_dir / "bottlenecks_described.json") as f:
            data["bottlenecks"][app] = json.load(f)

        # results.json
        with open(app_dir / "results.json") as f:
            data["results"][app] = json.load(f)

    # Shared files
    with open(TRANSITIONS_DIR / "adjacency.json") as f:
        data["adjacency"] = json.load(f)

    with open(TRANSITIONS_DIR / "state_registry.json") as f:
        data["state_registry"] = json.load(f)

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Task 4: Validate bottleneck identification results"
    )
    parser.add_argument("--app", choices=APPS + ["all"], default="all",
                        help="App to validate (default: all)")
    parser.add_argument("--percentile-k", type=float, default=None,
                        help="Override percentile threshold for bottlenecks")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory")
    args = parser.parse_args()

    _set_style()

    apps = APPS if args.app == "all" else [args.app]
    output_dir = Path(args.output_dir)

    logger.info("Loading data for apps: %s", apps)
    t0 = time.time()
    data = load_data(apps)
    logger.info("Data loaded in %.1fs", time.time() - t0)

    app_results = {}
    all_f_values_for_plot = {}
    all_traj_results = {}
    all_cat_results = {}

    for app in apps:
        logger.info("=" * 60)
        logger.info("Processing %s (%d bottlenecks, %d states)",
                     app.upper(), len(data["bottlenecks"][app]),
                     len(data["f_values"][app]))

        app_out = output_dir / app
        app_out.mkdir(parents=True, exist_ok=True)

        threshold = data["results"][app]["bottleneck_threshold"]
        if args.percentile_k is not None:
            # Recompute threshold from the percentile
            fv_arr = data["f_arrays"][app]
            threshold = float(np.percentile(fv_arr, args.percentile_k))
            logger.info("Using custom percentile_k=%.1f → threshold=%.4f",
                         args.percentile_k, threshold)

        # Module 1: Category classification
        logger.info("[%s] Running category classification...", app)
        cat_result = run_category_classification(
            data["bottlenecks"][app], app, data["state_registry"]
        )
        all_cat_results[app] = cat_result
        logger.info("[%s] Precision: %.2f%%, Categories identified: %d",
                     app, cat_result["precision"] * 100,
                     cat_result["categories_identified"])

        # Module 2: Clustering
        logger.info("[%s] Running bottleneck clustering...", app)
        cluster_result = run_bottleneck_clustering(data["bottlenecks"][app])
        logger.info("[%s] Found %d clusters", app, cluster_result["num_clusters"])

        # Module 3: Graph validation
        logger.info("[%s] Running graph structure validation...", app)
        graph_result = run_graph_validation(
            data["adjacency"], data["f_values"][app],
            threshold, app, data["state_registry"]
        )
        logger.info("[%s] Ncut=%.4f, Conductance=%.4f",
                     app, graph_result["normalized_cut"],
                     graph_result["conductance"])

        # Module 4: Trajectory analysis
        logger.info("[%s] Running trajectory analysis...", app)
        traj_result = run_trajectory_analysis(
            data["transitions_path"], data["f_values"][app], app
        )
        all_traj_results[app] = traj_result
        logger.info("[%s] Crossing rate (>1.0): %.2f%%, Avg max jump: %.4f",
                     app, traj_result["crossing_rates"]["1.0"] * 100,
                     traj_result["avg_max_single_step_jump"])

        # Module 5: Correlation analysis
        logger.info("[%s] Running correlation analysis...", app)
        corr_result = run_correlation_analysis(
            data["f_values"][app], data["adjacency"],
            data["state_registry"], app
        )
        logger.info("[%s] Spearman(f, out_degree) r=%.4f",
                     app, corr_result["spearman_vs_out_degree"]["r"])

        # Per-app visualizations
        logger.info("[%s] Generating per-app plots...", app)
        plot_f_value_histogram(
            data["f_arrays"][app], threshold, app,
            app_out / "f_value_histogram.png"
        )
        plot_f_by_dialog(data["bottlenecks"][app], app,
                          app_out / "f_value_by_dialog.png")
        plot_f_by_tab_category(data["bottlenecks"][app], app,
                                app_out / "f_value_by_tab_category.png")
        plot_cluster_summary(cluster_result["cluster_summary"], app,
                              app_out / "bottleneck_cluster_summary.png")

        all_f_values_for_plot[app] = data["f_arrays"][app]

        # Strip non-serializable data from traj_result before storing
        traj_result_clean = {k: v for k, v in traj_result.items()
                              if k not in ("traj_ranges", "max_jumps")}
        # Strip non-serializable data from cat_result
        cat_result_clean = {k: v for k, v in cat_result.items()
                             if k != "classifications"}

        app_results[app] = {
            "category": cat_result_clean,
            "clustering": {
                "num_clusters": cluster_result["num_clusters"],
                "cluster_summary": cluster_result["cluster_summary"],
            },
            "graph": graph_result,
            "trajectory": traj_result_clean,
            "correlation": corr_result,
        }

    # Aggregate visualizations
    logger.info("Generating aggregate plots...")
    agg_out = output_dir / "aggregate"
    agg_out.mkdir(parents=True, exist_ok=True)

    if len(apps) > 1:
        plot_cross_app_kde(all_f_values_for_plot, agg_out / "cross_app_f_distribution.png")
        plot_crossing_trajectories(all_traj_results, agg_out / "crossing_trajectory_analysis.png")
        plot_expected_vs_identified(all_cat_results, agg_out / "expected_vs_identified.png")
        plot_coverage_heatmap(all_cat_results, agg_out / "category_coverage_heatmap.png")

    # Go/No-Go evaluation
    logger.info("Evaluating Go/No-Go criteria...")
    go_no_go = evaluate_go_no_go(app_results)
    logger.info("Verdict: %s (%d/%d passed)",
                 go_no_go["verdict"], go_no_go["passed_count"],
                 go_no_go["total_count"])

    # Build full report
    report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "apps": apps,
            "output_dir": str(output_dir),
        },
        "go_no_go": go_no_go,
        "per_app": app_results,
    }

    # Write JSON report
    report_json_path = output_dir / "report.json"
    report_json_path.parent.mkdir(parents=True, exist_ok=True)

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=2, cls=NpEncoder)
    logger.info("JSON report: %s", report_json_path)

    # Write Markdown report
    md_path = output_dir / "report.md"
    md_content = generate_markdown_report(report)
    with open(md_path, "w") as f:
        f.write(md_content)
    logger.info("Markdown report: %s", md_path)

    # Print summary
    print("\n" + "=" * 60)
    print(f"  VERDICT: {go_no_go['verdict']}  "
          f"({go_no_go['passed_count']}/{go_no_go['total_count']} criteria passed)")
    print("=" * 60)
    for c in go_no_go["criteria"]:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] #{c['id']}: {c['description']}")
        for k, v in c["details"].items():
            print(f"         {k}: {v}")
    print("=" * 60)
    print(f"\nOutputs: {output_dir}")
    total_time = time.time() - t0
    print(f"Total time: {total_time:.1f}s")

    return 0 if go_no_go["verdict"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
