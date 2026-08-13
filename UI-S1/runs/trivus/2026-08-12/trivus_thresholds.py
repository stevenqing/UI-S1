import math

import numpy as np


FAMILY_CELLS = {
    "mind2web": ("C_uni", "C_cond", "C_rand", "C_self"),
    "screenspot_pro": ("C_uni", "C_cond", "C_rand", "C_self"),
    "androidcontrol": ("low", "high"),
}
REQUIRED_THRESHOLD_FIELDS = {
    "context_key", "family", "cell", "changed", "margin", "wrong_score",
    "direct_success", "fallback_success", "direct_index", "fallback_index",
}


def validate_threshold_rows(rows):
    if not rows:
        raise ValueError("TriVUS threshold rows are empty")
    seen = set()
    for row in rows:
        if not REQUIRED_THRESHOLD_FIELDS.issubset(row):
            raise ValueError("TriVUS threshold row schema mismatch")
        context_key = row["context_key"]
        if not isinstance(context_key, str) or not context_key or context_key in seen:
            raise ValueError("TriVUS threshold context identity mismatch")
        seen.add(context_key)
        if row["family"] not in FAMILY_CELLS or row["cell"] not in FAMILY_CELLS[row["family"]]:
            raise ValueError("TriVUS threshold family/cell mismatch")
        if any(type(row[key]) is not bool for key in ("changed", "direct_success", "fallback_success")):
            raise ValueError("TriVUS threshold boolean fields mismatch")
        if not all(
            isinstance(row[key], (int, float, np.integer, np.floating))
            and not isinstance(row[key], (bool, np.bool_))
            and math.isfinite(float(row[key]))
            for key in ("margin", "wrong_score")
        ):
            raise ValueError("TriVUS threshold numeric fields mismatch")
        if not 0.0 <= float(row["wrong_score"]) <= 1.0:
            raise ValueError("TriVUS threshold wrong score outside [0,1]")
        if (
            type(row["direct_index"]) is not int
            or type(row["fallback_index"]) is not int
            or row["direct_index"] < 0
            or row["fallback_index"] < 0
        ):
            raise ValueError("TriVUS threshold indices mismatch")
        if row["changed"] != (
            int(row["direct_index"]) != int(row["fallback_index"])
        ):
            raise ValueError("TriVUS threshold changed/index mismatch")
    return True


def axis_candidates(values):
    positive = [float(value) for value in values if float(value) > 0]
    output = {0.0, float("inf")}
    if positive:
        output.update(
            float(np.quantile(positive, quantile))
            for quantile in np.linspace(0.0, 1.0, 11)
        )
    return sorted(output)


def threshold_grid(rows):
    changed = [row for row in rows if row["changed"]]
    margins = axis_candidates(row["margin"] for row in changed)
    wrong = axis_candidates(row["wrong_score"] for row in changed)
    return tuple((margin, wrong_score) for margin in margins for wrong_score in wrong)


def apply_threshold(rows, threshold):
    validate_threshold_rows(rows)
    margin_threshold, wrong_threshold = threshold
    values = {}
    wins = losses = overrides = 0
    for row in rows:
        override = bool(
            row["changed"]
            and row["margin"] >= margin_threshold
            and row["wrong_score"] >= wrong_threshold
        )
        values[row["context_key"]] = bool(
            row["direct_success"] if override else row["fallback_success"]
        )
        wins += int(override and row["direct_success"] and not row["fallback_success"])
        losses += int(override and row["fallback_success"] and not row["direct_success"])
        overrides += int(override)
    return values, {
        "point_delta": (wins - losses) / len(rows),
        "wins": wins,
        "losses": losses,
        "overrides": overrides,
    }


def select_cell_threshold(rows, mde):
    candidates = []
    for threshold in threshold_grid(rows):
        _, report = apply_threshold(rows, threshold)
        if report["point_delta"] >= -0.5 * mde - 1e-15:
            candidates.append((
                report["point_delta"], threshold[1], threshold[0], threshold, report,
            ))
    if not candidates:
        raise AssertionError("TriVUS infinite cell threshold must be eligible")
    selected = max(candidates)
    return selected[3], selected[4]


def select_family_threshold(rows_by_cell, mde):
    expected = tuple(rows_by_cell)
    if not expected or any(not rows_by_cell[cell] for cell in expected):
        raise ValueError("TriVUS family threshold cells are empty")
    pooled = [row for cell in expected for row in rows_by_cell[cell]]
    candidates = []
    for threshold in threshold_grid(pooled):
        reports = {
            cell: apply_threshold(rows_by_cell[cell], threshold)[1]
            for cell in expected
        }
        mean = float(np.mean([reports[cell]["point_delta"] for cell in expected]))
        eligible = all(
            reports[cell]["point_delta"] >= -0.5 * mde - 1e-15
            for cell in expected
        ) and mean >= -0.25 * mde - 1e-15
        if eligible:
            candidates.append((mean, threshold[1], threshold[0], threshold, reports))
    if not candidates:
        raise AssertionError("TriVUS infinite family threshold must be eligible")
    selected = max(candidates)
    return selected[3], {"equal_cell_delta": selected[0], "cells": selected[4]}


def select_thresholds(rows, mde, minimum_opportunities=200, included_families=None):
    validate_threshold_rows(rows)
    included = tuple(FAMILY_CELLS) if included_families is None else tuple(included_families)
    if not included or len(set(included)) != len(included) or any(
        family not in FAMILY_CELLS for family in included
    ):
        raise ValueError("TriVUS threshold included-family mismatch")
    if set(row["family"] for row in rows) != set(included):
        raise ValueError("TriVUS threshold row family coverage mismatch")
    report = {"families": {}}
    for family in included:
        cells = FAMILY_CELLS[family]
        rows_by_cell = {
            cell: [
                row for row in rows
                if row["family"] == family and row["cell"] == cell
            ]
            for cell in cells
        }
        pooled, pooled_report = select_family_threshold(rows_by_cell, mde[family])
        cell_reports = {}
        for cell in cells:
            opportunities = sum(row["changed"] for row in rows_by_cell[cell])
            if opportunities >= minimum_opportunities:
                threshold, selection = select_cell_threshold(
                    rows_by_cell[cell], mde[family]
                )
                source = "cell"
            else:
                threshold = pooled
                selection = apply_threshold(rows_by_cell[cell], threshold)[1]
                source = "family_backoff"
            cell_reports[cell] = {
                "threshold": list(threshold),
                "threshold_source": source,
                "changed_opportunities": opportunities,
                "selection": selection,
            }
        report["families"][family] = {
            "family_threshold": list(pooled),
            "family_selection": pooled_report,
            "cells": cell_reports,
        }
    return report


def apply_selected_thresholds(rows, selected, included_families=None):
    validate_threshold_rows(rows)
    included = tuple(selected["families"]) if included_families is None else tuple(included_families)
    if set(included) != set(selected["families"]):
        raise ValueError("TriVUS selected threshold family mismatch")
    if set(row["family"] for row in rows) != set(included):
        raise ValueError("TriVUS threshold application family coverage mismatch")
    output = {}
    reports = {}
    for family in included:
        cells = FAMILY_CELLS[family]
        reports[family] = {}
        for cell in cells:
            cell_rows = [
                row for row in rows
                if row["family"] == family and row["cell"] == cell
            ]
            threshold = tuple(selected["families"][family]["cells"][cell]["threshold"])
            values, report = apply_threshold(cell_rows, threshold)
            overlap = set(output) & set(values)
            if overlap:
                raise ValueError("TriVUS duplicate threshold output contexts")
            output.update(values)
            reports[family][cell] = report
    if len(output) != len(rows):
        raise ValueError("TriVUS threshold output coverage mismatch")
    return output, reports


def compose_target_only(predictions_by_spec, expected_by_family):
    mapping = {
        "mind2web": "TARGET_ONLY_MIND2WEB",
        "screenspot_pro": "TARGET_ONLY_SCREENSPOT_PRO",
        "androidcontrol": "TARGET_ONLY_ANDROIDCONTROL",
    }
    if set(predictions_by_spec) != set(mapping.values()) or set(expected_by_family) != set(mapping):
        raise ValueError("TriVUS TARGET_ONLY roster mismatch")
    output = []
    seen = set()
    for family, spec in mapping.items():
        rows = predictions_by_spec[spec]
        if {row["context_key"] for row in rows} != set(expected_by_family[family]):
            raise ValueError(f"TriVUS TARGET_ONLY coverage mismatch: {family}")
        for row in rows:
            if row["family"] != family or row["context_key"] in seen:
                raise ValueError("TriVUS TARGET_ONLY composition mismatch")
            seen.add(row["context_key"])
            output.append(row)
    return sorted(output, key=lambda row: row["context_key"])