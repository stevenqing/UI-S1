import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/assembly_prereg.yaml"
VUS_DIR = ROOT / "runs/visual-utility-selector/2026-08-11"
sys.path.insert(0, str(RUN_DIR))

from context_common import (
    assert_context_environment, checkpoint_and_fit_folds, load_jsonl,
    load_sealed_rows, sha256_file,
)
from trivus_data import (
    FAMILIES, INPUT_DIMENSION, MAX_CANDIDATES, TriVUSData, assign_weights,
    base_features, structural_features, target_values, validate_trivus_data,
)


CANDIDATE_FIELDS = {"action", "coordinate", "parameter", "parse_ok"}
CONTEXT_FIELDS = {
    "schema_version", "context_key", "sample_key", "outer_fold", "role",
    "holdout_fold", "fit_folds", "fallback_index",
}
_VALIDATED_PHASE_TOKEN = object()


@dataclass(frozen=True)
class PhaseContext:
    outer_fold: int
    role: str
    holdout_fold: int | None
    fit_folds: tuple[int, ...]
    checkpoint_fold: int | None
    applied_folds: tuple[int, ...]
    rows: tuple[dict, ...]
    expected_fold_counts: tuple[tuple[int, int], ...]
    _validation_token: object = field(repr=False, compare=False)

    def __post_init__(self):
        if self._validation_token is not _VALIDATED_PHASE_TOKEN:
            raise PermissionError("TriVUS PhaseContext must come from a validated bank scan")


def require_validated_phase(phase):
    if not isinstance(phase, PhaseContext) or phase._validation_token is not _VALIDATED_PHASE_TOKEN:
        raise PermissionError("TriVUS unvalidated phase context")
    return phase


def load_config():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config.get("status") != "FROZEN_AFTER_CONTEXT_SEAL_BEFORE_REAL_DATA_ASSEMBLY":
        raise ValueError("TriVUS assembly protocol is not frozen")
    expected = config["expected"]
    if (
        config.get("python") != ".venv-scaleup/bin/python"
        or expected.get("public_records") != 18644
        or expected.get("context_records") != 391524
        or expected.get("input_dimension") != INPUT_DIMENSION
        or expected.get("valid_candidate_counts") != [3, 12]
    ):
        raise ValueError("TriVUS assembly contract mismatch")
    if Path(sys.executable).absolute() != (ROOT / config["python"]).absolute():
        raise RuntimeError("TriVUS assembly interpreter mismatch")
    assert_context_environment(config)
    for item in config["dependencies"].values():
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"TriVUS assembly dependency mismatch: {item['path']}")
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", config["implementation_commit_floor"], "HEAD"],
        cwd=ROOT, check=False,
    )
    if completed.returncode:
        raise PermissionError("TriVUS data-primitives commit is not an ancestor")
    return config


def load_context_manifest(config):
    path = ROOT / config["dependencies"]["context_manifest"]["path"]
    manifest = json.loads(path.read_text())
    if (
        manifest.get("status") != "PASS_TRIVUS_EXACT_FALLBACK_CONTEXTS"
        or manifest.get("records") != config["expected"]["context_records"]
        or manifest.get("public_records") != config["expected"]["public_records"]
        or manifest.get("record_schema") != [
            "schema_version", "context_key", "sample_key", "outer_fold",
            "role", "holdout_fold", "fit_folds", "fallback_index",
        ]
        or manifest.get("sha256") != config["dependencies"]["contexts"]["sha256"]
        or len(manifest.get("splits", ())) != 25
        or manifest.get("candidate_success_emitted") is not False
        or manifest.get("source_identity_emitted") is not False
        or manifest.get("training_started") is not False
    ):
        raise PermissionError("TriVUS context manifest mismatch")
    return manifest


def keyed(rows, name):
    output = {row["sample_key"]: row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"TriVUS duplicate {name} sample key")
    return output


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def audit_public_row(row, config):
    family = row.get("benchmark")
    schema = "android_public" if family == "androidcontrol" else "vus_public"
    if set(row) != set(config["schemas"][schema]):
        raise ValueError(f"TriVUS public row schema mismatch: {row.get('sample_key')}")
    count = len(row.get("candidates", ()))
    if (family == "androidcontrol" and count != 3) or (family != "androidcontrol" and count != 12):
        raise ValueError(f"TriVUS public candidate count mismatch: {row['sample_key']}")
    valid_cells = {"low", "high"} if family == "androidcontrol" else {"C_uni", "C_cond", "C_rand", "C_self"}
    cell_key = "setting" if family == "androidcontrol" else "arm"
    if (
        row["schema_version"] != 1
        or family not in FAMILIES
        or row[cell_key] not in valid_cells
        or type(row["fold"]) is not int
        or row["fold"] not in range(5)
        or any(not isinstance(row[key], str) or not row[key] for key in ("sample_key", "row_id", "group", "image_path"))
        or not isinstance(row["instruction"], str)
        or not is_sha256(row["image_sha256"])
        or (family == "androidcontrol" and not isinstance(row["history"], str))
        or (family != "androidcontrol" and (
            not isinstance(row["history"], list)
            or any(not isinstance(value, str) for value in row["history"])
        ))
    ):
        raise ValueError(f"TriVUS public family/fold mismatch: {row['sample_key']}")
    if any(set(candidate) != CANDIDATE_FIELDS for candidate in row["candidates"]):
        raise ValueError(f"TriVUS public candidate schema mismatch: {row['sample_key']}")
    structural_features(row["candidates"])
    cell = row[cell_key]
    expected_key = f"{family}/{cell}/{row['row_id']}"
    if row["sample_key"] != expected_key:
        raise ValueError(f"TriVUS public sample-key mismatch: {row['sample_key']}")
    return True


def audit_prediction(prediction, public, config, expected_model_index):
    family = public["benchmark"]
    schema = "android_prediction" if family == "androidcontrol" else "vus_prediction"
    if set(prediction) != set(config["schemas"][schema]):
        raise ValueError(f"TriVUS prediction schema mismatch: {public['sample_key']}")
    cell_key = "setting" if family == "androidcontrol" else "arm"
    for key in ("sample_key", "benchmark", cell_key, "row_id", "fold", "group", "image_sha256"):
        if prediction[key] != public[key]:
            raise ValueError(f"TriVUS prediction/public mismatch: {public['sample_key']}/{key}")
    count = len(public["candidates"])
    permutation = prediction["display_to_candidate"]
    if sorted(permutation) != list(range(count)):
        raise ValueError(f"TriVUS prediction permutation mismatch: {public['sample_key']}")
    logits = prediction["label_logits"]
    probabilities = prediction["label_probabilities"]
    if (
        prediction["schema_version"] != 1
        or not is_sha256(prediction["prompt_sha256"])
        or not is_sha256(prediction["image_sha256"])
        or not is_sha256(prediction["model_index_sha256"])
        or prediction["model_index_sha256"] != expected_model_index
        or (family == "androidcontrol" and not is_sha256(prediction["overlay_sha256"]))
        or
        len(logits) != count
        or len(probabilities) != count
        or not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in logits)
        or not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1 for value in probabilities)
        or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6)
    ):
        raise ValueError(f"TriVUS prediction values mismatch: {public['sample_key']}")
    labels = "ABC" if count == 3 else "ABCDEFGHIJKL"
    selected_display = labels.index(prediction["selected_label"])
    expected_display = max(range(count), key=probabilities.__getitem__)
    if (
        selected_display != expected_display
        or prediction["selected_candidate_index"] != permutation[selected_display]
    ):
        raise ValueError(f"TriVUS prediction selected-candidate mismatch: {public['sample_key']}")
    return True


def validate_prediction_manifests(config):
    vus = json.loads((ROOT / config["dependencies"]["vus_prediction_manifest"]["path"]).read_text())
    android = json.loads((ROOT / config["dependencies"]["android_prediction_manifest"]["path"]).read_text())
    if (
        vus.get("schema_version") != 1
        or vus.get("status") != "PASS_BLIND_INFERENCE_COMPLETE"
        or vus.get("records") != config["expected"]["vus_records"]
        or vus.get("predictions_sha256") != config["dependencies"]["vus_predictions"]["sha256"]
        or vus.get("public_records_sha256") != config["dependencies"]["vus_public"]["sha256"]
        or vus.get("private_labels_opened") is not False
        or android.get("schema_version") != 1
        or android.get("status") != "PASS_TRIVUS_SELECTOR_BLIND_LOCK"
        or android.get("records") != config["expected"]["android_records"]
        or android.get("predictions_sha256") != config["dependencies"]["android_predictions"]["sha256"]
        or android.get("public_sha256") != config["dependencies"]["android_public"]["sha256"]
        or android.get("label_metrics_computed") is not False
        or android.get("model_index_sha256") != vus.get("model_index_sha256")
        or not is_sha256(vus.get("model_index_sha256"))
    ):
        raise PermissionError("TriVUS blind prediction manifest mismatch")
    return vus["model_index_sha256"]


def load_locked_public_inputs(config=None):
    config = load_config() if config is None else config
    model_index = validate_prediction_manifests(config)
    vus_public_rows = load_jsonl(ROOT / config["dependencies"]["vus_public"]["path"])
    android_public_rows = load_jsonl(ROOT / config["dependencies"]["android_public"]["path"])
    vus_prediction_rows = load_jsonl(ROOT / config["dependencies"]["vus_predictions"]["path"])
    android_prediction_rows = load_jsonl(ROOT / config["dependencies"]["android_predictions"]["path"])
    public_rows = vus_public_rows + android_public_rows
    prediction_rows = vus_prediction_rows + android_prediction_rows
    public = keyed(public_rows, "public")
    predictions = keyed(prediction_rows, "prediction")
    if (
        len(vus_public_rows) != config["expected"]["vus_records"]
        or len(android_public_rows) != config["expected"]["android_records"]
        or len(public) != config["expected"]["public_records"]
        or set(public) != set(predictions)
    ):
        raise ValueError("TriVUS locked public/prediction coverage mismatch")
    for sample_key, row in public.items():
        audit_public_row(row, config)
        audit_prediction(predictions[sample_key], row, config, model_index)
    fold_counts = Counter(int(row["fold"]) for row in public.values())
    if dict(sorted(fold_counts.items())) != config["expected"]["context_records_by_public_fold"]:
        raise ValueError("TriVUS public fold-count mismatch")
    return public, predictions


def phase_contract(outer_fold, role, holdout_fold=None):
    if outer_fold not in range(5) or role not in {"inner", "final"}:
        raise ValueError("TriVUS phase contract mismatch")
    development = tuple(fold for fold in range(5) if fold != outer_fold)
    if role == "final":
        if holdout_fold is not None:
            raise ValueError("TriVUS final phase has no holdout")
        return {
            "fit_folds": development,
            "checkpoint_fold": None,
            "holdout_fold": None,
            "applied_folds": tuple(range(5)),
        }
    if holdout_fold not in development:
        raise ValueError("TriVUS inner holdout mismatch")
    checkpoint, fit_folds = checkpoint_and_fit_folds(outer_fold, holdout_fold)
    return {
        "fit_folds": fit_folds,
        "checkpoint_fold": checkpoint,
        "holdout_fold": holdout_fold,
        "applied_folds": development,
    }


def legal_requested_folds(contract, role, requested_folds, outer_fold):
    requested = tuple(sorted(set(int(fold) for fold in requested_folds)))
    if not requested or any(fold not in range(5) for fold in requested):
        raise ValueError("TriVUS requested folds mismatch")
    if role == "inner":
        legal = {
            tuple(contract["fit_folds"]),
            (contract["checkpoint_fold"],),
            (contract["holdout_fold"],),
        }
    else:
        legal = {tuple(contract["fit_folds"]), (outer_fold,)}
    if requested not in legal:
        raise PermissionError(f"TriVUS illegal phase label request: {requested}")
    return requested


def manifest_splits(context_manifest):
    output = {}
    for split in context_manifest.get("splits", ()):
        key = (split["outer_fold"], split["role"], split["holdout_fold"])
        if key in output:
            raise ValueError(f"TriVUS duplicate manifest split: {key}")
        output[key] = split
    expected = {
        (outer, "final", None)
        for outer in range(5)
    } | {
        (outer, "inner", holdout)
        for outer in range(5) for holdout in range(5) if holdout != outer
    }
    if set(output) != expected or len(output) != 25:
        raise ValueError("TriVUS context manifest must contain all 25 splits")
    return output


def load_context_phase(
    context_path, context_manifest, public, outer_fold, role,
    expected_fold_counts, holdout_fold=None,
):
    contract = phase_contract(outer_fold, role, holdout_fold)
    phase = "final" if role == "final" else f"inner-{holdout_fold}"
    prefix = f"outer-{outer_fold}/{phase}/"
    expected_counts = {int(fold): int(count) for fold, count in expected_fold_counts.items()}
    if set(expected_counts) != set(range(5)):
        raise ValueError("TriVUS expected public fold counts mismatch")
    splits = manifest_splits(context_manifest)
    if (
        context_manifest.get("records") != sum(int(split["contexts"]) for split in splits.values())
        or context_manifest.get("public_records") != len(public)
        or context_manifest.get("record_schema") != [
            "schema_version", "context_key", "sample_key", "outer_fold",
            "role", "holdout_fold", "fit_folds", "fallback_index",
        ]
    ):
        raise ValueError("TriVUS context manifest aggregate mismatch")
    selected_key = (outer_fold, role, holdout_fold)
    if selected_key not in splits:
        raise ValueError("TriVUS selected phase absent from context manifest")
    phase_counts = Counter()
    output = []
    selected_samples = set()
    previous_key = None
    with Path(context_path).open() as handle:
        for line in handle:
            row = json.loads(line)
            if set(row) != CONTEXT_FIELDS:
                raise ValueError("TriVUS context row schema mismatch")
            row_contract = phase_contract(
                int(row["outer_fold"]), row["role"], row["holdout_fold"]
            )
            row_phase = "final" if row["role"] == "final" else f"inner-{row['holdout_fold']}"
            canonical_key = f"outer-{row['outer_fold']}/{row_phase}/{row['sample_key']}"
            if previous_key is not None and row["context_key"] <= previous_key:
                raise ValueError("TriVUS context bank order/identity mismatch")
            previous_key = row["context_key"]
            if (
                row["schema_version"] != 1
                or row["context_key"] != canonical_key
                or row["sample_key"] not in public
                or tuple(row["fit_folds"]) != tuple(row_contract["fit_folds"])
                or int(public[row["sample_key"]]["fold"]) not in row_contract["applied_folds"]
            ):
                raise ValueError(f"TriVUS context slice mismatch: {row['context_key']}")
            count = len(public[row["sample_key"]]["candidates"])
            if not 0 <= int(row["fallback_index"]) < count:
                raise ValueError(f"TriVUS context fallback mismatch: {row['context_key']}")
            phase_key = (row["outer_fold"], row["role"], row["holdout_fold"])
            phase_counts[phase_key] += 1
            if row["context_key"].startswith(prefix):
                if row["sample_key"] in selected_samples:
                    raise ValueError(f"TriVUS duplicate selected sample: {row['sample_key']}")
                selected_samples.add(row["sample_key"])
                output.append(row)
    manifest_counts = {key: int(split["contexts"]) for key, split in splits.items()}
    if dict(phase_counts) != manifest_counts:
        raise ValueError("TriVUS context bank/manifest phase-count mismatch")
    for key, split in splits.items():
        row_contract = phase_contract(key[0], key[1], key[2])
        if (
            tuple(split["fit_folds"]) != tuple(row_contract["fit_folds"])
            or split["checkpoint_fold"] != row_contract["checkpoint_fold"]
            or tuple(split["applied_folds"]) != tuple(row_contract["applied_folds"])
            or int(split["contexts"]) != sum(expected_counts[fold] for fold in row_contract["applied_folds"])
        ):
            raise ValueError(f"TriVUS context manifest split mismatch: {key}")
    expected = {
        sample_key for sample_key, row in public.items()
        if int(row["fold"]) in contract["applied_folds"]
    }
    if selected_samples != expected:
        raise ValueError("TriVUS context slice identity mismatch")
    by_fold = Counter(int(public[row["sample_key"]]["fold"]) for row in output)
    if any(by_fold[fold] != expected_counts[fold] for fold in contract["applied_folds"]):
        raise ValueError("TriVUS context phase fold-count mismatch")
    return PhaseContext(
        outer_fold=outer_fold,
        role=role,
        holdout_fold=holdout_fold,
        fit_folds=tuple(contract["fit_folds"]),
        checkpoint_fold=contract["checkpoint_fold"],
        applied_folds=tuple(contract["applied_folds"]),
        rows=tuple(sorted(output, key=lambda row: row["sample_key"])),
        expected_fold_counts=tuple(sorted(expected_counts.items())),
        _validation_token=_VALIDATED_PHASE_TOKEN,
    )


def select_phase_contexts(phase, public, requested_folds):
    require_validated_phase(phase)
    contract = {
        "fit_folds": phase.fit_folds,
        "checkpoint_fold": phase.checkpoint_fold,
        "holdout_fold": phase.holdout_fold,
        "applied_folds": phase.applied_folds,
    }
    requested = legal_requested_folds(
        contract, phase.role, requested_folds, phase.outer_fold
    )
    output = [
        row for row in phase.rows
        if int(public[row["sample_key"]]["fold"]) in requested
    ]
    expected_counts = dict(phase.expected_fold_counts)
    if len(output) != sum(expected_counts[fold] for fold in requested):
        raise ValueError("TriVUS requested context fold-count mismatch")
    return output, requested


def phase_requested_folds(phase, requested_folds):
    require_validated_phase(phase)
    contract = {
        "fit_folds": phase.fit_folds,
        "checkpoint_fold": phase.checkpoint_fold,
        "holdout_fold": phase.holdout_fold,
        "applied_folds": phase.applied_folds,
    }
    return legal_requested_folds(
        contract, phase.role, requested_folds, phase.outer_fold
    )


def _load_phase_private_labels(config, public, phase, requested_folds):
    require_validated_phase(phase)
    requested = phase_requested_folds(phase, requested_folds)
    vus_manifest = json.loads((ROOT / config["dependencies"]["vus_private_manifest"]["path"]).read_text())
    android_manifest = json.loads((ROOT / config["dependencies"]["android_private_manifest"]["path"]).read_text())
    if (
        vus_manifest.get("status") != "PASS_FOLD_SEALED_LABELS"
        or vus_manifest.get("records") != config["expected"]["vus_records"]
        or android_manifest.get("status") != "PASS_TRIVUS_FOLD_SEALED_PRIVATE_LABELS"
        or android_manifest.get("records") != config["expected"]["android_records"]
        or android_manifest.get("schema") != ["schema_version", "sample_key", "candidate_success"]
        or android_manifest.get("training_started") is not False
    ):
        raise PermissionError("TriVUS private-label manifest mismatch")
    vus_keys = [
        key for key, row in public.items()
        if row["benchmark"] != "androidcontrol" and int(row["fold"]) in requested
    ]
    android_keys = [
        key for key, row in public.items()
        if row["benchmark"] == "androidcontrol" and int(row["fold"]) in requested
    ]
    vus, vus_opened = load_sealed_rows(
        vus_manifest, requested, VUS_DIR,
        config["expected"]["vus_label_rows_by_fold"], vus_keys,
    )
    android, android_opened = load_sealed_rows(
        android_manifest, requested, ROOT,
        config["expected"]["android_label_rows_by_fold"], android_keys,
    )
    labels = {**vus, **android}
    expected = set(vus_keys) | set(android_keys)
    if set(labels) != expected:
        raise ValueError("TriVUS private-label coverage mismatch")
    return labels, tuple(vus_opened + android_opened)


def assemble_phase_data(config, public, predictions, phase, requested_folds):
    context_rows, requested = select_phase_contexts(
        phase, public, requested_folds
    )
    labels, opened = _load_phase_private_labels(
        config, public, phase, requested
    )
    return assemble_data(context_rows, public, predictions, labels), opened


def assemble_data(context_rows, public, predictions, labels):
    rows = len(context_rows)
    context_keys_input = [row.get("context_key") for row in context_rows]
    sample_keys_input = [row.get("sample_key") for row in context_rows]
    if (
        rows < 1
        or len(set(context_keys_input)) != rows
        or len(set(sample_keys_input)) != rows
        or any(key not in public or key not in predictions for key in sample_keys_input)
    ):
        raise ValueError("TriVUS assembly context/sample identity mismatch")
    features = np.zeros((rows, MAX_CANDIDATES, INPUT_DIMENSION), dtype=np.float32)
    mask = np.zeros((rows, MAX_CANDIDATES), dtype=np.bool_)
    fallback = np.zeros(rows, dtype=np.int64)
    targets = np.zeros((rows, MAX_CANDIDATES + 1), dtype=np.float32)
    fallback_correct = np.zeros(rows, dtype=np.float32)
    active = np.zeros(rows, dtype=np.bool_)
    padded_labels = np.zeros((rows, MAX_CANDIDATES), dtype=np.bool_)
    context_keys = []
    sample_keys = []
    families = []
    cells = []
    row_ids = []
    folds = np.zeros(rows, dtype=np.int8)
    groups = []
    expected_keys = {row["sample_key"] for row in context_rows}
    if set(labels) != expected_keys:
        raise ValueError("TriVUS assembly label/context identity mismatch")
    for index, context in enumerate(context_rows):
        sample_key = context["sample_key"]
        row = public[sample_key]
        count = len(row["candidates"])
        fallback_index = int(context["fallback_index"])
        label = labels[sample_key]
        if (
            set(label) != {"schema_version", "sample_key", "candidate_success"}
            or label["schema_version"] != 1
            or label["sample_key"] != sample_key
        ):
            raise ValueError(f"TriVUS private-label row mismatch: {sample_key}")
        values = label["candidate_success"]
        if not isinstance(values, list) or len(values) != count or any(type(value) is not bool for value in values):
            raise ValueError(f"TriVUS private-label schema mismatch: {sample_key}")
        family = row["benchmark"]
        cell = row["setting"] if family == "androidcontrol" else row["arm"]
        features[index] = base_features(
            row["candidates"], predictions[sample_key], fallback_index, family, cell
        )
        mask[index, :count] = True
        fallback[index] = fallback_index
        padded_labels[index, :count] = values
        target, correct, is_active = target_values(values, fallback_index)
        targets[index] = target
        fallback_correct[index] = correct
        active[index] = is_active
        context_keys.append(context["context_key"])
        sample_keys.append(sample_key)
        families.append(family)
        cells.append(cell)
        row_ids.append(str(row["row_id"]))
        folds[index] = int(row["fold"])
        groups.append(str(row["group"]))
    data = TriVUSData(
        features=features,
        candidate_mask=mask,
        fallback_indices=fallback,
        target_distribution=targets,
        fallback_correct=fallback_correct,
        weights=np.zeros(rows, dtype=np.float64),
        active=active,
        labels=padded_labels,
        context_keys=tuple(context_keys),
        sample_keys=tuple(sample_keys),
        families=tuple(families),
        cells=tuple(cells),
        row_ids=tuple(row_ids),
        folds=folds,
        groups=tuple(groups),
    )
    validate_trivus_data(data)
    return data


def included_families_for_variant(variant, target_family=None):
    if variant == "TARGET_ONLY":
        if target_family not in FAMILIES:
            raise ValueError("TriVUS TARGET_ONLY family mismatch")
        return (target_family,)
    values = {
        "JOINT3": FAMILIES,
        "JOINT2_NO_ANDROID": FAMILIES[:2],
        "NO_VISUAL": FAMILIES,
        "RANDOM_ID_PLACEBO": FAMILIES,
    }
    if variant not in values or target_family is not None:
        raise ValueError("TriVUS variant family request mismatch")
    return tuple(values[variant])


def with_model_weights(data, variant, target_family=None):
    included = included_families_for_variant(variant, target_family)
    weights = assign_weights(data.families, data.cells, data.active, included)
    output = replace(data, weights=weights)
    validate_trivus_data(output, included)
    return output