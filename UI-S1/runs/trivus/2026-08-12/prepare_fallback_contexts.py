import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
VUS_DIR = ROOT / "runs/visual-utility-selector/2026-08-11"
sys.path.insert(0, str(RUN_DIR))

from context_common import (
    CONFIG_PATH, apply_android_policy, apply_vus_policies, atomic_json_file,
    android_reliability, build_vus_banks, checkpoint_and_fit_folds, context_record,
    committed_file, git_blob_sha256,
    fit_final_vus_policies, fit_inner_vus_policies, inject_vus_labels,
    fsync_directory, load_jsonl, load_prereg, load_sealed_rows,
    publish_directory, require_commit_order, safe_child_path, staging_directory,
    write_exclusive_json,
)
from recovery_common import assert_protected_process, load_config as load_recovery_config, sha256_file


def validate_authorization(config):
    path = ROOT / config["output"]["authorization"]
    if not path.is_file():
        raise PermissionError("TriVUS fallback-context generation is not implementation-authorized")
    value = json.loads(path.read_text())
    if (
        value.get("status") != "AUTHORIZED_TRIVUS_FALLBACK_CONTEXT_GENERATION_ONCE"
        or value.get("result_must_not_exist") is not True
    ):
        raise PermissionError("TriVUS invalid fallback-context authorization")
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", value["implementation_commit"], "HEAD"],
        cwd=ROOT, check=False,
    )
    if completed.returncode:
        raise PermissionError("TriVUS fallback-context implementation commit is not an ancestor")
    authorization_commit = committed_file(path)
    require_commit_order(
        value["implementation_commit"], authorization_commit,
        "fallback-context implementation must precede authorization",
    )
    scale_manifest = ROOT / config["output"]["private_scale_manifest"]
    scale_commit = committed_file(scale_manifest)
    require_commit_order(
        scale_commit, authorization_commit,
        "private-scale seal must precede fallback-context authorization",
    )
    implementation_files = {
        "configs/fallback_contexts_prereg.yaml", "context_common.py",
        "prepare_private_scales.py", "prepare_fallback_contexts.py",
        "test_fallback_contexts.py",
    }
    scale_file = "data/private_scales/PRIVATE_SCALE_MANIFEST.json"
    expected = implementation_files | {scale_file}
    if set(value.get("files", {})) != expected:
        raise PermissionError("TriVUS fallback-context authorization file set mismatch")
    for name, expected_hash in value["files"].items():
        if sha256_file(RUN_DIR / name) != expected_hash:
            raise PermissionError(f"TriVUS fallback-context implementation hash mismatch: {name}")
        commit = value["implementation_commit"] if name in implementation_files else scale_commit
        if git_blob_sha256(commit, RUN_DIR / name) != expected_hash:
            raise PermissionError(f"TriVUS fallback-context committed blob mismatch: {name}")
    output_directory = (ROOT / config["output"]["path"]).parent
    receipt = authorization_receipt_path(config, value)
    if output_directory.exists() or receipt.exists():
        raise FileExistsError("TriVUS fallback-context result already exists")
    nonce = value.get("authorization_nonce")
    if not isinstance(nonce, str) or len(nonce) != 64:
        raise PermissionError("TriVUS fallback-context authorization nonce mismatch")
    return value, authorization_commit, scale_commit


def authorization_receipt_path(config, authorization):
    nonce = authorization.get("authorization_nonce")
    if not isinstance(nonce, str) or len(nonce) != 64 or any(
        character not in "0123456789abcdef" for character in nonce
    ):
        raise PermissionError("TriVUS fallback-context authorization nonce mismatch")
    return ROOT / config["output"]["authorization_receipts"] / f"{nonce}.json"


def write_authorization_receipt(path, value, schema):
    write_exclusive_json(path, value, schema)


def consume_authorization(config, authorization):
    path = authorization_receipt_path(config, authorization)
    value = {
        "schema_version": 1,
        "status": "CONSUMED_TRIVUS_FALLBACK_CONTEXT_AUTHORIZATION",
        "authorization_sha256": sha256_file(ROOT / config["output"]["authorization"]),
        "authorization_nonce": authorization["authorization_nonce"],
        "implementation_commit": authorization["implementation_commit"],
        "prereg_sha256": sha256_file(CONFIG_PATH),
    }
    write_authorization_receipt(path, value, config["authorization_receipt_schema"])
    return value


def load_private_scale_manifest(config):
    path = ROOT / config["output"]["private_scale_manifest"]
    manifest = json.loads(path.read_text())
    if (
        manifest.get("status") != "PASS_TRIVUS_PHYSICALLY_SEALED_PRIVATE_SCALES"
        or manifest.get("records") != 2080
        or manifest.get("schema") != config["private_scale_schema"]
        or manifest.get("prereg_sha256") != sha256_file(CONFIG_PATH)
        or manifest.get("source_public_sha256") != config["dependencies"]["vus_public"]["sha256"]
        or manifest.get("source_targets_sha256") != config["dependencies"]["mind_targets"]["sha256"]
        or manifest.get("aggregate_target_statistics_computed") is not False
        or manifest.get("candidate_success_opened") is not False
        or manifest.get("training_started") is not False
        or manifest.get("environment") != config["environment"]
    ):
        raise PermissionError("TriVUS invalid private-scale seal")
    receipt_relative = manifest.get("authorization_receipt")
    if not isinstance(receipt_relative, str):
        raise PermissionError("TriVUS private-scale authorization receipt path mismatch")
    receipt_root = ROOT / config["output"]["private_scale_authorization_receipts"]
    receipt = safe_child_path(receipt_root, Path(receipt_relative).name)
    if (
        receipt.relative_to(ROOT).as_posix() != receipt_relative
        or not receipt.is_file()
        or sha256_file(receipt) != manifest.get("authorization_receipt_sha256")
    ):
        raise PermissionError("TriVUS private-scale authorization receipt mismatch")
    receipt_value = json.loads(receipt.read_text())
    if (
        set(receipt_value) != set(config["authorization_receipt_schema"])
        or receipt_value.get("status") != "CONSUMED_TRIVUS_PRIVATE_SCALE_AUTHORIZATION"
        or receipt_value.get("authorization_nonce") != manifest.get("authorization_nonce")
        or receipt_value.get("implementation_commit") != manifest.get("implementation_commit")
        or receipt_value.get("prereg_sha256") != manifest.get("prereg_sha256")
        or receipt_value.get("authorization_sha256") != manifest.get("authorization_sha256")
    ):
        raise PermissionError("TriVUS invalid private-scale authorization receipt")
    return manifest


def keyed(rows, field="sample_key"):
    output = {row[field]: row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"TriVUS duplicate public key: {field}")
    return output


def load_final_anchors(config):
    rows = load_jsonl(ROOT / config["dependencies"]["vus_final_anchors"]["path"])
    anchors = {
        (int(row["outer_fold"]), row["sample_key"]): int(row["fallback_index"])
        for row in rows if row["role"] == "test"
    }
    if len(anchors) != config["expected"]["final_anchor_records"]:
        raise ValueError("TriVUS final fallback-anchor coverage mismatch")
    return anchors


def load_vus_labels(config, manifest, public, folds):
    keys = [key for key, row in public.items() if int(row["fold"]) in set(folds)]
    rows, opened = load_sealed_rows(
        manifest, folds, VUS_DIR,
        config["expected"]["vus_label_rows_by_fold"], keys,
    )
    return rows, opened


def load_android_labels(config, manifest, public, folds):
    keys = [key for key, row in public.items() if int(row["fold"]) in set(folds)]
    rows, opened = load_sealed_rows(
        manifest, folds, ROOT,
        config["expected"]["android_label_rows_by_fold"], keys,
    )
    return rows, opened


def load_scales(config, manifest, mind_rows, folds):
    row_ids = [row_id for row_id, row in mind_rows.items() if row.fold in set(folds)]
    rows, opened = load_sealed_rows(
        manifest, folds, ROOT,
        config["expected"]["mind_scale_rows_by_fold"], row_ids,
    )
    scales = {
        row_id: (row["normalized_width"], row["normalized_height"])
        for row_id, row in rows.items()
    }
    return scales, opened


class AtomicContextWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        self.handle = None
        self.count = 0
        self.previous_key = None
        self.coverage = {}

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary.unlink(missing_ok=True)
        self.handle = self.temporary.open("w", buffering=1)
        return self

    def write(self, record):
        key = record["context_key"]
        if self.previous_key is not None and key <= self.previous_key:
            raise ValueError(f"TriVUS context order/identity mismatch: {key}")
        self.handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        self.previous_key = key
        self.count += 1
        counts = self.coverage.setdefault(record["sample_key"], Counter())
        counts[record["role"]] += 1

    def __exit__(self, exception_type, exception, traceback):
        try:
            if exception_type is None:
                self.handle.flush()
                os.fsync(self.handle.fileno())
            self.handle.close()
            if exception_type is None:
                self.temporary.replace(self.path)
                fsync_directory(self.path.parent)
            else:
                self.temporary.unlink(missing_ok=True)
        except BaseException:
            self.temporary.unlink(missing_ok=True)
            raise


def write_phase(writer, outer_fold, role, holdout_fold, fit_folds, fallbacks):
    for sample_key in sorted(fallbacks):
        writer.write(context_record(
            outer_fold, role, holdout_fold, fit_folds,
            sample_key, fallbacks[sample_key],
        ))


def split_input_hashes(vus_manifest, android_manifest, scale_manifest, fit_folds):
    return {
        "vus_private": {
            str(fold): vus_manifest["folds"][str(fold)]["sha256"]
            for fold in fit_folds
        },
        "android_private": {
            str(fold): android_manifest["folds"][str(fold)]["sha256"]
            for fold in fit_folds
        },
        "mind_private_scale": {
            str(fold): scale_manifest["folds"][str(fold)]["sha256"]
            for fold in fit_folds
        },
    }


def validate_context_coverage(coverage, expected_keys):
    if set(coverage) != set(expected_keys) or any(
        counts != {"final": 5, "inner": 16}
        for counts in coverage.values()
    ):
        raise ValueError("TriVUS per-sample context coverage mismatch")
    return True


def generate(config, output_path):
    vus_rows = load_jsonl(ROOT / config["dependencies"]["vus_public"]["path"])
    android_rows = load_jsonl(ROOT / config["dependencies"]["android_public"]["path"])
    regions = load_jsonl(ROOT / config["dependencies"]["screen_regions"]["path"])
    base_banks, vus_public = build_vus_banks(vus_rows, regions)
    android_public = keyed(android_rows)
    if len(android_public) != config["expected"]["android_records"]:
        raise ValueError("TriVUS Android public coverage mismatch")
    vus_manifest = json.loads((ROOT / config["dependencies"]["vus_private_manifest"]["path"]).read_text())
    android_manifest = json.loads((ROOT / config["dependencies"]["android_private_manifest"]["path"]).read_text())
    scale_manifest = load_private_scale_manifest(config)
    anchors = load_final_anchors(config)
    mind_rows = base_banks["C_uni"]["mind2web"]
    split_reports = []
    opened_files = set()
    anchor_matches = 0
    with AtomicContextWriter(output_path) as writer:
        for outer_fold in range(5):
            development = tuple(fold for fold in range(5) if fold != outer_fold)
            vus_labels, opened = load_vus_labels(config, vus_manifest, vus_public, development)
            opened_files.update(opened)
            android_labels, opened = load_android_labels(
                config, android_manifest, android_public, development,
            )
            opened_files.update(opened)
            scales, opened = load_scales(config, scale_manifest, mind_rows, development)
            opened_files.update(opened)
            final_banks = inject_vus_labels(base_banks, vus_public, vus_labels, development)
            final_policies = fit_final_vus_policies(final_banks, outer_fold, scales)
            final_vus = apply_vus_policies(final_banks, vus_public, final_policies, range(5))
            reliability = android_reliability(
                android_public, android_labels, development, config["seed"],
            )
            final_android = apply_android_policy(
                android_public, reliability, range(5), config["seed"],
            )
            for sample_key, fallback in final_vus.items():
                if int(vus_public[sample_key]["fold"]) == outer_fold:
                    if anchors[(outer_fold, sample_key)] != fallback:
                        raise ValueError(f"TriVUS final CEV index-anchor mismatch: {outer_fold}/{sample_key}")
                    anchor_matches += 1
            final = {**final_vus, **final_android}
            write_phase(writer, outer_fold, "final", None, development, final)
            split_reports.append({
                "outer_fold": outer_fold,
                "role": "final",
                "holdout_fold": None,
                "checkpoint_fold": None,
                "fit_folds": list(development),
                "applied_folds": list(range(5)),
                "contexts": len(final),
                "opened_fold_hashes": split_input_hashes(
                    vus_manifest, android_manifest, scale_manifest, development,
                ),
            })
            del android_labels, final_banks, final_policies, reliability, scales, vus_labels
            for holdout_fold in development:
                checkpoint, fit_folds = checkpoint_and_fit_folds(outer_fold, holdout_fold)
                vus_labels, opened = load_vus_labels(config, vus_manifest, vus_public, fit_folds)
                opened_files.update(opened)
                android_labels, opened = load_android_labels(
                    config, android_manifest, android_public, fit_folds,
                )
                opened_files.update(opened)
                scales, opened = load_scales(config, scale_manifest, mind_rows, fit_folds)
                opened_files.update(opened)
                inner_banks = inject_vus_labels(base_banks, vus_public, vus_labels, fit_folds)
                policies = fit_inner_vus_policies(inner_banks, fit_folds, checkpoint, scales)
                inner_vus = apply_vus_policies(inner_banks, vus_public, policies, development)
                reliability = android_reliability(
                    android_public, android_labels, fit_folds, config["seed"],
                )
                inner_android = apply_android_policy(
                    android_public, reliability, development, config["seed"],
                )
                values = {**inner_vus, **inner_android}
                write_phase(writer, outer_fold, "inner", holdout_fold, fit_folds, values)
                split_reports.append({
                    "outer_fold": outer_fold,
                    "role": "inner",
                    "holdout_fold": holdout_fold,
                    "checkpoint_fold": checkpoint,
                    "fit_folds": list(fit_folds),
                    "applied_folds": list(development),
                    "contexts": len(values),
                    "opened_fold_hashes": split_input_hashes(
                        vus_manifest, android_manifest, scale_manifest, fit_folds,
                    ),
                })
                del android_labels, inner_banks, policies, reliability, scales, vus_labels
        validate_context_coverage(
            writer.coverage, set(vus_public) | set(android_public),
        )
    return {
        "contexts": writer.count,
        "public_records": len(writer.coverage),
        "per_sample_final_contexts": 5,
        "per_sample_inner_contexts": 16,
        "anchor_matches": anchor_matches,
        "splits": split_reports,
        "opened_private_files": sorted(opened_files),
    }


def main():
    config = load_prereg()
    authorization, authorization_commit, scale_commit = validate_authorization(config)
    recovery = load_recovery_config()
    assert_protected_process(recovery)
    receipt = consume_authorization(config, authorization)
    receipt_path = authorization_receipt_path(config, authorization)
    output_path = ROOT / config["output"]["path"]
    manifest_path = ROOT / config["output"]["manifest"]
    output_directory = output_path.parent
    with staging_directory(output_directory) as staging:
        output = staging / output_path.name
        report = generate(config, output)
        if report["contexts"] != config["expected"]["contexts"]:
            raise ValueError(f"TriVUS fallback-context count mismatch: {report['contexts']}")
        if report["public_records"] != config["expected"]["total_records"]:
            raise ValueError(f"TriVUS fallback-context public coverage mismatch: {report['public_records']}")
        if report["anchor_matches"] != config["expected"]["final_anchor_records"]:
            raise ValueError(f"TriVUS final-anchor count mismatch: {report['anchor_matches']}")
        assert_protected_process(recovery)
        manifest = {
            "schema_version": 1,
            "status": "PASS_TRIVUS_EXACT_FALLBACK_CONTEXTS",
            "records": report["contexts"],
            "public_records": report["public_records"],
            "contexts_per_public_record": config["expected"]["contexts_per_record"],
            "per_sample_final_contexts": report["per_sample_final_contexts"],
            "per_sample_inner_contexts": report["per_sample_inner_contexts"],
            "record_schema": config["context_schema"],
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "final_index_anchor_records": report["anchor_matches"],
            "final_index_anchor_mismatches": 0,
            "splits": report["splits"],
            "opened_private_files": report["opened_private_files"],
            "authorization_sha256": receipt["authorization_sha256"],
            "authorization_receipt": str(receipt_path.relative_to(ROOT)),
            "authorization_receipt_sha256": sha256_file(receipt_path),
            "authorization_nonce": receipt["authorization_nonce"],
            "implementation_commit": authorization["implementation_commit"],
            "authorization_commit": authorization_commit,
            "private_scale_commit": scale_commit,
            "prereg_sha256": sha256_file(CONFIG_PATH),
            "environment": config["environment"],
            "candidate_success_emitted": False,
            "source_identity_emitted": False,
            "reliability_emitted": False,
            "configuration_score_emitted": False,
            "aggregate_performance_computed": False,
            "training_started": False,
        }
        atomic_json_file(staging / manifest_path.name, manifest)
        assert_protected_process(recovery)
        publish_directory(staging, output_directory)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()