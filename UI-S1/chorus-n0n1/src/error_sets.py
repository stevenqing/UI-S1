from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


VERSION_FIELDS = ("error_set_version", "run_scope", "sample_name")


def tag_record(
    record: Dict[str, Any],
    *,
    error_set_version: str,
    run_scope: str,
    sample_name: str,
) -> Dict[str, Any]:
    tagged = dict(record)
    tagged["error_set_version"] = error_set_version
    tagged["run_scope"] = run_scope
    tagged["sample_name"] = sample_name
    return tagged


def tag_records(
    records: Iterable[Dict[str, Any]],
    *,
    error_set_version: str,
    run_scope: str,
    sample_name: str,
) -> List[Dict[str, Any]]:
    return [
        tag_record(
            record,
            error_set_version=error_set_version,
            run_scope=run_scope,
            sample_name=sample_name,
        )
        for record in records
    ]


def assert_error_set_version(
    records: Iterable[Dict[str, Any]],
    *,
    expected_error_set_version: str,
    expected_run_scope: Optional[str] = None,
    expected_sample_name: Optional[str] = None,
) -> None:
    for row_number, record in enumerate(records, start=1):
        observed_version = record.get("error_set_version")
        if observed_version != expected_error_set_version:
            raise ValueError(
                f"error_set_version mismatch at row {row_number}: "
                f"expected {expected_error_set_version!r}, got {observed_version!r}"
            )
        if expected_run_scope is not None and record.get("run_scope") != expected_run_scope:
            raise ValueError(
                f"run_scope mismatch at row {row_number}: "
                f"expected {expected_run_scope!r}, got {record.get('run_scope')!r}"
            )
        if expected_sample_name is not None and record.get("sample_name") != expected_sample_name:
            raise ValueError(
                f"sample_name mismatch at row {row_number}: "
                f"expected {expected_sample_name!r}, got {record.get('sample_name')!r}"
            )