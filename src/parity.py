"""Auditable token-level parity reports for the unified workload."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .run_artifacts import require_unique_sample_ids, write_json_once, write_text_once


FOUR_METHODS = ("fastsd", "specedge_cpu_adapted", "standard_sd", "draft_only")
ORACLE_METHOD = "target_only"


def _token_ids(record: dict[str, Any]) -> list[int]:
    values = record.get("output_token_ids", record.get("token_ids", []))
    return [int(value) for value in (values or [])]


def _records_by_sample(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    items = list(records)
    result = {str(record["sample_id"]): record for record in items}
    if len(result) != len(items):
        raise ValueError("duplicate sample_id in parity records")
    require_unique_sample_ids(result.values())
    return result


def build_token_parity_rows(
    records_by_method: dict[str, Iterable[dict[str, Any]]],
    *,
    reference_method: str = ORACLE_METHOD,
) -> list[dict[str, Any]]:
    """Build one JSON object per sample and token position.

    A missing position is represented by ``null``.  This makes early EOS or a
    stop-policy mismatch visible instead of silently comparing only the common
    prefix.
    """

    if reference_method not in records_by_method:
        raise ValueError(f"reference method is missing: {reference_method}")
    indexed = {
        method: _records_by_sample(list(records))
        for method, records in records_by_method.items()
    }
    reference_ids = list(indexed[reference_method])
    rows: list[dict[str, Any]] = []
    for sample_id in reference_ids:
        token_lists = {
            method: _token_ids(indexed[method].get(sample_id, {}))
            for method in indexed
        }
        max_len = max((len(values) for values in token_lists.values()), default=0)
        workload_hashes = {
            str(indexed[method][sample_id].get("workload_hash"))
            for method in indexed
            if sample_id in indexed[method]
            and indexed[method][sample_id].get("workload_hash") is not None
        }
        workload_hash = next(iter(workload_hashes)) if len(workload_hashes) == 1 else None
        for position in range(max_len):
            token_ids = {
                method: values[position] if position < len(values) else None
                for method, values in token_lists.items()
            }
            reference_token = token_ids[reference_method]
            rows.append(
                {
                    "sample_id": sample_id,
                    "global_index": indexed[reference_method][sample_id].get("global_index"),
                    "position": position,
                    "reference_method": reference_method,
                    "reference_token_id": reference_token,
                    "token_ids": token_ids,
                    "equal_to_reference": {
                        method: token == reference_token and token is not None
                        for method, token in token_ids.items()
                        if method != reference_method
                    },
                    "workload_hash": workload_hash,
                }
            )
    return rows


def summarize_token_parity(
    rows: Iterable[dict[str, Any]],
    *,
    methods: Iterable[str] | None = None,
    reference_method: str = ORACLE_METHOD,
) -> dict[str, Any]:
    rows_list = list(rows)
    method_names = list(methods or (next(iter(rows_list), {}).get("token_ids", {}).keys()))
    result: dict[str, Any] = {
        "reference_method": reference_method,
        "total_token_positions": len(rows_list),
        "methods": {},
    }
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_list:
        by_sample[str(row["sample_id"])].append(row)
    for method in method_names:
        if method == reference_method:
            continue
        matches = sum(bool(row.get("equal_to_reference", {}).get(method)) for row in rows_list)
        comparable = sum(
            row.get("token_ids", {}).get(method) is not None
            and row.get("reference_token_id") is not None
            for row in rows_list
        )
        exact_sequences = 0
        first_mismatches: list[int] = []
        for sample_rows in by_sample.values():
            mismatch = next(
                (
                    int(row["position"])
                    for row in sample_rows
                    if not row.get("equal_to_reference", {}).get(method, False)
                ),
                None,
            )
            if mismatch is None:
                exact_sequences += 1
            else:
                first_mismatches.append(mismatch)
        result["methods"][method] = {
            "matching_positions": matches,
            "comparable_positions": comparable,
            "token_parity_rate": matches / comparable if comparable else None,
            "exact_sequence_count": exact_sequences,
            "sample_count": len(by_sample),
            "first_mismatch_position_avg": (
                sum(first_mismatches) / len(first_mismatches) if first_mismatches else None
            ),
        }
    return result


def write_token_parity_report(
    rows: Iterable[dict[str, Any]],
    path: str | Path,
    *,
    summary_path: str | Path | None = None,
    methods: Iterable[str] | None = None,
    reference_method: str = ORACLE_METHOD,
) -> tuple[Path, Path | None]:
    rows_list = list(rows)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows_list)
    output = write_text_once(path, content)
    summary_output = None
    if summary_path is not None:
        summary_output = write_json_once(
            summary_path,
            summarize_token_parity(rows_list, methods=methods, reference_method=reference_method),
        )
    return output, summary_output
