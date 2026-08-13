"""Common request-level metric schema for FastSD comparison methods."""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _numeric(records: list[dict[str, Any]], key: str) -> list[float]:
    return [float(record[key]) for record in records if record.get(key) is not None]


def _metric_stats(records: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = _numeric(records, key)
    return {
        "avg": statistics.mean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def _last_number(value: Any) -> str | None:
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return matches[-1] if matches else None


def exact_match_quality(records: list[dict[str, Any]], dataset: str) -> float | None:
    if dataset not in {"gsm8k", "mgsm"}:
        return None
    scored = 0
    correct = 0
    for record in records:
        if record.get("output_text") is None or record.get("reference") is None:
            continue
        predicted = _last_number(record["output_text"])
        expected = _last_number(record["reference"])
        if predicted is None or expected is None:
            continue
        scored += 1
        correct += int(predicted == expected)
    return correct / scored if scored else None


def summarize_requests(
    records: Iterable[dict[str, Any]],
    *,
    method: str,
    dataset: str,
    workload_hash: str,
    run_id: str,
    wallclock_s: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_records = list(records)
    for record in request_records:
        if record.get("ttft_ms") is not None and record.get("arrival_lag_ms") is not None:
            record.setdefault(
                "scheduled_ttft_ms",
                float(record["ttft_ms"]) + float(record["arrival_lag_ms"]),
            )
    generated = sum(int(record.get("generated_tokens", 0)) for record in request_records)
    if wallclock_s is None:
        completions = _numeric(request_records, "completion_s")
        arrivals = _numeric(request_records, "actual_arrival_s")
        if completions and arrivals:
            wallclock_s = max(completions) - min(arrivals)
        else:
            wallclock_s = sum(_numeric(request_records, "e2e_ms")) / 1000.0

    accepted_per_verify = _numeric(request_records, "mean_accepted_tokens_per_verify")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "method": method,
        "dataset": dataset,
        "workload_hash": workload_hash,
        "num_requests": len(request_records),
        "total_generated_tokens": generated,
        "wallclock_s": float(wallclock_s or 0.0),
        "throughput_tok_s": generated / wallclock_s if wallclock_s and wallclock_s > 0 else None,
        "ttft_ms": _metric_stats(request_records, "ttft_ms"),
        "scheduled_ttft_ms": _metric_stats(request_records, "scheduled_ttft_ms"),
        "tpot_ms": _metric_stats(request_records, "tpot_ms"),
        "e2e_ms": _metric_stats(request_records, "e2e_ms"),
        "accept_rate": (
            sum(int(record.get("accepted_tokens", 0)) for record in request_records)
            / sum(int(record.get("drafted_tokens", 0)) for record in request_records)
            if sum(int(record.get("drafted_tokens", 0)) for record in request_records) > 0
            else None
        ),
        "mean_accepted_tokens_per_verify": (
            statistics.mean(accepted_per_verify) if accepted_per_verify else None
        ),
        "quality_exact_match": exact_match_quality(request_records, dataset),
        "metric_definitions": {
            "ttft_ms": "actual worker start to first output token visible",
            "scheduled_ttft_ms": "scheduled workload release to first output token, including arrival lag",
            "tpot_ms": "post-first-output generation time divided by remaining output tokens",
            "throughput_tok_s": "all generated output tokens divided by measured run wallclock",
        },
    }
    if extra:
        summary.update(extra)
    return summary


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output
