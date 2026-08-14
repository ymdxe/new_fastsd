"""Canonicalize and validate requests before scheduler/model admission.

The HTTP models validate types, but they do not validate relationships between
``prefix_len``, ``gamma`` and the token payload.  The same validator is used at
the Cloud boundary and again at the worker ingress so malformed internal queue
messages cannot terminate the target worker.
"""

from __future__ import annotations

import math
from collections.abc import Mapping


MAX_REQUEST_PREFIX_LEN = 131_072
DEFAULT_MAX_TOKENS = 400


def _integer(value, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{field} must be a finite integer")
    return int(number)


def _token_count(draft_output) -> int:
    shape = getattr(draft_output, "shape", None)
    if shape is not None:
        if len(shape) != 2 or int(shape[0]) != 1:
            raise ValueError("draft_output must have shape [1, tokens]")
        return int(shape[1])
    try:
        return len(draft_output)
    except TypeError as exc:
        raise ValueError("draft_output must be a token sequence") from exc


def canonicalize_request(
    request: Mapping,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_prefix_len: int = MAX_REQUEST_PREFIX_LEN,
) -> dict:
    """Return a validated request copy with canonical integer fields.

    ``gamma`` is required and positive for Verify.  Prefill requests do not
    need gamma and are canonicalized to zero.  The token-length checks cover
    both external full-prefix and tail-only payloads, including a physical
    bridge token.
    """
    out = dict(request)
    task_type = str(out.get("task_type", ""))
    if task_type not in {"prefill", "verify"}:
        raise ValueError(f"unknown task_type: {task_type!r}")

    prefix_len = _integer(out.get("prefix_len"), "prefix_len")
    max_tokens = _integer(max_tokens, "max_tokens")
    max_prefix_len = _integer(max_prefix_len, "max_prefix_len")
    if max_tokens <= 0 or max_prefix_len <= 0:
        raise ValueError("request validation limits must be positive")
    if prefix_len <= 0 or prefix_len > max_prefix_len:
        raise ValueError(f"prefix_len must be in [1, {max_prefix_len}]")

    token_count = _token_count(out.get("draft_output"))
    if token_count < 0 or token_count > max_prefix_len + max_tokens + 1:
        raise ValueError("draft_output is too large")

    if task_type == "prefill":
        if token_count < prefix_len:
            raise ValueError("prefill prefix_len exceeds draft_output length")
        gamma = 0
    else:
        gamma = _integer(out.get("gamma"), "gamma")
        if gamma < 1 or gamma > max_tokens:
            raise ValueError(f"gamma must be in [1, {max_tokens}]")
        bridge = 1 if bool(out.get("has_bridge_token", False)) else 0
        if bool(out.get("tail_only", False)):
            required_tokens = gamma + bridge
        else:
            required_tokens = prefix_len + gamma
        if token_count < required_tokens:
            raise ValueError(
                f"draft_output has {token_count} tokens, requires at least {required_tokens}"
            )

    out["task_type"] = task_type
    out["prefix_len"] = prefix_len
    out["gamma"] = gamma
    return out
