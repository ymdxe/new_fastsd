"""Pure FastSD admission policy.

The cloud executor owns model/KV operations; this module only decides which
logical work items may advance in one scheduling iteration.  Keeping the
planner side-effect free is deliberate: the same code is used by the real
admission path and by the KV prefetch predictor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


FASTSD_DEFAULT_R1 = 128
FASTSD_DEFAULT_R2 = 512
FASTSD_DYNAMIC_WINDOW = 100
FASTSD_VERIFY_PREFILL_SWITCH_THRESHOLD = 6  # legacy API compatibility
FASTSD_VERIFY_WRR_ORDER = (
    "short", "short", "short", "short", "short", "short",
    "mid", "mid", "mid", "long",
)
FASTSD_QUEUE_ORDER = ("SV", "SP", "MV", "MP", "LV", "LP")


def build_fixed_wrr_order():
    return list(FASTSD_VERIFY_WRR_ORDER)


def compute_quantile(sorted_values, q: float) -> int:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    idx = max(0, min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1))
    return int(sorted_values[idx])


def update_length_thresholds(
    recent_prefix_lens,
    default_r1: int = FASTSD_DEFAULT_R1,
    default_r2: int = FASTSD_DEFAULT_R2,
    min_samples: int = FASTSD_DYNAMIC_WINDOW,
    q1: float = 0.7,
    q2: float = 0.9,
):
    if len(recent_prefix_lens) < min_samples:
        return default_r1, default_r2
    sorted_lens = sorted(int(v) for v in recent_prefix_lens)
    r1 = compute_quantile(sorted_lens, q1)
    r2 = compute_quantile(sorted_lens, q2)
    if r2 < r1:
        r2 = r1
    return r1, r2


def length_category(prefix_len: int, r1: int, r2: int) -> str:
    if prefix_len <= r1:
        return "short"
    if prefix_len <= r2:
        return "mid"
    return "long"


def compute_priority_score(req, accept_stats, now=None, lamda=0.01):
    if now is None:
        now = 0.0
    elapsed = max(0.0, float(now) - float(req.get("current_time", 0.0)))
    wait_term = math.exp(lamda * elapsed)
    if req["task_type"] == "prefill":
        return wait_term

    pid = req["proc_id"]
    accepted_sum, total_sum = accept_stats.get(pid, (0, 0))
    if total_sum <= 0:
        acc_prob = 1.0
    else:
        acc_prob = (accepted_sum + 1.0) / (total_sum + 1.0)
    acc_prob = max(acc_prob, 1e-6)
    draft_len = max(1, int(req.get("gamma", 1) or 1))
    draft_total_time = float(req.get("lag", 0.0))
    if draft_total_time <= 0.0 and "draft_time_per_token" in req:
        draft_total_time = draft_len * max(0.0, float(req["draft_time_per_token"]))
    transport_rtt = max(0.0, float(req.get("transport_rtt", req.get("edge_rtt", 0.0))))
    return -((draft_total_time + transport_rtt) / acc_prob) + wait_term


def should_switch_to_prefill(verify_underutilized_flags, has_prefill_tasks: bool) -> bool:
    """Legacy helper retained for callers outside the new FastSD planner."""
    return bool(has_prefill_tasks) and sum(int(flag) for flag in verify_underutilized_flags) >= FASTSD_VERIFY_PREFILL_SWITCH_THRESHOLD


@dataclass
class WorkItem:
    """Persistent logical request; it is never split into independent HTTP requests."""

    work_id: str
    proc_id: Any
    task_type: str
    category: str
    request: Mapping[str, Any] = field(default_factory=dict, repr=False)
    total_tokens: int = 0
    cursor: int = 0
    enqueue_cycle: int = 0
    last_served_cycle: int = -1
    missed_cycles: int = 0
    base_prefix_len: int = 0
    original_gamma: int = 0
    accepted_so_far: int = 0
    bridge_pending: int = 0
    response_key: str | None = None
    finished: bool = False
    state: str = "ready"  # ready -> reserved -> running -> ready/finished

    @property
    def remaining_tokens(self) -> int:
        return max(0, int(self.total_tokens) - int(self.cursor))

    @property
    def is_overdue_prefill(self) -> bool:
        return self.task_type == "prefill" and self.missed_cycles >= 2

    @classmethod
    def from_request(cls, request: Mapping[str, Any], category: str, cycle: int = 0, work_id: str | None = None):
        task_type = str(request["task_type"])
        if task_type == "prefill":
            prompt_len = request.get("prompt_len")
            if prompt_len is None:
                prompt_len = request["draft_output"].shape[1]
            total = int(prompt_len)
            gamma = 0
        else:
            total = int(request.get("gamma", 0) or 0)
            gamma = total
        if total <= 0:
            raise ValueError("WorkItem total_tokens must be positive")
        pid = request["proc_id"]
        return cls(
            work_id=work_id or str(request.get("work_id", f"{pid}:{id(request)}")),
            proc_id=pid,
            task_type=task_type,
            category=category,
            request=request,
            total_tokens=total,
            enqueue_cycle=cycle,
            base_prefix_len=int(request.get("prefix_len", 0) or 0),
            original_gamma=gamma,
            bridge_pending=(
                1 if request.get("has_bridge_token", False)
                else int(request.get("bridge_tokens", 0) or 0)
            ),
            response_key=request.get("response_key"),
        )


@dataclass(frozen=True)
class ExecutionSlice:
    work_id: str
    proc_id: Any
    task_type: str
    category: str
    offset: int
    draft_token_count: int
    forward_token_count: int
    includes_bridge: bool = False

    @property
    def num_tokens(self) -> int:
        """Compatibility alias: admission cost is real forward tokens."""
        return self.forward_token_count


@dataclass
class SchedulerState:
    wrr_cursor: int = 0
    current_cycle: int = 0


@dataclass
class AdmissionPlan:
    verify_slices: list[ExecutionSlice]
    prefill_slices: list[ExecutionSlice]
    used_tokens: int
    token_budget: int
    max_num_seqs: int
    next_wrr_cursor: int
    next_cycle: int
    slots_consumed: int
    selected_work_ids: list[str] = field(default_factory=list)
    _selected_items: list[WorkItem] = field(default_factory=list, repr=False)

    @property
    def verify_proc_ids(self):
        return [s.proc_id for s in self.verify_slices]

    @property
    def prefill_proc_ids(self):
        return [s.proc_id for s in self.prefill_slices]

    @property
    def remaining_tokens(self) -> int:
        return self.token_budget - self.used_tokens


def _normalise_queues(queues: Mapping[str, Any]) -> dict[str, dict[str, list[WorkItem]]]:
    """Accept either {task_type: {category: items}} or {category: {task_type: items}}."""
    out = {"verify": {c: [] for c in ("short", "mid", "long")}, "prefill": {c: [] for c in ("short", "mid", "long")}}
    if set(queues).issubset({"verify", "prefill"}):
        for task_type, by_cat in queues.items():
            for cat, items in by_cat.items():
                out[task_type][cat] = list(items)
    else:
        for cat, by_type in queues.items():
            for task_type, items in by_type.items():
                out[task_type][cat] = list(items)
    return out


def _as_work_item(item: WorkItem | Mapping[str, Any], task_type: str, category: str, cycle: int) -> WorkItem:
    if isinstance(item, WorkItem):
        return item
    return WorkItem.from_request(item, category=category, cycle=cycle)


def _priority(item: WorkItem, accept_stats: Mapping[Any, Sequence[float]], now: float) -> float:
    return compute_priority_score(item.request, accept_stats, now=now)


def _candidate_for_category(
    state: dict[str, dict[str, list[WorkItem]]],
    category: str,
    selected: set[str],
    accept_stats: Mapping[Any, Sequence[float]],
    now: float,
    remaining_budget: int,
    min_prefill_chunk_tokens: int,
    reserved: set[str],
) -> WorkItem | None:
    candidates: list[WorkItem] = []
    overdue = []
    for task_type in ("verify", "prefill"):
        for item in state[task_type][category]:
            if item.work_id in selected or item.work_id in reserved or item.finished or item.remaining_tokens <= 0:
                continue
            if task_type == "verify":
                # A bridge token is an external pipeline token carried only
                # by the first slice of a logical verify round.  The executor
                # clears ``bridge_pending`` after an all-accepted partial
                # slice, before the next plan is built.
                # Verify is itself chunkable; only the minimum viable slice
                # (one draft token plus any bridge) must fit the remainder.
                if item.remaining_tokens >= 1 and item.bridge_pending + 1 <= remaining_budget:
                    candidates.append(item)
            else:
                if remaining_budget >= min_prefill_chunk_tokens or item.remaining_tokens < min_prefill_chunk_tokens:
                    candidates.append(item)
            if item.is_overdue_prefill and item in candidates:
                overdue.append(item)
    if overdue:
        return max(overdue, key=lambda x: _priority(x, accept_stats, now))
    verifies = [x for x in candidates if x.task_type == "verify"]
    if verifies:
        return max(verifies, key=lambda x: _priority(x, accept_stats, now))
    prefills = [x for x in candidates if x.task_type == "prefill"]
    if prefills:
        return max(prefills, key=lambda x: _priority(x, accept_stats, now))
    return None


def _all_candidates(state, selected, reserved):
    for task_type in ("verify", "prefill"):
        for category in ("short", "mid", "long"):
            for item in state[task_type][category]:
                if item.work_id not in selected and item.work_id not in reserved and not item.finished and item.remaining_tokens > 0:
                    yield item


def _make_slice(item: WorkItem, budget: int, quantum: int, min_prefill: int) -> ExecutionSlice | None:
    if budget <= 0 or item.remaining_tokens <= 0:
        return None
    # bridge_pending is explicit WorkItem state so admission charges the
    # external pipeline bridge exactly once.
    bridge = item.bridge_pending if item.task_type == "verify" else 0
    if item.task_type == "verify":
        draft = min(item.remaining_tokens, budget - bridge)
        if draft <= 0:
            return None
    else:
        if budget < min_prefill and item.remaining_tokens >= min_prefill:
            return None
        draft = min(item.remaining_tokens, budget, quantum)
        if draft <= 0:
            return None
    return ExecutionSlice(
        work_id=item.work_id,
        proc_id=item.proc_id,
        task_type=item.task_type,
        category=item.category,
        offset=item.cursor,
        draft_token_count=draft,
        forward_token_count=draft + bridge,
        includes_bridge=bool(bridge),
    )


def plan_iteration(
    queues: Mapping[str, Any],
    token_budget: int,
    current_cycle: int = 0,
    wrr_cursor: int = 0,
    consume: bool = False,
    max_num_seqs: int = 4,
    min_prefill_chunk_tokens: int = 16,
    prefill_chunk_quantum: int = 128,
    accept_stats: Mapping[Any, Sequence[float]] | None = None,
    now: float = 0.0,
    reserved_work_ids: Iterable[str] = (),
) -> AdmissionPlan:
    """Build an admission plan without mutating queues or WorkItems.

    ``consume`` is accepted for compatibility with early callers but is
    intentionally ignored.  Call :func:`commit_admission_plan` only after the
    corresponding forward succeeds.
    """
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if max_num_seqs <= 0:
        raise ValueError("max_num_seqs must be positive")
    if min_prefill_chunk_tokens <= 0 or prefill_chunk_quantum <= 0:
        raise ValueError("prefill chunk sizes must be positive")
    if min_prefill_chunk_tokens > token_budget:
        raise ValueError("min_prefill_chunk_tokens must not exceed token_budget")
    state_raw = _normalise_queues(queues)
    state = {t: {c: [_as_work_item(x, t, c, current_cycle) for x in state_raw[t][c]] for c in state_raw[t]} for t in state_raw}
    accept_stats = accept_stats or {}
    selected: set[str] = set()
    reserved = set(reserved_work_ids)
    verify: list[ExecutionSlice] = []
    prefill: list[ExecutionSlice] = []
    used = 0
    slots = 0
    cursor = int(wrr_cursor) % len(FASTSD_VERIFY_WRR_ORDER)
    no_progress = 0

    while used < token_budget and len(selected) < max_num_seqs:
        category = FASTSD_VERIFY_WRR_ORDER[cursor]
        item = _candidate_for_category(
            state, category, selected, accept_stats, now,
            token_budget - used, min_prefill_chunk_tokens, reserved,
        )
        if item is None:
            item = max(
                _all_candidates(state, selected, reserved),
                key=lambda x: (x.is_overdue_prefill, x.task_type == "verify", _priority(x, accept_stats, now)),
                default=None,
            )
        if item is not None:
            sl = _make_slice(item, token_budget - used, prefill_chunk_quantum, min_prefill_chunk_tokens)
            if sl is not None:
                selected.add(item.work_id)
                if sl.task_type == "verify":
                    verify.append(sl)
                else:
                    prefill.append(sl)
                used += sl.forward_token_count
                no_progress = 0
            else:
                no_progress += 1
        else:
            no_progress += 1
        slots += 1
        cursor = (cursor + 1) % len(FASTSD_VERIFY_WRR_ORDER)
        if no_progress >= len(FASTSD_VERIFY_WRR_ORDER):
            break

    # If a single selected Prefill is the only remaining work, fill the tail of
    # the budget while retaining the one-slice-per-WorkItem invariant.
    if used < token_budget and len(selected) < max_num_seqs:
        for item in _all_candidates(state, selected, reserved):
            if item.task_type != "prefill":
                continue
            sl = _make_slice(item, token_budget - used, token_budget, min_prefill_chunk_tokens)
            if sl is None:
                continue
            prefill.append(sl)
            selected.add(item.work_id)
            used += sl.forward_token_count
            break

    total_steps = int(wrr_cursor) + slots
    next_cycle = int(current_cycle) + total_steps // len(FASTSD_VERIFY_WRR_ORDER)
    return AdmissionPlan(
        verify_slices=verify,
        prefill_slices=prefill,
        used_tokens=used,
        token_budget=token_budget,
        max_num_seqs=max_num_seqs,
        next_wrr_cursor=total_steps % len(FASTSD_VERIFY_WRR_ORDER),
        next_cycle=next_cycle,
        slots_consumed=slots,
        selected_work_ids=[s.work_id for s in verify + prefill],
        _selected_items=[item for item in _all_candidates(state, set(), reserved) if item.work_id in selected],
    )


def reserve_admission_plan(plan: AdmissionPlan) -> None:
    for item in plan._selected_items:
        if item.state not in {"ready", "reserved"}:
            raise RuntimeError(f"work item {item.work_id} is already {item.state}")
        item.state = "reserved"


def commit_admission_plan(
    plan: AdmissionPlan,
    completed_work_ids: Iterable[str] = (),
    progress: Mapping[str, int] | None = None,
) -> None:
    completed = set(completed_work_ids)
    by_id = {item.work_id: item for item in plan._selected_items}
    slices = {s.work_id: s for s in plan.verify_slices + plan.prefill_slices}
    for work_id, item in by_id.items():
        sl = slices[work_id]
        actual_progress = sl.draft_token_count if progress is None else int(progress.get(work_id, sl.draft_token_count))
        item.cursor = max(item.cursor, sl.offset + max(0, actual_progress))
        item.last_served_cycle = plan.next_cycle
        item.missed_cycles = 0
        item.state = "finished" if work_id in completed or item.remaining_tokens <= 0 else "ready"
        item.finished = item.state == "finished"


def abort_admission_plan(plan: AdmissionPlan) -> None:
    for item in plan._selected_items:
        if item.state == "reserved":
            item.state = "ready"


def predict_next_verify_proc_ids(
    verify_queues,
    order,
    batch_size: int,
    start_idx: int = 0,
    pinned_gpu_pids=None,
    max_slots: int | None = None,
):
    if batch_size <= 0 or not order:
        return []
    pinned = set(pinned_gpu_pids or ())
    total_slots = len(order)
    slots_to_scan = total_slots if max_slots is None else max(0, min(int(max_slots), total_slots))
    predicted = []
    seen = set()
    for offset in range(slots_to_scan):
        cat = order[(start_idx + offset) % total_slots]
        for item in verify_queues.get(cat, ()):
            pid = item["proc_id"] if isinstance(item, Mapping) else item.proc_id
            if pid in pinned or pid in seen:
                continue
            predicted.append(pid)
            seen.add(pid)
            if len(predicted) >= batch_size:
                return predicted
    return predicted
