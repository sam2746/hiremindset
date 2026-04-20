"""
probe_queue 조작 공통 유틸.

- next_queue_suffix: 큐에 넣을 'q{n}' id의 다음 인덱스 계산.
- max_priority: 현재 큐의 최대 priority (빈 큐면 0).
"""

from __future__ import annotations

from hiremindset.graph.state import GraphState, ProbeItem


def next_queue_suffix(state: GraphState) -> int:
    seen_idx: set[int] = set()
    for it in state.get("probe_queue") or []:
        _add_if_q_id(it.get("id"), seen_idx)
    for pq in state.get("probing_questions") or []:
        _add_if_q_id(pq.get("queue_id"), seen_idx)
    return (max(seen_idx) + 1) if seen_idx else 0


def _add_if_q_id(sid: str | None, bucket: set[int]) -> None:
    if not sid or not sid.startswith("q"):
        return
    tail = sid[1:]
    if tail.isdigit():
        bucket.add(int(tail))


def max_priority(queue: list[ProbeItem]) -> int:
    if not queue:
        return 0
    return max(int(it.get("priority", 0)) for it in queue)
