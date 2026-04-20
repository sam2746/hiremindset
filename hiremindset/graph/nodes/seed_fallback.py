"""
seed_fallback_probe: 면접관이 ``fallback``을 고른 경우에만 꼬리질문 시드를 큐에 투입.

``decide_action``이 ``control="fallback"``으로 내려오면 이 노드가 seeder를 돌려
이전 질문의 profile을 승계한 새 ProbeItem을 큐 최상위(priority = max+10)에 push한다.
``pre_generated_text``가 채워지므로 다음 ``emit_question``은 LLM을 다시 돌리지 않는다.

이후 control은 ``"continue"``로 리셋한다.
"""

from __future__ import annotations

from collections.abc import Callable

from hiremindset.graph.queue_ops import max_priority, next_queue_suffix
from hiremindset.graph.state import (
    Claim,
    GraphState,
    ProbeItem,
    ProbingQuestion,
    SuspicionFlag,
)

SeederFn = Callable[
    [ProbingQuestion, str, list[Claim], SuspicionFlag | None],
    str,
]


def seed_fallback_probe(
    state: GraphState, *, seeder: SeederFn | None = None
) -> GraphState:
    if state.get("control") != "fallback":
        return {}

    pqs = list(state.get("probing_questions") or [])
    turns = list(state.get("turns") or [])
    if not pqs or not turns:
        return {"control": "continue"}

    last_q: ProbingQuestion = pqs[-1]
    answer_text = str(turns[-1].get("answer_text") or "")

    claims = list(state.get("claims") or [])
    target_ids = set(last_q.get("target_claim_ids") or [])
    target_claims = [c for c in claims if c["id"] in target_ids]

    flag: SuspicionFlag | None = None
    flag_id = last_q.get("target_flag_id")
    if flag_id:
        flag = next(
            (f for f in state.get("suspicion_flags") or [] if f["id"] == flag_id),
            None,
        )

    if seeder is None:
        from hiremindset.graph.llm import default_fallback_seeder

        seeder = default_fallback_seeder()
    seed_text = seeder(last_q, answer_text, target_claims, flag)

    queue = list(state.get("probe_queue") or [])
    new_priority = max_priority(queue) + 10
    new_idx = next_queue_suffix(state)
    new_item: ProbeItem = {
        "id": f"q{new_idx}",
        "target_claim_ids": list(target_ids),
        "intent": "이전 답변의 공백을 파고드는 꼬리질문",
        "expected_signal": "부족했던 주변 디테일 확보",
        "profile": last_q.get("profile") or "story",  # type: ignore[typeddict-item]
        "attempts": 0,
        "priority": new_priority,
        "source": "fallback",
        "pre_generated_text": seed_text,
    }
    if flag_id:
        new_item["target_flag_id"] = flag_id

    return {
        "probe_queue": queue + [new_item],
        "control": "continue",
    }
