"""
collect_answer: HITL interrupt로 면접관 답변·액션을 수신.

LangGraph의 ``interrupt(payload)``로 그래프를 정지시키고, 면접관이 화면에서
답변을 채워 ``Command(resume=...)``로 재개하면 그 응답이 노드 안에서 반환된다.

액션 3종 (사용자 설계 결정 #2, #3):
- accept   : 답변이 충분. 연관 flag.resolved=True. control="continue"
- fallback : 답변 부족. evaluate_answer가 AI 꼬리질문을 생성해 큐 최상위 push. control="fallback"
- inject   : 면접관이 새 질문을 직접 주입. 큐 최상위에 ProbeItem push, 원 질문은 답변만 기록. control="continue"

인터럽트 페이로드:
    {
        "type": "collect_answer",
        "question_id": str,
        "queue_id": str,
        "question": str,
        "target_flag_id": str | None,
        "profile": str | None,
    }

기대 응답:
    {
        "answer_text": str,                    # 필수
        "action": "accept" | "fallback" | "inject",
        "injected_question": str | None,       # action == "inject" 일 때 필수
    }
"""

from __future__ import annotations

from typing import Any

from hiremindset.graph.queue_ops import max_priority, next_queue_suffix
from hiremindset.graph.state import (
    ControlSignal,
    GraphState,
    ProbeItem,
    ProbingQuestion,
    SuspicionFlag,
)

try:  # pragma: no cover - 진입점은 LangGraph 실행 환경
    from langgraph.types import interrupt as _interrupt
except ImportError:  # pragma: no cover
    def _interrupt(payload: Any) -> Any:  # type: ignore[misc]
        raise RuntimeError("langgraph가 설치되어 있지 않아 interrupt를 사용할 수 없습니다")


def _mark_resolved(flags: list[SuspicionFlag], flag_id: str) -> list[SuspicionFlag]:
    return [{**f, "resolved": True} if f["id"] == flag_id else f for f in flags]


def _bump_fallback(flags: list[SuspicionFlag], flag_id: str) -> list[SuspicionFlag]:
    return [
        {**f, "fallback_attempts": int(f.get("fallback_attempts", 0)) + 1}
        if f["id"] == flag_id
        else f
        for f in flags
    ]


def _build_injected_item(
    state: GraphState, last_q: ProbingQuestion, question_text: str
) -> ProbeItem:
    queue = list(state.get("probe_queue") or [])
    new_id = f"q{next_queue_suffix(state)}"
    new_priority = max_priority(queue) + 10
    profile = last_q.get("profile") or "story"
    item: ProbeItem = {
        "id": new_id,
        "target_claim_ids": list(last_q.get("target_claim_ids") or []),
        "intent": "면접관이 직접 주입한 꼬리 질문",
        "expected_signal": "(면접관 지정)",
        "profile": profile,  # type: ignore[typeddict-item]
        "attempts": 0,
        "priority": new_priority,
        "source": "hitl",
        "pre_generated_text": question_text,
    }
    if last_q.get("target_flag_id"):
        item["target_flag_id"] = last_q["target_flag_id"]
    return item


def apply_interrupt_response(
    state: GraphState, response: dict[str, Any] | None
) -> GraphState:
    """interrupt의 resume 값으로 받은 응답을 state 패치로 변환 (순수 로직, 테스트용)."""
    pqs = list(state.get("probing_questions") or [])
    turns = list(state.get("turns") or [])
    if not pqs or not turns:
        return {}

    last_q: ProbingQuestion = pqs[-1]
    resp = response or {}
    answer_text: str = str(resp.get("answer_text", ""))
    action: str = str(resp.get("action", "accept"))
    injected: str = str(resp.get("injected_question") or "")

    if action not in {"accept", "fallback", "inject"}:
        raise ValueError(f"unknown action: {action}")
    if action == "inject" and not injected:
        raise ValueError("inject action requires non-empty 'injected_question'")

    patch: GraphState = {}

    # 1) 현재 턴의 답변 기록 (role을 human으로 승격)
    updated_turn = {**turns[-1], "answer_text": answer_text, "role": "human"}
    patch["turns"] = turns[:-1] + [updated_turn]

    flag_id = last_q.get("target_flag_id")
    flags = list(state.get("suspicion_flags") or [])
    control: ControlSignal

    if action == "accept":
        if flag_id:
            patch["suspicion_flags"] = _mark_resolved(flags, flag_id)
        control = "continue"
    elif action == "fallback":
        if flag_id:
            patch["suspicion_flags"] = _bump_fallback(flags, flag_id)
        control = "fallback"
    else:  # inject
        new_item = _build_injected_item(state, last_q, injected)
        patch["probe_queue"] = (state.get("probe_queue") or []) + [new_item]
        control = "continue"

    patch["control"] = control
    return patch


def collect_answer(state: GraphState) -> GraphState:
    pqs = list(state.get("probing_questions") or [])
    turns = list(state.get("turns") or [])
    if not pqs or not turns:
        return {}

    last_q: ProbingQuestion = pqs[-1]
    payload = {
        "type": "collect_answer",
        "question_id": last_q["id"],
        "queue_id": last_q["queue_id"],
        "question": last_q["text"],
        "target_flag_id": last_q.get("target_flag_id"),
        "profile": last_q.get("profile"),
    }
    response = _interrupt(payload)
    return apply_interrupt_response(state, response)
