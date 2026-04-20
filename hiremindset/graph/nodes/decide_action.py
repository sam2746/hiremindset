"""
decide_action: HITL interrupt #2 — AI 평가를 본 뒤 면접관이 액션을 확정한다.

``evaluate_answer``가 직전 턴의 답변을 채점해 ``answer_eval``에 누적한 뒤,
이 노드가 그 평가를 함께 실어 두 번째 interrupt를 발생시킨다. 면접관은
accept / fallback / inject 중 하나를 고르고, inject면 다음 질문을 직접 적어
보낸다.

인터럽트 페이로드:
    {
        "type": "decide_action",
        "question_id": str,
        "queue_id": str,
        "question": str,
        "answer_text": str,
        "target_flag_id": str | None,
        "target_claim_ids": list[str],
        "profile": str | None,
        "ai_eval": AnswerEval | None,
    }

기대 응답:
    {
        "action": "accept" | "fallback" | "inject",
        "injected_question": str | None,   # inject 일 때 필수
    }
"""

from __future__ import annotations

from typing import Any

from hiremindset.graph.queue_ops import max_priority, next_queue_suffix
from hiremindset.graph.state import (
    AnswerEval,
    ControlSignal,
    GraphState,
    ProbeItem,
    ProbingQuestion,
    SuspicionFlag,
)

try:  # pragma: no cover
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


def apply_decision_response(
    state: GraphState, response: dict[str, Any] | None
) -> GraphState:
    """decision interrupt 응답을 state 패치로 변환 (순수 로직, 테스트용)."""
    pqs = list(state.get("probing_questions") or [])
    if not pqs:
        return {}

    last_q: ProbingQuestion = pqs[-1]
    resp = response or {}
    action: str = str(resp.get("action", "accept"))
    injected: str = str(resp.get("injected_question") or "")

    if action not in {"accept", "fallback", "inject"}:
        raise ValueError(f"unknown action: {action}")
    if action == "inject" and not injected:
        raise ValueError("inject action requires non-empty 'injected_question'")

    flag_id = last_q.get("target_flag_id")
    flags = list(state.get("suspicion_flags") or [])
    patch: GraphState = {}
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


def _latest_eval(state: GraphState, q_id: str) -> AnswerEval | None:
    for ev in reversed(state.get("answer_eval") or []):
        if ev.get("q_id") == q_id:
            return ev
    return None


def decide_action(state: GraphState) -> GraphState:
    pqs = list(state.get("probing_questions") or [])
    turns = list(state.get("turns") or [])
    if not pqs or not turns:
        return {}

    last_q: ProbingQuestion = pqs[-1]
    last_turn = turns[-1]
    ai_eval = _latest_eval(state, last_q["id"])

    payload = {
        "type": "decide_action",
        "question_id": last_q["id"],
        "queue_id": last_q["queue_id"],
        "question": last_q["text"],
        "answer_text": str(last_turn.get("answer_text") or ""),
        "target_flag_id": last_q.get("target_flag_id"),
        "target_claim_ids": list(last_q.get("target_claim_ids") or []),
        "profile": last_q.get("profile"),
        "ai_eval": ai_eval,
    }
    response = _interrupt(payload)
    return apply_decision_response(state, response)
