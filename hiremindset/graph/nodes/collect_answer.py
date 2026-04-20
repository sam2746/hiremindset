"""
collect_answer: HITL interrupt #1 — 후보자의 답변 텍스트만 수집.

설계 변경 (Phase 1):
    기존에는 한 번의 interrupt에서 답변+액션을 동시에 받았지만, 면접관이
    AI 평가를 보기 전에 accept/fallback을 먼저 결정해버리는 역순 문제가
    있었다. 이제 ``collect_answer``는 답변만 받고, AI 평가가 끝난 뒤
    ``decide_action``에서 두 번째 interrupt로 액션을 수집한다.

인터럽트 페이로드:
    {
        "type": "collect_answer",
        "question_id": str,
        "queue_id": str,
        "question": str,
        "target_flag_id": str | None,
        "profile": str | None,
        "target_claim_ids": list[str],
    }

기대 응답:
    {
        "answer_text": str,   # 필수 (빈 문자열 허용)
        "immediate_action": "accept" | None,  # accept면 AI 평가·decide 생략하고 곧바로 통과
    }

``immediate_action="accept"``일 때:
    답변을 state에 반영한 뒤 ``decide_action``과 동일한 accept 패치(flag resolved 등)를
    적용하고, ``evaluate_answer`` / 두 번째 interrupt를 건너뛴다.
"""

from __future__ import annotations

from typing import Any

from hiremindset.graph.nodes.decide_action import apply_decision_response
from hiremindset.graph.sources import build_source_excerpts, flag_evidence
from hiremindset.graph.state import GraphState, ProbingQuestion

try:  # pragma: no cover - 진입점은 LangGraph 실행 환경
    from langgraph.types import interrupt as _interrupt
except ImportError:  # pragma: no cover
    def _interrupt(payload: Any) -> Any:  # type: ignore[misc]
        raise RuntimeError("langgraph가 설치되어 있지 않아 interrupt를 사용할 수 없습니다")


def apply_answer_response(
    state: GraphState, response: dict[str, Any] | None
) -> GraphState:
    """answer-only interrupt 응답을 state 패치로 변환 (순수 로직, 테스트용)."""
    pqs = list(state.get("probing_questions") or [])
    turns = list(state.get("turns") or [])
    if not pqs or not turns:
        return {}

    resp = response or {}
    answer_text: str = str(resp.get("answer_text", ""))

    updated_turn = {**turns[-1], "answer_text": answer_text, "role": "human"}
    return {"turns": turns[:-1] + [updated_turn]}


def apply_collect_response(
    state: GraphState, response: dict[str, Any] | None
) -> GraphState:
    """interrupt 응답을 state 패치로 변환. ``immediate_action=accept``면 평가 단계 생략."""
    patch = apply_answer_response(state, response)
    if not patch:
        return {}

    resp = response or {}
    if resp.get("immediate_action") != "accept":
        return patch

    merged: GraphState = {**state, **patch}
    dec = apply_decision_response(merged, {"action": "accept"})
    out: GraphState = {**patch, **dec}
    out["skip_evaluate_decide"] = True

    log = list(out.get("decision_log") or [])
    if log:
        last = dict(log[-1])
        last["why"] = "면접관 즉시 통과 — AI 평가 생략"
        last["immediate"] = True
        log[-1] = last  # type: ignore[assignment]
        out["decision_log"] = log  # type: ignore[typeddict-item]

    return out


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
        "target_claim_ids": list(last_q.get("target_claim_ids") or []),
        "profile": last_q.get("profile"),
        "source_excerpts": build_source_excerpts(state, last_q),
        "flag_evidence": flag_evidence(state, last_q.get("target_flag_id")),
    }
    response = _interrupt(payload)
    return apply_collect_response(state, response)
