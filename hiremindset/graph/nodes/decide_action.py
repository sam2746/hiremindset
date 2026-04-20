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
        "action": "accept" | "fallback" | "drill" | "pass" | "skip" | "inject",
        "injected_question": str | None,   # inject 일 때 필수
    }

액션 의미:
- accept : 답변이 충분. 연관 flag resolved=True. control=continue
- fallback: 답변이 부족. flag.fallback_attempts+=1. seed_fallback_probe가 주변 디테일
           질문을 생성해 큐 최상위에 push. control=fallback
- drill  : 답변은 괜찮지만 더 깊게 파고들 여지가 있음. seed_drill_probe가 답변을
           기반으로 심층 기술 질문을 생성해 큐 최상위에 push. flag는 건드리지 않음.
           control=drill
- pass   : 답변이 의미 없거나 추궁할 가치 없음. flag는 그대로, 다음 큐로.
           control=continue
- skip   : 시스템이 추천한 이 질문 라인은 쓰지 않고 넘김 (답변 품질과 무관).
           flag는 그대로, 다음 큐로. control=continue
- inject : 면접관이 직접 다음 질문을 주입. 큐 최상위에 ProbeItem push. control=continue
"""

from __future__ import annotations

from typing import Any

from hiremindset.graph.queue_ops import max_priority, next_queue_suffix
from hiremindset.graph.sources import build_source_excerpts, flag_evidence
from hiremindset.graph.state import (
    AnswerEval,
    ControlSignal,
    DecisionAction,
    DecisionLogEntry,
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

    if action not in {"accept", "fallback", "drill", "pass", "skip", "inject"}:
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
    elif action == "drill":
        # flag는 그대로 둔다 (답변 자체에 의심이 있는 게 아니라 더 파고들기 위함).
        # 실제 큐 push는 seed_drill_probe 노드가 담당.
        control = "drill"
    elif action == "pass":
        # 답변이 추궁할 가치 없음 — 다음 큐로.
        control = "continue"
    elif action == "skip":
        # 추천 질문/꼬리 라인을 쓰지 않음 — 다음 큐로 (pass와 동일 분기, 로그만 다름).
        control = "continue"
    else:  # inject
        new_item = _build_injected_item(state, last_q, injected)
        patch["probe_queue"] = (state.get("probe_queue") or []) + [new_item]
        control = "continue"

    patch["control"] = control

    # decision_log: 라운드별 결정 기록 (리포트·감사용)
    turns = state.get("turns") or []
    answer_text = str(turns[-1].get("answer_text") if turns else "")
    ai_eval = _latest_eval(state, last_q["id"])
    log_entry = _build_decision_log(
        state, last_q, answer_text, action, ai_eval  # type: ignore[arg-type]
    )
    patch["decision_log"] = list(state.get("decision_log") or []) + [log_entry]

    return patch


def _latest_eval(state: GraphState, q_id: str) -> AnswerEval | None:
    for ev in reversed(state.get("answer_eval") or []):
        if ev.get("q_id") == q_id:
            return ev
    return None


_ACTION_WHY = {
    "accept": "면접관 승인",
    "fallback": "답변 부족 — 폴백 시드",
    "drill": "기술 심층 추가 질문",
    "pass": "답변 추궁 가치 없음 — 다음 큐로",
    "skip": "추천 질문 미사용 — 다음 큐로",
    "inject": "면접관 직접 주입",
}


def _build_decision_log(
    state: GraphState,
    last_q: ProbingQuestion,
    answer_text: str,
    action: DecisionAction,
    ai_eval: AnswerEval | None,
) -> DecisionLogEntry:
    strategy = state.get("strategy") or {}
    entry: DecisionLogEntry = {
        "round": int(strategy.get("round", 0)),
        "question_id": last_q["id"],
        "question": last_q.get("text", ""),
        "answer_text": answer_text,
        "action": action,
        "why": _ACTION_WHY[action],
        "fallback_used": action == "fallback",
        "chosen_probe_id": last_q.get("queue_id", ""),
    }
    if last_q.get("target_flag_id"):
        entry["flag_id"] = last_q["target_flag_id"]
    if last_q.get("profile"):
        entry["profile"] = str(last_q["profile"])
    if ai_eval:
        entry["ai_suggest"] = list(ai_eval.get("suggest") or [])
        entry["ai_specificity"] = float(ai_eval.get("specificity", 0.0))
        entry["ai_consistency"] = float(ai_eval.get("consistency", 0.0))
        entry["ai_epistemic"] = float(ai_eval.get("epistemic", 0.0))
        entry["ai_hedge"] = bool(ai_eval.get("refusal_or_hedge", False))
    return entry


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
        "asked_round": int(last_q.get("asked_round", 0)),
        "answer_text": str(last_turn.get("answer_text") or ""),
        "target_flag_id": last_q.get("target_flag_id"),
        "target_claim_ids": list(last_q.get("target_claim_ids") or []),
        "profile": last_q.get("profile"),
        "ai_eval": ai_eval,
        "source_excerpts": build_source_excerpts(state, last_q),
        "flag_evidence": flag_evidence(state, last_q.get("target_flag_id")),
    }
    response = _interrupt(payload)
    return apply_decision_response(state, response)
