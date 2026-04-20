"""FastAPI 엔트리.

엔드포인트:
- GET  /health          : 헬스체크
- POST /session/start   : 세션 시작 → 첫 interrupt (pending_question)까지
- POST /session/resume  : 답변/액션으로 그래프 재개 → 다음 interrupt 또는 종료

그래프는 FastAPI의 Depends(get_graph)로 주입하므로 테스트에서
``app.dependency_overrides[get_graph]``로 가짜 그래프를 넣을 수 있다.

HITL은 한 라운드에 두 번 인터럽트한다 (phase='collect_answer' → 'decide_action').
"""

from __future__ import annotations

import os
import uuid
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command

from hiremindset.api.schemas import (
    AnswerEvalSnapshot,
    PendingQuestion,
    SessionResumeRequest,
    SessionStartRequest,
    SessionStepResponse,
    SourceExcerpt,
)
from hiremindset.graph.builder import build_graph

load_dotenv()

app = FastAPI(title="HireMindset API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def get_graph():
    return build_graph()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _summarize(values: dict[str, Any]) -> dict[str, Any]:
    return {
        "claims": values.get("claims") or [],
        "cross_check": values.get("cross_check") or [],
        "suspicion_flags": values.get("suspicion_flags") or [],
        "probing_questions": values.get("probing_questions") or [],
        "turns": values.get("turns") or [],
        "answer_eval": values.get("answer_eval") or [],
        "strategy": values.get("strategy") or {},
        "documents": values.get("documents") or {},
    }


def _extract_interrupt_payload(raw: Any) -> dict[str, Any]:
    # LangGraph interrupt 객체(.value) 또는 (value, ...) 튜플 모두 수용
    value = getattr(raw, "value", None)
    if value is None and isinstance(raw, tuple) and raw:
        value = raw[0]
    if value is None and isinstance(raw, dict):
        value = raw
    return value if isinstance(value, dict) else {}


def _build_pending(payload: dict[str, Any], state_values: dict[str, Any]) -> PendingQuestion:
    strategy = state_values.get("strategy") or {}
    phase = payload.get("type") or "collect_answer"
    if phase not in ("collect_answer", "decide_action"):
        phase = "collect_answer"

    ai_eval_raw = payload.get("ai_eval") if phase == "decide_action" else None
    ai_eval = None
    if isinstance(ai_eval_raw, dict):
        ai_eval = AnswerEvalSnapshot(
            specificity=float(ai_eval_raw.get("specificity", 0.0)),
            consistency=float(ai_eval_raw.get("consistency", 0.0)),
            epistemic=float(ai_eval_raw.get("epistemic", 0.0)),
            refusal_or_hedge=bool(ai_eval_raw.get("refusal_or_hedge", False)),
            suggest=list(ai_eval_raw.get("suggest") or []),
        )

    excerpts_raw = payload.get("source_excerpts") or []
    source_excerpts = [
        SourceExcerpt(
            claim_id=str(e.get("claim_id") or ""),
            claim_text=str(e.get("claim_text") or ""),
            paragraph_id=e.get("paragraph_id"),
            paragraph_text=e.get("paragraph_text"),
        )
        for e in excerpts_raw
        if isinstance(e, dict)
    ]

    return PendingQuestion(
        phase=phase,  # type: ignore[arg-type]
        question_id=str(payload.get("question_id") or ""),
        queue_id=str(payload.get("queue_id") or ""),
        text=str(payload.get("question") or ""),
        profile=payload.get("profile"),
        target_flag_id=payload.get("target_flag_id"),
        target_claim_ids=list(payload.get("target_claim_ids") or []),
        asked_round=int(strategy.get("round", 0)),
        source_excerpts=source_excerpts,
        flag_evidence=payload.get("flag_evidence"),
        answer_text=payload.get("answer_text") if phase == "decide_action" else None,
        ai_eval=ai_eval,
    )


def _to_step(thread_id: str, result: dict[str, Any], graph: Any) -> SessionStepResponse:
    interrupts = result.get("__interrupt__") or []
    config = {"configurable": {"thread_id": thread_id}}
    state_values = graph.get_state(config).values or {}
    summary = _summarize(state_values)

    if interrupts:
        payload = _extract_interrupt_payload(interrupts[0])
        pending = _build_pending(payload, state_values)
        return SessionStepResponse(
            thread_id=thread_id,
            done=False,
            pending_question=pending,
            summary=summary,
        )

    return SessionStepResponse(thread_id=thread_id, done=True, summary=summary)


@app.post("/session/start", response_model=SessionStepResponse)
def session_start(
    body: SessionStartRequest, graph: Any = Depends(get_graph)
) -> SessionStepResponse:
    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = graph.invoke(
            {
                "documents": {
                    "kind": body.kind,
                    "raw": body.text,
                    "paragraphs": [],
                },
                "jd": body.jd,
            },
            config,
        )
    except Exception as e:  # noqa: BLE001 — 개발 중 원인 그대로 노출
        raise HTTPException(status_code=500, detail=str(e)) from e
    return _to_step(thread_id, result, graph)


def _resume_payload(body: SessionResumeRequest) -> dict[str, Any]:
    if body.phase == "collect_answer":
        return {"answer_text": body.answer_text}

    if body.action is None:
        raise HTTPException(
            status_code=422, detail="phase='decide_action' requires 'action'"
        )
    if body.action == "inject" and not (body.injected_question or "").strip():
        raise HTTPException(
            status_code=422, detail="inject action requires 'injected_question'"
        )
    return {
        "action": body.action,
        "injected_question": body.injected_question,
    }


@app.post("/session/resume", response_model=SessionStepResponse)
def session_resume(
    body: SessionResumeRequest, graph: Any = Depends(get_graph)
) -> SessionStepResponse:
    resume_value = _resume_payload(body)
    config = {"configurable": {"thread_id": body.thread_id}}
    try:
        result = graph.invoke(Command(resume=resume_value), config)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e
    return _to_step(body.thread_id, result, graph)
