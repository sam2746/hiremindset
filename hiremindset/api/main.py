"""FastAPI 엔트리.

엔드포인트:
- GET  /health          : 헬스체크
- POST /session/start   : 세션 시작 → 첫 interrupt (pending_question)까지
- POST /session/resume  : 답변/액션으로 그래프 재개 → 다음 interrupt 또는 종료

그래프는 FastAPI의 Depends(get_graph)로 주입하므로 테스트에서
``app.dependency_overrides[get_graph]``로 가짜 그래프를 넣을 수 있다.
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
    PendingQuestion,
    SessionResumeRequest,
    SessionStartRequest,
    SessionStepResponse,
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
    }


def _extract_interrupt_payload(raw: Any) -> dict[str, Any]:
    # LangGraph interrupt 객체(.value) 또는 (value, ...) 튜플 모두 수용
    value = getattr(raw, "value", None)
    if value is None and isinstance(raw, tuple) and raw:
        value = raw[0]
    if value is None and isinstance(raw, dict):
        value = raw
    return value if isinstance(value, dict) else {}


def _to_step(thread_id: str, result: dict[str, Any], graph: Any) -> SessionStepResponse:
    interrupts = result.get("__interrupt__") or []
    config = {"configurable": {"thread_id": thread_id}}
    state_values = graph.get_state(config).values or {}
    summary = _summarize(state_values)

    if interrupts:
        payload = _extract_interrupt_payload(interrupts[0])
        strategy = state_values.get("strategy") or {}
        pending = PendingQuestion(
            question_id=str(payload.get("question_id") or ""),
            queue_id=str(payload.get("queue_id") or ""),
            text=str(payload.get("question") or ""),
            profile=payload.get("profile"),
            target_flag_id=payload.get("target_flag_id"),
            asked_round=int(strategy.get("round", 0)),
        )
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


@app.post("/session/resume", response_model=SessionStepResponse)
def session_resume(
    body: SessionResumeRequest, graph: Any = Depends(get_graph)
) -> SessionStepResponse:
    if body.action == "inject" and not (body.injected_question or "").strip():
        raise HTTPException(
            status_code=422, detail="inject action requires 'injected_question'"
        )
    config = {"configurable": {"thread_id": body.thread_id}}
    payload = {
        "answer_text": body.answer_text,
        "action": body.action,
        "injected_question": body.injected_question,
    }
    try:
        result = graph.invoke(Command(resume=payload), config)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e
    return _to_step(body.thread_id, result, graph)
