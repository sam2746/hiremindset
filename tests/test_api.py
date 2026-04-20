"""FastAPI 세션 엔드포인트 스모크. 그래프는 fake LLM으로 주입 오버라이드.

Phase 1 흐름:
    /session/start → pending.phase="collect_answer"
    /session/resume (answer) → pending.phase="decide_action" (ai_eval 포함)
    /session/resume (action)  → 다음 질문 or done
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from hiremindset.api.main import app, get_graph
from hiremindset.graph.builder import build_graph
from tests.test_graph import (
    _fake_detector,
    _fake_essay_extractor,
    _fake_evaluator,
    _fake_generator,
    _fake_resume_extractor,
    _fake_seeder,
    _fake_verifier,
)

_FAKE_GRAPH = build_graph(
    resume_extractor=_fake_resume_extractor,
    essay_extractor=_fake_essay_extractor,
    cross_check_verifier=_fake_verifier,
    suspicion_detector=_fake_detector,
    question_generator=_fake_generator,
    answer_evaluator=_fake_evaluator,
    fallback_seeder=_fake_seeder,
)


def _get_fake_graph():
    return _FAKE_GRAPH


app.dependency_overrides[get_graph] = _get_fake_graph
client = TestClient(app)


def _start(text: str = "하나만"):
    return client.post(
        "/session/start",
        json={"kind": "resume", "text": text, "jd": ""},
    ).json()


def _answer(thread_id: str, text: str):
    return client.post(
        "/session/resume",
        json={
            "thread_id": thread_id,
            "phase": "collect_answer",
            "answer_text": text,
        },
    )


def _decide(thread_id: str, action: str, injected: str | None = None):
    body = {
        "thread_id": thread_id,
        "phase": "decide_action",
        "action": action,
    }
    if injected is not None:
        body["injected_question"] = injected
    return client.post("/session/resume", json=body)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_session_start_returns_collect_answer_phase():
    r = client.post(
        "/session/start",
        json={"kind": "resume", "text": "한 줄\n두 줄", "jd": ""},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["done"] is False
    assert data["thread_id"]
    pq = data["pending_question"]
    assert pq and pq["text"].startswith("Q[")
    assert pq["phase"] == "collect_answer"
    assert pq["ai_eval"] is None
    assert pq["asked_round"] == 1
    # 출처 원문이 함께 내려와야 한다.
    assert pq["source_excerpts"], "첫 질문부터 source_excerpts가 포함돼야 함"
    first = pq["source_excerpts"][0]
    assert first["claim_id"].startswith("c")
    assert first["paragraph_text"]


def test_session_start_rejects_empty_text():
    r = client.post("/session/start", json={"kind": "resume", "text": "", "jd": ""})
    assert r.status_code == 422


def test_answer_transitions_to_decide_action_phase_with_ai_eval():
    start = _start()
    r = _answer(start["thread_id"], "응답시간을 k6로 1시간 측정했습니다")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["done"] is False
    pq = data["pending_question"]
    assert pq["phase"] == "decide_action"
    assert pq["answer_text"] == "응답시간을 k6로 1시간 측정했습니다"
    assert pq["ai_eval"] is not None
    assert pq["ai_eval"]["specificity"] == 0.5


def _pass_context(thread_id: str) -> None:
    """세션 최상위 context probe를 accept로 넘기기."""
    _answer(thread_id, "학부 프로젝트였습니다")
    _decide(thread_id, "accept")


def test_accept_closes_session_after_context_and_flag_probe():
    start = _start()
    _pass_context(start["thread_id"])
    _answer(start["thread_id"], "k6로 측정했습니다")
    r = _decide(start["thread_id"], "accept")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["done"] is True
    summary = data["summary"]
    assert summary and summary["probing_questions"]
    assert summary["turns"][0]["role"] == "human"


def test_fallback_on_flag_probe_produces_next_collect_answer_question():
    start = _start()
    _pass_context(start["thread_id"])
    _answer(start["thread_id"], "잘 모르겠습니다")
    r = _decide(start["thread_id"], "fallback")
    assert r.status_code == 200
    data = r.json()
    assert data["done"] is False
    assert data["pending_question"]["phase"] == "collect_answer"
    assert data["pending_question"]["text"].startswith("그 프로젝트")


def test_inject_requires_injected_question():
    start = _start()
    _answer(start["thread_id"], "")
    r = _decide(start["thread_id"], "inject", injected="")
    assert r.status_code == 422


def test_decide_action_requires_action_field():
    start = _start()
    _answer(start["thread_id"], "답")
    r = client.post(
        "/session/resume",
        json={"thread_id": start["thread_id"], "phase": "decide_action"},
    )
    assert r.status_code == 422
