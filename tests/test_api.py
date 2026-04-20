"""FastAPI 세션 엔드포인트 스모크. 그래프는 fake LLM으로 주입 오버라이드."""

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


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_session_start_returns_pending_question():
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
    assert pq["asked_round"] == 1


def test_session_start_rejects_empty_text():
    r = client.post("/session/start", json={"kind": "resume", "text": "", "jd": ""})
    assert r.status_code == 422


def test_session_resume_accept_closes_session():
    start = client.post(
        "/session/start",
        json={"kind": "resume", "text": "하나만", "jd": ""},
    ).json()
    r = client.post(
        "/session/resume",
        json={
            "thread_id": start["thread_id"],
            "answer_text": "구체적으로 답변",
            "action": "accept",
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["done"] is True
    summary = data["summary"]
    assert summary and summary["probing_questions"]
    assert summary["turns"][0]["role"] == "human"


def test_session_resume_fallback_produces_next_question():
    start = client.post(
        "/session/start",
        json={"kind": "resume", "text": "한 줄", "jd": ""},
    ).json()
    r = client.post(
        "/session/resume",
        json={
            "thread_id": start["thread_id"],
            "answer_text": "잘 모르겠습니다",
            "action": "fallback",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["done"] is False
    assert data["pending_question"]["text"] == "구체적인 수치는요?"


def test_session_resume_rejects_inject_without_text():
    start = client.post(
        "/session/start",
        json={"kind": "resume", "text": "한 줄", "jd": ""},
    ).json()
    r = client.post(
        "/session/resume",
        json={
            "thread_id": start["thread_id"],
            "action": "inject",
            "answer_text": "",
            "injected_question": "",
        },
    )
    assert r.status_code == 422
