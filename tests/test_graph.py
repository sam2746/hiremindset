"""build_graph() 스모크 테스트. LLM은 모두 fake를 주입하여 네트워크 없이 돌린다.

Phase 1부터는 한 라운드가 두 번의 interrupt로 분리된다:
    emit_question → (HITL #1: answer) → evaluate_answer → (HITL #2: action) → ...
"""

from __future__ import annotations

from langgraph.types import Command

from hiremindset.graph.builder import build_graph


def _fake_resume_extractor(paragraphs, jd):
    return [
        {
            "text": f"claim-{p['id']}",
            "type": "factual",
            "source_paragraph_id": p["id"],
            "entities": ["프로젝트"],
            "confidence": 0.9,
        }
        for p in paragraphs
    ]


def _fake_essay_extractor(paragraphs, jd):
    return _fake_resume_extractor(paragraphs, jd)


def _fake_verifier(claims, pairs, paragraphs):
    return []


def _fake_detector(claims, paragraphs):
    return [
        {
            "id": "",
            "claim_ids": [claims[0]["id"]],
            "category": "cliche_template",
            "severity": 3,
            "evidence": "template feel",
            "strikes": 0,
            "fallback_attempts": 0,
            "resolved": False,
        }
    ]


def _fake_generator(item, claims_by_id, flag, paragraphs):
    return f"Q[{item['profile']}]: {item['intent']}"


def _fake_evaluator(last_q, answer_text, target_claims, flag, paragraphs):
    return {
        "q_id": last_q["id"],
        "specificity": 0.5,
        "consistency": 0.5,
        "epistemic": 0.5,
        "refusal_or_hedge": False,
        "suggest": [],
    }


def _fake_seeder(last_q, answer_text, target_claims, flag):
    return "그 프로젝트의 .gitignore에는 뭐가 들어있었나요?"


def _build():
    return build_graph(
        resume_extractor=_fake_resume_extractor,
        essay_extractor=_fake_essay_extractor,
        cross_check_verifier=_fake_verifier,
        suspicion_detector=_fake_detector,
        question_generator=_fake_generator,
        answer_evaluator=_fake_evaluator,
        fallback_seeder=_fake_seeder,
    )


def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _start_resume(g, thread_id: str):
    return g.invoke(
        {
            "documents": {
                "kind": "resume",
                "raw": "첫 문단\n둘째 문단",
                "paragraphs": [],
            },
            "jd": "",
        },
        _cfg(thread_id),
    )


def _first_interrupt_payload(result: dict) -> dict:
    raw = result["__interrupt__"][0]
    value = getattr(raw, "value", raw)
    if isinstance(value, tuple):
        value = value[0]
    assert isinstance(value, dict)
    return value


def _submit_answer(g, thread_id: str, answer: str) -> dict:
    """HITL #1에 답변 제출 → decide_action interrupt까지 진행."""
    return g.invoke(Command(resume={"answer_text": answer}), _cfg(thread_id))


def _submit_decision(
    g, thread_id: str, action: str, injected: str | None = None
) -> dict:
    payload = {"action": action}
    if injected is not None:
        payload["injected_question"] = injected
    return g.invoke(Command(resume=payload), _cfg(thread_id))


# ---------- tests ----------

def test_start_runs_until_collect_answer_interrupt():
    g = _build()
    out = _start_resume(g, "t-start")

    assert "__interrupt__" in out
    payload = _first_interrupt_payload(out)
    assert payload["type"] == "collect_answer"
    assert payload["question"].startswith("Q[")

    values = g.get_state(_cfg("t-start")).values
    assert len(values["claims"]) == 2
    assert len(values["probing_questions"]) == 1
    assert values["strategy"]["round"] == 1


def test_answer_submission_runs_evaluate_then_decide_interrupt():
    g = _build()
    _start_resume(g, "t-eval")
    out = _submit_answer(g, "t-eval", "네, 백엔드 전체를 맡았습니다.")

    assert "__interrupt__" in out
    payload = _first_interrupt_payload(out)
    assert payload["type"] == "decide_action"
    assert payload["answer_text"] == "네, 백엔드 전체를 맡았습니다."
    assert payload["ai_eval"] is not None
    assert payload["ai_eval"]["specificity"] == 0.5

    values = g.get_state(_cfg("t-eval")).values
    assert len(values["answer_eval"]) == 1
    assert values["turns"][-1]["role"] == "human"


def test_accept_ends_session_when_queue_empty():
    g = _build()
    _start_resume(g, "t-accept")
    _submit_answer(g, "t-accept", "네, 구체적으로는…")
    final = _submit_decision(g, "t-accept", "accept")

    assert "__interrupt__" not in final
    values = g.get_state(_cfg("t-accept")).values
    assert values["suspicion_flags"][0]["resolved"] is True


def test_fallback_adds_seeded_probe_and_loops():
    g = _build()
    _start_resume(g, "t-fb")
    _submit_answer(g, "t-fb", "잘 기억이 안 나요")
    out = _submit_decision(g, "t-fb", "fallback")

    assert "__interrupt__" in out
    payload = _first_interrupt_payload(out)
    assert payload["type"] == "collect_answer"

    values = g.get_state(_cfg("t-fb")).values
    assert len(values["probing_questions"]) == 2
    assert values["probing_questions"][1]["text"] == (
        "그 프로젝트의 .gitignore에는 뭐가 들어있었나요?"
    )
    assert values["suspicion_flags"][0]["fallback_attempts"] == 1


def test_inject_puts_human_question_on_top():
    g = _build()
    _start_resume(g, "t-inj")
    _submit_answer(g, "t-inj", "…")
    out = _submit_decision(
        g, "t-inj", "inject", injected="정말 혼자 하셨어요?"
    )

    assert "__interrupt__" in out
    payload = _first_interrupt_payload(out)
    assert payload["type"] == "collect_answer"

    values = g.get_state(_cfg("t-inj")).values
    assert values["probing_questions"][-1]["text"] == "정말 혼자 하셨어요?"


def test_essay_route_uses_essay_extractor():
    g = _build()
    g.invoke(
        {
            "documents": {"kind": "essay", "raw": "한 문단만", "paragraphs": []},
            "jd": "",
        },
        _cfg("t-essay"),
    )
    values = g.get_state(_cfg("t-essay")).values
    assert len(values["claims"]) == 1
