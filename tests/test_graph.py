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


def _fake_drill_seeder(last_q, answer_text, target_claims):
    return "JWT의 서명 검증은 구체적으로 어떤 단계로 이뤄지나요?"


def _build():
    return build_graph(
        resume_extractor=_fake_resume_extractor,
        essay_extractor=_fake_essay_extractor,
        cross_check_verifier=_fake_verifier,
        suspicion_detector=_fake_detector,
        question_generator=_fake_generator,
        answer_evaluator=_fake_evaluator,
        fallback_seeder=_fake_seeder,
        drill_seeder=_fake_drill_seeder,
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


def _pass_context_probe(g, thread_id: str) -> None:
    """세션 최상위 context probe를 accept로 바로 통과."""
    _submit_answer(g, thread_id, "학부 CS 수업 프로젝트였습니다")
    _submit_decision(g, thread_id, "accept")


def test_first_emitted_question_is_context_probe():
    g = _build()
    _start_resume(g, "t-ctx")
    values = g.get_state(_cfg("t-ctx")).values
    first_pq = values["probing_questions"][0]
    assert first_pq["profile"] == "context"
    # context probe는 flag 기반이 아니므로 target_flag_id가 붙지 않는다.
    assert first_pq.get("target_flag_id") is None


def test_accept_through_context_then_flag_probe_ends_session():
    g = _build()
    _start_resume(g, "t-accept")
    _pass_context_probe(g, "t-accept")

    # 이제 두 번째 질문은 flag 기반
    _submit_answer(g, "t-accept", "네, 측정은 k6로 했습니다")
    final = _submit_decision(g, "t-accept", "accept")

    assert "__interrupt__" not in final
    values = g.get_state(_cfg("t-accept")).values
    assert values["suspicion_flags"][0]["resolved"] is True
    # 세션 종료 시 assemble_report가 점수+리포트를 채워야 한다.
    assert values["scoring"]["total"] == 100  # 모두 resolved
    assert "HireMindset 인터뷰 리포트" in values["report_markdown"]
    # decision_log도 누적돼 있어야 한다 (context + flag = 2건)
    assert len(values["decision_log"]) == 2


def test_fallback_on_flag_probe_seeds_next_question():
    g = _build()
    _start_resume(g, "t-fb")
    _pass_context_probe(g, "t-fb")

    _submit_answer(g, "t-fb", "잘 기억이 안 나요")
    out = _submit_decision(g, "t-fb", "fallback")

    assert "__interrupt__" in out
    payload = _first_interrupt_payload(out)
    assert payload["type"] == "collect_answer"

    values = g.get_state(_cfg("t-fb")).values
    # context + flag + seeded fallback = 3
    assert len(values["probing_questions"]) == 3
    assert values["probing_questions"][-1]["text"] == (
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


def test_pass_on_flag_probe_does_not_resolve_or_bump():
    """pass는 flag를 건드리지 않고 다음 큐로 넘어간다."""
    g = _build()
    _start_resume(g, "t-pass")
    _pass_context_probe(g, "t-pass")

    _submit_answer(g, "t-pass", "…")
    final = _submit_decision(g, "t-pass", "pass")

    # flag가 하나 뿐이라 큐가 비어 세션은 종료
    assert "__interrupt__" not in final
    values = g.get_state(_cfg("t-pass")).values
    flag = values["suspicion_flags"][0]
    assert flag["resolved"] is False
    assert flag["fallback_attempts"] == 0
    # 미해결 flag가 있으므로 감점이 반영돼야 한다 (severity 3 → 9점 감점, 강감점 X)
    assert values["scoring"]["total"] < 100
    assert values["scoring"]["severe_penalty_triggered"] is False


def test_drill_on_flag_probe_seeds_drill_question():
    """drill은 drill_seeder로 새 질문을 큐에 투입하고 flag는 건드리지 않는다."""
    g = _build()
    _start_resume(g, "t-drill")
    _pass_context_probe(g, "t-drill")

    _submit_answer(g, "t-drill", "JWT로 인증했어요")
    out = _submit_decision(g, "t-drill", "drill")

    assert "__interrupt__" in out
    payload = _first_interrupt_payload(out)
    assert payload["type"] == "collect_answer"

    values = g.get_state(_cfg("t-drill")).values
    # context + flag + drill = 3
    assert len(values["probing_questions"]) == 3
    last = values["probing_questions"][-1]
    assert last["text"] == "JWT의 서명 검증은 구체적으로 어떤 단계로 이뤄지나요?"
    assert last["profile"] == "mechanism"
    # drill은 flag 해소 목적이 아니므로 flag는 그대로
    flag = values["suspicion_flags"][0]
    assert flag["resolved"] is False
    assert flag["fallback_attempts"] == 0
    # 이 drill 질문 자체는 특정 flag에 연결되지 않는다
    assert last.get("target_flag_id") is None


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
