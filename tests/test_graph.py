"""build_graph() 스모크 테스트. LLM은 모두 fake를 주입하여 네트워크 없이 돌린다."""

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
    # 큐에 들어갈 최소 플래그 1개만 생성
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
    return "구체적인 수치는요?"


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
        {"configurable": {"thread_id": thread_id}},
    )


def test_start_runs_until_first_interrupt():
    g = _build()
    out = _start_resume(g, "t-start")
    assert "__interrupt__" in out

    values = g.get_state({"configurable": {"thread_id": "t-start"}}).values
    assert len(values["claims"]) == 2
    assert len(values["probing_questions"]) == 1
    assert values["strategy"]["round"] == 1


def test_accept_ends_session_when_queue_empty():
    g = _build()
    _start_resume(g, "t-accept")
    config = {"configurable": {"thread_id": "t-accept"}}

    final = g.invoke(
        Command(resume={"answer_text": "네, 구체적으로는…", "action": "accept"}),
        config,
    )
    assert "__interrupt__" not in final

    values = g.get_state(config).values
    assert len(values["turns"]) == 1
    assert values["turns"][0]["role"] == "human"
    # accept 시 타깃 flag가 resolved 처리
    assert values["suspicion_flags"][0]["resolved"] is True


def test_fallback_adds_seeded_probe_and_loops():
    g = _build()
    _start_resume(g, "t-fb")
    config = {"configurable": {"thread_id": "t-fb"}}

    out = g.invoke(
        Command(resume={"answer_text": "잘 기억이 안 나요", "action": "fallback"}),
        config,
    )
    # seeder가 넣은 새 ProbeItem이 다시 emit → 두 번째 interrupt
    assert "__interrupt__" in out

    values = g.get_state(config).values
    assert len(values["probing_questions"]) == 2
    assert values["probing_questions"][1]["text"] == "구체적인 수치는요?"
    assert values["suspicion_flags"][0]["fallback_attempts"] == 1


def test_inject_puts_human_question_on_top():
    g = _build()
    _start_resume(g, "t-inj")
    config = {"configurable": {"thread_id": "t-inj"}}

    out = g.invoke(
        Command(
            resume={
                "answer_text": "…",
                "action": "inject",
                "injected_question": "정말 혼자 하셨어요?",
            }
        ),
        config,
    )
    assert "__interrupt__" in out

    values = g.get_state(config).values
    assert values["probing_questions"][-1]["text"] == "정말 혼자 하셨어요?"


def test_essay_route_uses_essay_extractor():
    g = _build()
    g.invoke(
        {
            "documents": {"kind": "essay", "raw": "한 문단만", "paragraphs": []},
            "jd": "",
        },
        {"configurable": {"thread_id": "t-essay"}},
    )
    values = g.get_state({"configurable": {"thread_id": "t-essay"}}).values
    assert len(values["claims"]) == 1
