"""
collect_answer 노드가 LangGraph interrupt→Command(resume) 사이클에서
정상적으로 재개되는지 검증하는 통합 테스트.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from hiremindset.graph.nodes.collect_answer import collect_answer
from hiremindset.graph.state import GraphState


def _build_app():
    g = StateGraph(GraphState)
    g.add_node("collect", collect_answer)
    g.add_edge(START, "collect")
    g.add_edge("collect", END)
    return g.compile(checkpointer=InMemorySaver())


def _initial_state():
    return {
        "probing_questions": [
            {
                "id": "pq0",
                "queue_id": "q0",
                "text": "측정 기준은?",
                "asked_round": 0,
                "target_flag_id": "f0",
                "target_claim_ids": ["c0"],
                "profile": "numeric",
            }
        ],
        "turns": [{"q_id": "pq0", "role": "simulator", "answer_text": ""}],
        "suspicion_flags": [
            {
                "id": "f0",
                "claim_ids": ["c0"],
                "category": "metric_unsupported",
                "severity": 3,
                "evidence": "",
                "strikes": 0,
                "fallback_attempts": 0,
                "resolved": False,
            }
        ],
        "probe_queue": [],
    }


def test_interrupt_pauses_and_resumes_with_accept():
    app = _build_app()
    config = {"configurable": {"thread_id": "t-accept"}}

    first = app.invoke(_initial_state(), config)
    assert "__interrupt__" in first  # 그래프가 interrupt로 멈춰야 함

    final = app.invoke(
        Command(resume={"answer_text": "k6로 1시간 측정", "action": "accept"}),
        config,
    )
    assert final["turns"][-1]["answer_text"] == "k6로 1시간 측정"
    assert final["turns"][-1]["role"] == "human"
    assert final["suspicion_flags"][0]["resolved"] is True
    assert final["control"] == "continue"


def test_interrupt_resume_with_inject_pushes_queue_item():
    app = _build_app()
    config = {"configurable": {"thread_id": "t-inject"}}

    app.invoke(_initial_state(), config)
    final = app.invoke(
        Command(
            resume={
                "answer_text": "기억 안 납니다",
                "action": "inject",
                "injected_question": ".gitignore에 뭐가 들어있었죠?",
            }
        ),
        config,
    )
    assert len(final["probe_queue"]) == 1
    injected = final["probe_queue"][0]
    assert injected["source"] == "hitl"
    assert injected["pre_generated_text"] == ".gitignore에 뭐가 들어있었죠?"
    assert injected["profile"] == "numeric"
