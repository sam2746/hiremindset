"""
collect_answer 노드가 LangGraph interrupt→Command(resume) 사이클에서
정상적으로 재개되는지 검증하는 통합 테스트.

Phase 1에서는 노드 책임이 '답변 수집'으로 축소되었으므로,
control/flag 변경은 본 테스트 범위 밖이다 (decide_action 테스트에서 검증).
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
    }


def test_interrupt_pauses_and_resumes_with_answer_text():
    app = _build_app()
    config = {"configurable": {"thread_id": "t-answer"}}

    first = app.invoke(_initial_state(), config)
    assert "__interrupt__" in first

    final = app.invoke(
        Command(resume={"answer_text": "k6로 1시간 측정했습니다"}),
        config,
    )
    assert final["turns"][-1]["answer_text"] == "k6로 1시간 측정했습니다"
    assert final["turns"][-1]["role"] == "human"


def test_interrupt_payload_exposes_question_context():
    app = _build_app()
    config = {"configurable": {"thread_id": "t-payload"}}
    first = app.invoke(_initial_state(), config)

    interrupts = first.get("__interrupt__") or []
    raw = interrupts[0]
    value = getattr(raw, "value", raw)
    if isinstance(value, tuple):
        value = value[0]
    assert value["type"] == "collect_answer"
    assert value["question"] == "측정 기준은?"
    assert value["profile"] == "numeric"
    assert value["target_flag_id"] == "f0"
