"""
LangGraph 그래프 조립.

흐름:
    START
      ▼
    ingest ── route_doc_profile ──▶ extract_resume ──┐
                 │                                    ├──▶ cross_check ──▶ flag
                 └────────▶ extract_essay ────────────┘                      │
                                                                             ▼
                                                                       plan_probe
                                                                             │
                                                ┌────────────────────────────┤
                                                ▼                            │
                                         emit_question                       │
                                                ▼                            │
                                    collect_answer  ◀── HITL interrupt       │
                                                ▼                            │
                                      evaluate_answer                        │
                                                │                            │
                                   ┌────────────┴────────────┐               │
                                   ▼                         ▼               │
                               (loop) ◀── 큐 남음          (done) ── END ◀───┘
                                         & round<max

LLM 호출 노드는 builder 인자로 fake를 주입할 수 있게 partial로 래핑한다.
기본값 None이면 노드 내부에서 Gemini 체인을 만들어 사용한다.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from hiremindset.graph.nodes.collect_answer import collect_answer
from hiremindset.graph.nodes.cross_check import cross_check_claims
from hiremindset.graph.nodes.emit_question import emit_question
from hiremindset.graph.nodes.evaluate_answer import evaluate_answer
from hiremindset.graph.nodes.extract import (
    extract_claims_essay,
    extract_claims_resume,
)
from hiremindset.graph.nodes.flag import flag_suspicion
from hiremindset.graph.nodes.ingest import ingest_normalize
from hiremindset.graph.nodes.plan_probe import plan_probe
from hiremindset.graph.routers import route_doc_profile
from hiremindset.graph.state import GraphState


def _route_after_evaluate(state: GraphState) -> str:
    queue = state.get("probe_queue") or []
    if not queue:
        return "done"
    strategy = state.get("strategy") or {}
    meta = state.get("meta") or {}
    current_round = int(strategy.get("round", 0))
    max_rounds = int(meta.get("max_rounds", 10))
    if current_round >= max_rounds:
        return "done"
    return "loop"


def build_graph(
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    resume_extractor: Callable[..., Any] | None = None,
    essay_extractor: Callable[..., Any] | None = None,
    cross_check_verifier: Callable[..., Any] | None = None,
    suspicion_detector: Callable[..., Any] | None = None,
    question_generator: Callable[..., Any] | None = None,
    answer_evaluator: Callable[..., Any] | None = None,
    fallback_seeder: Callable[..., Any] | None = None,
):
    """그래프를 조립하고 compile된 앱을 반환.

    checkpointer는 interrupt/resume에 필수. 기본 InMemorySaver.
    LLM 의존 노드 7개는 모두 주입 가능하여 테스트·오프라인에서 Gemini 없이 돌릴 수 있다.
    """

    g = StateGraph(GraphState)

    g.add_node("ingest", ingest_normalize)
    g.add_node(
        "extract_resume",
        partial(extract_claims_resume, extractor=resume_extractor),
    )
    g.add_node(
        "extract_essay",
        partial(extract_claims_essay, extractor=essay_extractor),
    )
    g.add_node(
        "cross_check",
        partial(cross_check_claims, verifier=cross_check_verifier),
    )
    g.add_node("flag", partial(flag_suspicion, detector=suspicion_detector))
    g.add_node("plan_probe", plan_probe)
    g.add_node(
        "emit_question",
        partial(emit_question, generator=question_generator),
    )
    g.add_node("collect_answer", collect_answer)
    g.add_node(
        "evaluate_answer",
        partial(
            evaluate_answer,
            evaluator=answer_evaluator,
            seeder=fallback_seeder,
        ),
    )

    g.add_edge(START, "ingest")
    g.add_conditional_edges(
        "ingest",
        route_doc_profile,
        {"resume": "extract_resume", "essay": "extract_essay"},
    )
    g.add_edge("extract_resume", "cross_check")
    g.add_edge("extract_essay", "cross_check")
    g.add_edge("cross_check", "flag")
    g.add_edge("flag", "plan_probe")
    g.add_edge("plan_probe", "emit_question")
    g.add_edge("emit_question", "collect_answer")
    g.add_edge("collect_answer", "evaluate_answer")
    g.add_conditional_edges(
        "evaluate_answer",
        _route_after_evaluate,
        {"loop": "emit_question", "done": END},
    )

    return g.compile(checkpointer=checkpointer or InMemorySaver())
