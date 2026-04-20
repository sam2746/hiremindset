"""
LangGraph 그래프 조립.

흐름 (Phase 1 — 2-interrupt HITL):
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
                                    collect_answer   ◀── HITL #1 (answer)    │
                                         │  (일반: evaluate→decide /       │
                                         │   즉시 accept: 둘 다 생략)        │
                                         ▼                                 │
                                      evaluate_answer (AI 채점만)            │
                                                ▼                            │
                                      decide_action    ◀── HITL #2 (action)  │
                        ┌──────────┬──────┴──────┬──────────┐                │
                        ▼          ▼             ▼          ▼                │
                 seed_fallback seed_drill (accept/pass/     (queue empty)    │
                                                 inject)         ▼           │
                        │          │             │               END ◀───────┘
                        └──────────┴──────► emit_question

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

from hiremindset.graph.nodes.assemble_report import assemble_report
from hiremindset.graph.nodes.collect_answer import collect_answer
from hiremindset.graph.nodes.cross_check import cross_check_claims
from hiremindset.graph.nodes.decide_action import decide_action
from hiremindset.graph.nodes.emit_question import emit_question
from hiremindset.graph.nodes.evaluate_answer import evaluate_answer
from hiremindset.graph.nodes.extract import (
    extract_claims_essay,
    extract_claims_resume,
)
from hiremindset.graph.nodes.flag import flag_suspicion
from hiremindset.graph.nodes.ingest import ingest_normalize
from hiremindset.graph.nodes.plan_probe import plan_probe
from hiremindset.graph.nodes.seed_drill import seed_drill_probe
from hiremindset.graph.nodes.seed_fallback import seed_fallback_probe
from hiremindset.graph.routers import route_doc_profile
from hiremindset.graph.state import GraphState


def _has_next_round(state: GraphState) -> bool:
    queue = state.get("probe_queue") or []
    if not queue:
        return False
    strategy = state.get("strategy") or {}
    meta = state.get("meta") or {}
    current_round = int(strategy.get("round", 0))
    max_rounds = int(meta.get("max_rounds", 10))
    return current_round < max_rounds


def _route_after_decide(state: GraphState) -> str:
    """decide_action 직후 분기.

    - control='fallback' → seed_fallback
    - control='drill'    → seed_drill
    - 그 외 (accept/pass/skip/inject 후 continue) → 큐 확인 후 loop/done
    """
    control = state.get("control")
    if control == "fallback":
        return "seed_fallback"
    if control == "drill":
        return "seed_drill"
    return "loop" if _has_next_round(state) else "done"


def _route_after_seed(state: GraphState) -> str:
    """seed 후: 큐에 추가됐을 테니 보통 loop."""
    return "loop" if _has_next_round(state) else "done"


def _route_after_collect(state: GraphState) -> str:
    """collect_answer 직후: 즉시 통과면 evaluate·decide를 건너뛰고 decide_action과 동일 분기."""
    if state.get("skip_evaluate_decide"):
        return _route_after_decide(state)
    return "evaluate"


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
    drill_seeder: Callable[..., Any] | None = None,
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
        partial(evaluate_answer, evaluator=answer_evaluator),
    )
    g.add_node("decide_action", decide_action)
    g.add_node(
        "seed_fallback_probe",
        partial(seed_fallback_probe, seeder=fallback_seeder),
    )
    g.add_node(
        "seed_drill_probe",
        partial(seed_drill_probe, seeder=drill_seeder),
    )
    g.add_node("assemble_report", assemble_report)

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
    g.add_conditional_edges(
        "collect_answer",
        _route_after_collect,
        {
            "evaluate": "evaluate_answer",
            "seed_fallback": "seed_fallback_probe",
            "seed_drill": "seed_drill_probe",
            "loop": "emit_question",
            "done": "assemble_report",
        },
    )
    g.add_edge("evaluate_answer", "decide_action")
    g.add_conditional_edges(
        "decide_action",
        _route_after_decide,
        {
            "seed_fallback": "seed_fallback_probe",
            "seed_drill": "seed_drill_probe",
            "loop": "emit_question",
            "done": "assemble_report",
        },
    )
    g.add_conditional_edges(
        "seed_fallback_probe",
        _route_after_seed,
        {"loop": "emit_question", "done": "assemble_report"},
    )
    g.add_conditional_edges(
        "seed_drill_probe",
        _route_after_seed,
        {"loop": "emit_question", "done": "assemble_report"},
    )
    g.add_edge("assemble_report", END)

    return g.compile(checkpointer=checkpointer or InMemorySaver())
