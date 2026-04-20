"""
emit_question: 우선순위 큐에서 top 하나를 꺼내 실제 질문 문장으로 변환.

- 턴당 1개만 생성. 큐가 비면 아무 것도 하지 않는다.
- LLM은 ``generator`` 인자로 주입 가능. 기본값은 Gemini 체인.
- state 갱신:
    probe_queue: 선택된 항목 제거
    probing_questions: 새 ProbingQuestion 추가 (queue_id로 역추적)
    turns: 빈 answer_text의 interviewer(simulator) turn 선행 기록
    strategy.round +1, last_profile 갱신
"""

from __future__ import annotations

from collections.abc import Callable

from hiremindset.graph.state import (
    Claim,
    GraphState,
    Paragraph,
    ProbeItem,
    ProbingQuestion,
    Strategy,
    SuspicionFlag,
    Turn,
)

GeneratorFn = Callable[
    [ProbeItem, dict[str, Claim], SuspicionFlag | None, list[Paragraph]], str
]


def _pick_top(queue: list[ProbeItem]) -> tuple[ProbeItem, list[ProbeItem]]:
    ordered = sorted(queue, key=lambda it: -int(it.get("priority", 0)))
    top = ordered[0]
    rest = [it for it in queue if it["id"] != top["id"]]
    return top, rest


def emit_question(
    state: GraphState, *, generator: GeneratorFn | None = None
) -> GraphState:
    queue = list(state.get("probe_queue") or [])
    if not queue:
        return {}

    top, remaining = _pick_top(queue)

    claims_by_id: dict[str, Claim] = {
        c["id"]: c for c in state.get("claims") or []
    }
    flag: SuspicionFlag | None = None
    flag_id = top.get("target_flag_id")
    if flag_id:
        flag = next(
            (f for f in state.get("suspicion_flags") or [] if f["id"] == flag_id),
            None,
        )
    paragraphs: list[Paragraph] = (
        state.get("documents") or {}
    ).get("paragraphs") or []

    if generator is None:
        from hiremindset.graph.llm import default_question_generator

        generator = default_question_generator()
    text = generator(top, claims_by_id, flag, paragraphs)

    existing_pqs = list(state.get("probing_questions") or [])
    strategy: Strategy = dict(state.get("strategy") or {})  # type: ignore[assignment]
    current_round = int(strategy.get("round", 0))

    pq: ProbingQuestion = {
        "id": f"pq{len(existing_pqs)}",
        "queue_id": top["id"],
        "text": text,
        "asked_round": current_round,
    }

    existing_turns = list(state.get("turns") or [])
    turn: Turn = {"q_id": pq["id"], "role": "simulator", "answer_text": ""}

    strategy["round"] = current_round + 1
    strategy["last_profile"] = top["profile"]

    return {
        "probe_queue": remaining,
        "probing_questions": existing_pqs + [pq],
        "turns": existing_turns + [turn],
        "strategy": strategy,
    }
