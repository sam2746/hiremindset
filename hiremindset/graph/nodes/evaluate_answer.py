"""
evaluate_answer: 직전 턴의 답변을 AI로 채점만 한다.

Phase 1에서 fallback 시드 생성 로직은 ``seed_fallback_probe`` 노드로 분리했다.
이 노드는 이제 **면접관 결정(decide_action) 이전에** 실행되어 AI 평가를
``answer_eval``에 누적하기만 한다. control 플래그는 건드리지 않는다.
"""

from __future__ import annotations

from collections.abc import Callable

from hiremindset.graph.state import (
    AnswerEval,
    Claim,
    GraphState,
    Paragraph,
    ProbingQuestion,
    SuspicionFlag,
)

EvaluatorFn = Callable[
    [ProbingQuestion, str, list[Claim], SuspicionFlag | None, list[Paragraph]],
    AnswerEval,
]


def evaluate_answer(
    state: GraphState,
    *,
    evaluator: EvaluatorFn | None = None,
) -> GraphState:
    pqs = list(state.get("probing_questions") or [])
    turns = list(state.get("turns") or [])
    if not pqs or not turns:
        return {}

    last_q: ProbingQuestion = pqs[-1]
    last_turn = turns[-1]
    answer_text = str(last_turn.get("answer_text") or "")
    if not answer_text:
        return {}

    claims = list(state.get("claims") or [])
    target_ids = set(last_q.get("target_claim_ids") or [])
    target_claims = [c for c in claims if c["id"] in target_ids]

    flag: SuspicionFlag | None = None
    flag_id = last_q.get("target_flag_id")
    if flag_id:
        flag = next(
            (f for f in state.get("suspicion_flags") or [] if f["id"] == flag_id),
            None,
        )
    paragraphs = (state.get("documents") or {}).get("paragraphs") or []

    if evaluator is None:
        from hiremindset.graph.llm import default_answer_evaluator

        evaluator = default_answer_evaluator()
    eval_result: AnswerEval = evaluator(last_q, answer_text, target_claims, flag, paragraphs)
    eval_result["q_id"] = last_q["id"]

    existing_evals = list(state.get("answer_eval") or [])
    return {"answer_eval": existing_evals + [eval_result]}
