"""
evaluate_answer: 직전 턴의 답변을 AI로 평가하고, fallback이면 꼬리질문 시드를 큐에 push.

- 항상 실행: answer_eval에 AnswerEval을 누적 (accept 경로도 리포트용 기록).
- control == "fallback" 일 때만:
    seeder가 다음 꼬리질문 문장을 생성 → 기존 ProbeItem의 profile을 그대로 승계한
    새 ProbeItem을 큐 최상위(priority = max+10)에 push. pre_generated_text를 채워
    다음 emit_question이 LLM을 다시 돌리지 않는다.
- 평가 verdict 자체는 AI가 '제안'만 하고 실제 판정은 collect_answer에서 면접관이 이미 내렸다
  (사용자 설계 결정 #3).

evaluator / seeder는 주입 가능. 기본은 Gemini 체인.
"""

from __future__ import annotations

from collections.abc import Callable

from hiremindset.graph.queue_ops import max_priority, next_queue_suffix
from hiremindset.graph.state import (
    AnswerEval,
    Claim,
    GraphState,
    Paragraph,
    ProbeItem,
    ProbingQuestion,
    SuspicionFlag,
)

EvaluatorFn = Callable[
    [ProbingQuestion, str, list[Claim], SuspicionFlag | None, list[Paragraph]],
    AnswerEval,
]
SeederFn = Callable[
    [ProbingQuestion, str, list[Claim], SuspicionFlag | None],
    str,
]


def evaluate_answer(
    state: GraphState,
    *,
    evaluator: EvaluatorFn | None = None,
    seeder: SeederFn | None = None,
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
    patch: GraphState = {"answer_eval": existing_evals + [eval_result]}

    if state.get("control") != "fallback":
        return patch

    # --- fallback 경로: 꼬리질문 시드 생성 ---
    if seeder is None:
        from hiremindset.graph.llm import default_fallback_seeder

        seeder = default_fallback_seeder()
    seed_text = seeder(last_q, answer_text, target_claims, flag)

    queue = list(state.get("probe_queue") or [])
    new_priority = max_priority(queue) + 10
    new_idx = next_queue_suffix(state)
    new_item: ProbeItem = {
        "id": f"q{new_idx}",
        "target_claim_ids": list(target_ids),
        "intent": "이전 답변의 공백을 파고드는 꼬리질문",
        "expected_signal": "부족했던 구체 디테일 확보",
        "profile": last_q.get("profile") or "story",  # type: ignore[typeddict-item]
        "attempts": 0,
        "priority": new_priority,
        "source": "fallback",
        "pre_generated_text": seed_text,
    }
    if flag_id:
        new_item["target_flag_id"] = flag_id

    patch["probe_queue"] = queue + [new_item]
    patch["control"] = "continue"
    return patch
