"""
seed_drill_probe: 면접관이 ``drill``을 고른 경우 답변을 기반으로 심층 기술 질문을 생성.

fallback이 "그 경험이 진짜냐"를 의심하며 주변 디테일을 묻는 거라면, drill은
"그 답변의 기술적 핵심을 더 깊게 본다"는 목적. 답변에 언급된 스택·개념에 대해
신입에게 요구될 만한 수준의 심층 질문을 AI가 만들어 큐 최상위에 투입한다.

- 기존 flag는 건드리지 않는다 (답변이 의심스럽지는 않음).
- 생성된 ProbeItem은 source="drill" / profile="mechanism" 고정.
- pre_generated_text가 채워지므로 다음 emit_question은 LLM을 다시 돌리지 않는다.
- 이후 control은 "continue"로 리셋.
"""

from __future__ import annotations

from collections.abc import Callable

from hiremindset.graph.queue_ops import max_priority, next_queue_suffix
from hiremindset.graph.state import (
    Claim,
    GraphState,
    ProbeItem,
    ProbingQuestion,
)

DrillSeederFn = Callable[
    [ProbingQuestion, str, list[Claim]],
    str,
]


def seed_drill_probe(
    state: GraphState, *, seeder: DrillSeederFn | None = None
) -> GraphState:
    if state.get("control") != "drill":
        return {}

    pqs = list(state.get("probing_questions") or [])
    turns = list(state.get("turns") or [])
    if not pqs or not turns:
        return {"control": "continue"}

    last_q: ProbingQuestion = pqs[-1]
    answer_text = str(turns[-1].get("answer_text") or "")

    claims = list(state.get("claims") or [])
    target_ids = set(last_q.get("target_claim_ids") or [])
    target_claims = [c for c in claims if c["id"] in target_ids]

    if seeder is None:
        from hiremindset.graph.llm import default_drill_seeder

        seeder = default_drill_seeder()
    seed_text = seeder(last_q, answer_text, target_claims)

    queue = list(state.get("probe_queue") or [])
    new_priority = max_priority(queue) + 10
    new_idx = next_queue_suffix(state)
    new_item: ProbeItem = {
        "id": f"q{new_idx}",
        "target_claim_ids": list(target_ids),
        "intent": "답변에서 드러난 기술 키워드를 더 깊게 파고든다",
        "expected_signal": "원리·트레이드오프·실패 케이스에 대한 이해",
        "profile": "mechanism",
        "attempts": 0,
        "priority": new_priority,
        "source": "drill",
        "pre_generated_text": seed_text,
    }
    # target_flag_id는 일부러 승계하지 않는다 (flag 해소 목적이 아님).

    return {
        "probe_queue": queue + [new_item],
        "control": "continue",
    }
