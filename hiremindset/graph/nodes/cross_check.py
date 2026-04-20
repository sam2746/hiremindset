"""
cross_check_claims: claim 간 정합성(타임라인·수치·역할) 판정 노드.

흐름:
1) 룰 기반 후보 페어 선별 (공유 entity / 둘 다 timeline)
2) 후보 페어를 LLM 검증기에 배치로 넘겨 판정
3) verdict/rationale을 state["cross_check"]에 기록

LLM 호출은 ``verifier`` 인자로 주입 가능하며, 기본값은 Gemini 체인.
"""

from __future__ import annotations

from collections.abc import Callable

from hiremindset.graph.state import Claim, CrossCheckPair, GraphState, Paragraph

VerifierFn = Callable[
    [list[Claim], list[tuple[str, str]], list[Paragraph]], list[CrossCheckPair]
]


def _norm_entities(entities: list[str] | None) -> set[str]:
    return {e.strip().lower() for e in (entities or []) if e and e.strip()}


def candidate_pairs(claims: list[Claim], *, max_pairs: int = 20) -> list[tuple[str, str]]:
    """후보 페어만 선별. LLM 호출 비용 방어용."""
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(claims):
        a_ents = _norm_entities(a.get("entities"))
        a_is_timeline = a["type"] == "timeline"
        for b in claims[i + 1 :]:
            b_ents = _norm_entities(b.get("entities"))
            shared = bool(a_ents & b_ents)
            both_timeline = a_is_timeline and b["type"] == "timeline"
            if shared or both_timeline:
                pairs.append((a["id"], b["id"]))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def cross_check_claims(
    state: GraphState, *, verifier: VerifierFn | None = None
) -> GraphState:
    claims = list(state.get("claims") or [])
    if len(claims) < 2:
        return {"cross_check": []}

    pairs = candidate_pairs(claims)
    if not pairs:
        return {"cross_check": []}

    if verifier is None:
        from hiremindset.graph.llm import default_cross_check_verifier

        verifier = default_cross_check_verifier()

    paragraphs = (state.get("documents") or {}).get("paragraphs") or []
    result = verifier(claims, pairs, paragraphs)
    return {"cross_check": result}
