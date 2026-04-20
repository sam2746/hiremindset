"""
extract_claims_resume / extract_claims_essay: 문서 종류별 claim 추출 노드.

노드는 LLM 세부를 몰라야 하므로 실제 호출은 ``extractor`` 인자로 주입받는다.
기본값이 None이면 ``hiremindset.graph.llm.default_claim_extractor``가 Gemini 체인을 만든다.
테스트는 가짜 extractor를 주입해 네트워크 없이 검증한다.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from hiremindset.graph.state import Claim, GraphState, Paragraph

ExtractorFn = Callable[[list[Paragraph], str], list[Claim]]


def _assign_ids(existing_count: int, new_claims: list[Claim]) -> list[Claim]:
    return [{**c, "id": f"c{existing_count + i}"} for i, c in enumerate(new_claims)]


def _run_extraction(
    state: GraphState,
    kind: Literal["resume", "essay"],
    extractor: ExtractorFn | None,
) -> GraphState:
    existing = list(state.get("claims") or [])
    documents = state.get("documents")
    if not documents or not documents.get("paragraphs"):
        return {"claims": existing}

    if extractor is None:
        from hiremindset.graph.llm import default_claim_extractor

        extractor = default_claim_extractor(kind)

    new_claims = extractor(documents["paragraphs"], state.get("jd") or "")
    assigned = _assign_ids(len(existing), new_claims)
    return {"claims": existing + assigned}


def extract_claims_resume(
    state: GraphState, *, extractor: ExtractorFn | None = None
) -> GraphState:
    return _run_extraction(state, "resume", extractor)


def extract_claims_essay(
    state: GraphState, *, extractor: ExtractorFn | None = None
) -> GraphState:
    return _run_extraction(state, "essay", extractor)
