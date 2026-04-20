"""
인터럽트 payload에 실어 보낼 '출처 원문' 스니펫을 조립하는 유틸.

질문이 어떤 claim을 겨냥하는지는 ``ProbingQuestion.target_claim_ids``로 알 수 있고,
claim은 ``source_paragraph_id``로 원문 문단과 연결된다. UI가 "이 질문이
어디서 나왔는지"를 보여줄 수 있도록 한 번에 묶어 반환한다.
"""

from __future__ import annotations

from typing import TypedDict

from hiremindset.graph.state import GraphState, ProbingQuestion


class SourceExcerpt(TypedDict, total=False):
    claim_id: str
    claim_text: str
    paragraph_id: str
    paragraph_text: str


def build_source_excerpts(
    state: GraphState, probing_question: ProbingQuestion
) -> list[SourceExcerpt]:
    target_ids = list(probing_question.get("target_claim_ids") or [])
    if not target_ids:
        return []

    claims_by_id = {c["id"]: c for c in state.get("claims") or []}
    paragraphs_by_id = {
        p["id"]: p
        for p in (state.get("documents") or {}).get("paragraphs") or []
    }

    result: list[SourceExcerpt] = []
    for cid in target_ids:
        claim = claims_by_id.get(cid)
        if not claim:
            continue
        excerpt: SourceExcerpt = {
            "claim_id": cid,
            "claim_text": claim.get("text", ""),
        }
        pid = claim.get("source_paragraph_id")
        if pid and pid in paragraphs_by_id:
            excerpt["paragraph_id"] = pid
            excerpt["paragraph_text"] = paragraphs_by_id[pid].get("text", "")
        result.append(excerpt)
    return result


def flag_evidence(state: GraphState, flag_id: str | None) -> str | None:
    if not flag_id:
        return None
    for f in state.get("suspicion_flags") or []:
        if f.get("id") == flag_id:
            return f.get("evidence") or None
    return None
