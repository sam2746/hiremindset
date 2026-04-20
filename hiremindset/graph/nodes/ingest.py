"""
ingest_normalize: 원문과 메타를 받아 그래프에서 쓸 표준 형태로 정리.

책임:
- documents.paragraphs 부여 (줄바꿈/빈 줄 단위로 한 문단씩 id=p{n})
- meta 기본값 채우기 (allow_hitl=True)
- strategy 카운터 초기화
- 레거시 스텁 입력(resume_text/jd_text)과도 호환
"""

from __future__ import annotations

from hiremindset.graph.state import Documents, GraphState, Meta, Paragraph, Strategy

DEFAULT_META: Meta = {
    "interview_mode": "technical",
    "max_rounds": 20,
    "max_fallbacks": 3,
    "allow_hitl": True,
}


def _split_paragraphs(text: str) -> list[Paragraph]:
    paragraphs: list[Paragraph] = []
    idx = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        paragraphs.append({"id": f"p{idx}", "text": line})
        idx += 1
    return paragraphs


def _resolve_documents(state: GraphState) -> Documents:
    existing = state.get("documents")
    if existing and existing.get("paragraphs"):
        return existing

    kind = (existing or {}).get("kind") or "resume"
    raw = (existing or {}).get("raw") or state.get("resume_text") or ""
    return {"kind": kind, "raw": raw, "paragraphs": _split_paragraphs(raw)}


def ingest_normalize(state: GraphState) -> GraphState:
    documents = _resolve_documents(state)

    meta_in = state.get("meta") or {}
    meta: Meta = {**DEFAULT_META, **meta_in}

    strategy_in = state.get("strategy") or {}
    strategy: Strategy = {
        "round": 0,
        "max_rounds": meta.get("max_rounds", DEFAULT_META["max_rounds"]),
        "fallbacks_used": 0,
        "max_fallbacks": meta.get("max_fallbacks", DEFAULT_META["max_fallbacks"]),
        **strategy_in,
    }

    jd = state.get("jd") or state.get("jd_text") or ""

    return {
        "documents": documents,
        "jd": jd,
        "meta": meta,
        "strategy": strategy,
    }
