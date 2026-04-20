"""
LLM 어댑터: 기본 claim 추출기(Gemini).

- 노드에는 이 함수를 직접 호출하지 않고, ``extractor=`` 파라미터로 주입한다.
- 실제 호출이 필요할 때만 ``default_claim_extractor(...)``가 체인을 구성한다.
- 구조화 출력은 Pydantic 모델로 고정한다.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field

from hiremindset.graph.state import Claim, ClaimType, Paragraph


class ExtractedClaim(BaseModel):
    """LLM이 한 건의 검증 가능한 주장을 담아내는 모양."""

    text: str = Field(..., description="검증 가능한 단위 주장")
    type: ClaimType = Field(..., description="factual | achievement | timeline | value")
    source_paragraph_id: str = Field(..., description="근거 문단의 id(p0, p1...)")
    entities: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class ExtractedClaimList(BaseModel):
    items: list[ExtractedClaim] = Field(default_factory=list)


_SYSTEM = (
    "당신은 채용 면접 분석용 사실 추출기입니다.\n"
    "주어진 문단들에서 '검증 가능한 단위 주장'만 뽑아냅니다.\n"
    "각 주장은 반드시 가장 가까운 문단의 id(p0, p1, ...)를 source_paragraph_id로 답니다.\n"
    "중복 주장은 하나로 합치고, 주관적 수사/일반론만 있는 문단은 비워둡니다."
)

_USER_TEMPLATE = (
    "# 문서 종류: {kind}\n"
    "# JD (참고, 비어 있을 수 있음)\n{jd}\n\n"
    "# 문단 목록 (id | 본문)\n{paragraphs}\n\n"
    "# 규칙\n"
    "- 이력서면 학력·자격·수상, 근무/프로젝트 기간, 스택, 수치 성과를 개별 주장으로 쪼갠다.\n"
    "- 자기소개서면 사건·선택·성과 위주로 주장화하고 원문의 뉘앙스를 유지한다.\n"
    "- type은 factual | achievement | timeline | value 중 하나.\n"
    "- entities에는 회사명/스택/숫자 등 핵심 키워드만.\n"
    "- confidence는 본인 확신도(0.0~1.0)."
)


def _format_paragraphs(paragraphs: list[Paragraph]) -> str:
    return "\n".join(f"{p['id']} | {p['text']}" for p in paragraphs)


def default_claim_extractor(
    kind: Literal["resume", "essay"],
    *,
    model: str | None = None,
    temperature: float = 0.0,
) -> Callable[[list[Paragraph], str], list[Claim]]:
    """Gemini 기반 claim 추출기. 반환 함수는 순수하게 문단/JD만 받는다."""

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI

    model_name = model or os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    structured = llm.with_structured_output(ExtractedClaimList)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM), ("human", _USER_TEMPLATE)]
    )
    chain = prompt | structured

    def _extract(paragraphs: list[Paragraph], jd: str) -> list[Claim]:
        if not paragraphs:
            return []
        result = chain.invoke(
            {
                "kind": kind,
                "jd": jd or "(없음)",
                "paragraphs": _format_paragraphs(paragraphs),
            }
        )
        valid_ids = {p["id"] for p in paragraphs}
        claims: list[Claim] = []
        for item in result.items:
            src = (
                item.source_paragraph_id
                if item.source_paragraph_id in valid_ids
                else paragraphs[0]["id"]
            )
            claims.append(
                {
                    "id": "",  # 노드에서 c{n} 부여
                    "text": item.text,
                    "type": item.type,
                    "source_paragraph_id": src,
                    "entities": list(item.entities),
                    "confidence": float(item.confidence),
                }
            )
        return claims

    return _extract
