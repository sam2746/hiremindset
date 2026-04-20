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

from hiremindset.graph.state import (
    Claim,
    ClaimType,
    CrossCheckPair,
    CrossVerdict,
    FlagCategory,
    Paragraph,
    SuspicionFlag,
)


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


# ==============================================================
# Cross-check: 주어진 후보 페어에 대해 LLM이 정합성 판정
# ==============================================================


class _CrossVerdictItem(BaseModel):
    a_id: str
    b_id: str
    verdict: CrossVerdict = Field(..., description="ok | weak | contradict")
    rationale: str = Field(..., description="판정 근거 1~2문장")


class _CrossVerdictList(BaseModel):
    items: list[_CrossVerdictItem] = Field(default_factory=list)


_CROSS_SYSTEM = (
    "당신은 후보자 주장 간 정합성을 판정하는 검증기입니다.\n"
    "- contradict: 두 주장이 양립 불가능한 숫자·기간·역할 충돌\n"
    "- weak: 양립은 가능하나 한쪽이 다른 쪽을 반증할 수 있는 긴장 관계\n"
    "- ok: 이상 없음\n"
    "근거 문단 밖의 사실을 끌어오지 마세요."
)

_CROSS_USER = (
    "# 전체 claims (id | type | text | entities)\n{claims}\n\n"
    "# 근거 문단 (id | text)\n{paragraphs}\n\n"
    "# 판정 요청 쌍\n{pairs}\n\n"
    "각 요청 쌍에 대해 verdict와 rationale(1~2문장)을 반환하세요."
)


def default_cross_check_verifier(
    *,
    model: str | None = None,
    temperature: float = 0.0,
) -> Callable[
    [list[Claim], list[tuple[str, str]], list[Paragraph]], list[CrossCheckPair]
]:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI

    model_name = model or os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    structured = llm.with_structured_output(_CrossVerdictList)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _CROSS_SYSTEM), ("human", _CROSS_USER)]
    )
    chain = prompt | structured

    def _verify(
        claims: list[Claim],
        pairs: list[tuple[str, str]],
        paragraphs: list[Paragraph],
    ) -> list[CrossCheckPair]:
        if not pairs:
            return []
        claims_str = "\n".join(
            f"{c['id']} | {c['type']} | {c['text']} | "
            f"{','.join(c.get('entities') or [])}"
            for c in claims
        )
        paragraphs_str = (
            "\n".join(f"{p['id']} | {p['text']}" for p in paragraphs) or "(없음)"
        )
        pairs_str = "\n".join(f"- {a} vs {b}" for a, b in pairs)
        result = chain.invoke(
            {"claims": claims_str, "paragraphs": paragraphs_str, "pairs": pairs_str}
        )

        pair_set = {(a, b) for a, b in pairs}
        out: list[CrossCheckPair] = []
        for item in result.items:
            key = (item.a_id, item.b_id)
            if key not in pair_set:
                rev = (item.b_id, item.a_id)
                if rev in pair_set:
                    key = rev
                else:
                    continue
            out.append(
                {
                    "claim_ids": key,
                    "verdict": item.verdict,
                    "rationale": item.rationale,
                }
            )
        return out

    return _verify


# ==============================================================
# Suspicion: LLM 판정용 카테고리만 담당 (나머지는 flag 노드의 룰)
# ==============================================================


LLMFlagCategory = Literal[
    "cliche_template",
    "metric_unsupported",
    "suspected_exaggeration",
    "inauthentic_company_ref",
]


class _LLMFlagItem(BaseModel):
    claim_id: str
    category: LLMFlagCategory
    severity: int = Field(..., ge=1, le=5, description="1(경미) / 3(주의) / 5(명확)")
    evidence: str = Field(..., description="원문 인용 또는 1줄 요약")


class _LLMFlagList(BaseModel):
    items: list[_LLMFlagItem] = Field(default_factory=list)


_FLAG_SYSTEM = (
    "당신은 면접관 관점에서 후보자 주장의 약점을 탐지합니다.\n"
    "다음 카테고리 중 해당되는 것만 flag로 생성하세요.\n"
    "- cliche_template: 자소서 템플릿/상투어, 구체 디테일 없이 가치 수사만 있는 주장\n"
    "- metric_unsupported: 수치 성과 주장에 측정 방법·기준·단위가 부재\n"
    "- suspected_exaggeration: 스코프·역할이 과장된 정황\n"
    "- inauthentic_company_ref: 회사 언급이 껍데기만, 직무·제품 맥락 부재\n"
    "severity는 1/3/5만 사용. 애매하면 플래그를 만들지 마세요."
)

_FLAG_USER = (
    "# claims (id | type | text | entities)\n{claims}\n\n"
    "# 근거 문단 (id | text)\n{paragraphs}\n\n"
    "각 claim_id에 대해 0개 이상의 flag를 반환하세요."
)


def default_suspicion_detector(
    *,
    model: str | None = None,
    temperature: float = 0.0,
) -> Callable[[list[Claim], list[Paragraph]], list[SuspicionFlag]]:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI

    model_name = model or os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    structured = llm.with_structured_output(_LLMFlagList)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _FLAG_SYSTEM), ("human", _FLAG_USER)]
    )
    chain = prompt | structured

    def _detect(
        claims: list[Claim], paragraphs: list[Paragraph]
    ) -> list[SuspicionFlag]:
        if not claims:
            return []
        claims_str = "\n".join(
            f"{c['id']} | {c['type']} | {c['text']} | "
            f"{','.join(c.get('entities') or [])}"
            for c in claims
        )
        paragraphs_str = (
            "\n".join(f"{p['id']} | {p['text']}" for p in paragraphs) or "(없음)"
        )
        result = chain.invoke({"claims": claims_str, "paragraphs": paragraphs_str})

        valid_ids = {c["id"] for c in claims}
        flags: list[SuspicionFlag] = []
        for item in result.items:
            if item.claim_id not in valid_ids:
                continue
            flags.append(
                {
                    "id": "",  # 노드에서 f{n} 부여
                    "claim_ids": [item.claim_id],
                    "category": item.category,  # type: ignore[typeddict-item]
                    "severity": int(item.severity),
                    "evidence": item.evidence,
                    "strikes": 0,
                    "fallback_attempts": 0,
                    "resolved": False,
                }
            )
        return flags

    return _detect
