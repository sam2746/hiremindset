"""
flag_suspicion: claim·cross_check·문단을 종합해 의심 플래그 생성.

룰 기반 (LLM 불필요):
- timeline_conflict: cross_check에서 verdict='contradict'이고 양쪽이 timeline type
- technical_probe_needed: claim.entities에 기술 스택 토큰 포함
- depth_collapse: 같은 문단에서 2+ claim이 나왔는데 고유 entity 1개 이하

LLM 기반 (detector 주입):
- cliche_template / metric_unsupported / suspected_exaggeration / inauthentic_company_ref

플래그 ID는 기존 suspicion_flags를 이어받아 f{n}로 순차 부여한다.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from hiremindset.graph.state import (
    Claim,
    CrossCheckPair,
    GraphState,
    Paragraph,
    SuspicionFlag,
)

TECH_TOKENS: frozenset[str] = frozenset(
    {
        "python", "java", "kotlin", "typescript", "javascript", "go", "rust",
        "django", "flask", "fastapi", "spring", "react", "vue", "nextjs", "next.js",
        "redis", "kafka", "postgres", "postgresql", "mysql", "mongodb",
        "docker", "kubernetes", "k8s", "aws", "gcp", "azure",
        "langgraph", "langchain", "openai", "gemini",
    }
)

DetectorFn = Callable[[list[Claim], list[Paragraph]], list[SuspicionFlag]]


def _new_flag(
    *,
    claim_ids: list[str],
    category: str,
    severity: int,
    evidence: str,
) -> SuspicionFlag:
    return {
        "id": "",  # 노드에서 부여
        "claim_ids": claim_ids,
        "category": category,  # type: ignore[typeddict-item]
        "severity": severity,
        "evidence": evidence,
        "strikes": 0,
        "fallback_attempts": 0,
        "resolved": False,
    }


def _rule_timeline_conflict(
    cross_pairs: list[CrossCheckPair], claims_by_id: dict[str, Claim]
) -> list[SuspicionFlag]:
    flags: list[SuspicionFlag] = []
    for pair in cross_pairs:
        if pair.get("verdict") != "contradict":
            continue
        a_id, b_id = pair["claim_ids"]
        a, b = claims_by_id.get(a_id), claims_by_id.get(b_id)
        if not a or not b:
            continue
        if a["type"] != "timeline" or b["type"] != "timeline":
            continue
        flags.append(
            _new_flag(
                claim_ids=[a_id, b_id],
                category="timeline_conflict",
                severity=4,
                evidence=pair.get("rationale", ""),
            )
        )
    return flags


def _rule_technical_probe(claims: list[Claim]) -> list[SuspicionFlag]:
    """한 claim에 기술 스택 토큰이 2개 이상 뭉쳐 있을 때만 flag.
    토큰 1개는 단순 언급일 수 있어 질문 밀도만 높이므로 제외한다.
    """
    flags: list[SuspicionFlag] = []
    for c in claims:
        ents = {e.strip().lower() for e in (c.get("entities") or [])}
        hits = ents & TECH_TOKENS
        if len(hits) < 2:
            continue
        flags.append(
            _new_flag(
                claim_ids=[c["id"]],
                category="technical_probe_needed",
                severity=2,
                evidence=f"기술 키워드({', '.join(sorted(hits))}) 언급: {c['text']}",
            )
        )
    return flags


def _rule_depth_collapse(claims: list[Claim]) -> list[SuspicionFlag]:
    by_para: dict[str, list[Claim]] = defaultdict(list)
    for c in claims:
        by_para[c["source_paragraph_id"]].append(c)

    flags: list[SuspicionFlag] = []
    for pid, items in by_para.items():
        if len(items) < 2:
            continue
        unique: set[str] = set()
        for c in items:
            for e in c.get("entities") or []:
                unique.add(e.strip().lower())
        if len(unique) <= 1:
            flags.append(
                _new_flag(
                    claim_ids=[c["id"] for c in items],
                    category="depth_collapse",
                    severity=2,
                    evidence=f"문단 {pid}: {len(items)}개 주장에 고유 엔티티 {len(unique)}개",
                )
            )
    return flags


def _assign_ids(start: int, flags: list[SuspicionFlag]) -> list[SuspicionFlag]:
    return [{**f, "id": f"f{start + i}"} for i, f in enumerate(flags)]


def flag_suspicion(
    state: GraphState, *, detector: DetectorFn | None = None
) -> GraphState:
    claims = list(state.get("claims") or [])
    paragraphs = (state.get("documents") or {}).get("paragraphs") or []
    cross_pairs = list(state.get("cross_check") or [])
    existing = list(state.get("suspicion_flags") or [])

    claims_by_id = {c["id"]: c for c in claims}
    rule_flags: list[SuspicionFlag] = []
    rule_flags += _rule_timeline_conflict(cross_pairs, claims_by_id)
    rule_flags += _rule_technical_probe(claims)
    rule_flags += _rule_depth_collapse(claims)

    llm_flags: list[SuspicionFlag] = []
    if claims:
        if detector is None:
            from hiremindset.graph.llm import default_suspicion_detector

            detector = default_suspicion_detector()
        llm_flags = detector(claims, paragraphs)

    new = _assign_ids(len(existing), rule_flags + llm_flags)
    return {"suspicion_flags": existing + new}
