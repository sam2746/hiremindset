"""
flag_suspicion: claim·cross_check·문단을 종합해 의심 플래그 생성.

룰 기반 (LLM 불필요):
- timeline_conflict: cross_check에서 verdict='contradict'이고 양쪽이 timeline type
- technical_probe_needed: claim.entities에 기술 스택 토큰 포함
- depth_collapse: 같은 문단에서 2+ claim이 나왔는데 고유 entity 1개 이하

LLM 기반 (detector 주입):
- cliche_template / metric_unsupported / suspected_exaggeration / inauthentic_company_ref

플래그 ID는 기존 suspicion_flags를 이어받아 f{n}로 순차 부여한다.

중복 완화:
- 추출기가 한 문단을 잘게 쪼개 같은 카테고리(cliche 등) 플래그가 여러 개 나오면
  질문이 거의 동일하게 반복된다. ``_merge_flags_same_paragraph_category`` 로
  (문단 ID × 카테고리) 당 플래그를 하나로 합친 뒤 ID를 부여한다.
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


def _primary_paragraph_id(
    flag: SuspicionFlag, claims_by_id: dict[str, Claim]
) -> str:
    """플래그의 대표 문단: claim_ids 순서대로 첫 ``source_paragraph_id``."""
    for cid in flag.get("claim_ids") or []:
        c = claims_by_id.get(cid)
        if c and c.get("source_paragraph_id"):
            return str(c["source_paragraph_id"])
    return ""


def _merge_flags_same_paragraph_category(
    flags: list[SuspicionFlag], claims_by_id: dict[str, Claim]
) -> list[SuspicionFlag]:
    """같은 문단·같은 카테고리 플래그를 하나로 합쳐 꼬리질문이 반복되지 않게 한다."""
    buckets: dict[tuple[str, str], SuspicionFlag] = {}
    order: list[tuple[str, str]] = []

    for f in flags:
        para = _primary_paragraph_id(f, claims_by_id)
        cat = str(f.get("category", ""))
        key = (para, cat)
        if key not in buckets:
            order.append(key)
            buckets[key] = {
                **f,
                "claim_ids": list(dict.fromkeys(f.get("claim_ids") or [])),
            }
            continue
        cur = buckets[key]
        merged_ids = list(
            dict.fromkeys((cur.get("claim_ids") or []) + (f.get("claim_ids") or []))
        )
        sev = max(int(cur.get("severity", 0)), int(f.get("severity", 0)))
        ev_a = (cur.get("evidence") or "").strip()
        ev_b = (f.get("evidence") or "").strip()
        if ev_a and ev_b and ev_a != ev_b:
            evidence = f"{ev_a} / {ev_b}"
        else:
            evidence = ev_a or ev_b
        buckets[key] = {
            **cur,
            "claim_ids": merged_ids,
            "severity": sev,
            "evidence": evidence,
        }

    return [buckets[k] for k in order]


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

    combined = _merge_flags_same_paragraph_category(
        rule_flags + llm_flags, claims_by_id
    )
    new = _assign_ids(len(existing), combined)
    return {"suspicion_flags": existing + new}
