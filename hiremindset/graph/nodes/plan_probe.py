"""
plan_probe: suspicion_flags → 초기 ProbeItem 우선순위 큐.

이 노드는 세션 시작 시 '한 번' 돌아 초기 큐를 구성한다.
이후 HITL/fallback에서 추가되는 질문은 다른 노드에서 priority를 높여 큐에 푸시한다.
plan_probe를 다시 호출해도 이미 큐에 있는 flag는 중복 추가되지 않는다 (idempotent).

priority = severity * 10 + category_tier
category_tier가 클수록 면접에서 먼저 터뜨려야 하는 카테고리.
"""

from __future__ import annotations

from hiremindset.graph.state import (
    FlagCategory,
    GraphState,
    ProbeItem,
    ProbeProfile,
    SuspicionFlag,
)

CATEGORY_TIER: dict[str, int] = {
    "timeline_conflict": 6,
    "metric_unsupported": 5,
    "suspected_exaggeration": 4,
    "technical_probe_needed": 3,
    "cliche_template": 2,
    "inauthentic_company_ref": 1,
    "depth_collapse": 0,
}

PROFILE_MAP: dict[str, ProbeProfile] = {
    "timeline_conflict": "consistency",
    "metric_unsupported": "numeric",
    "suspected_exaggeration": "numeric",
    "technical_probe_needed": "mechanism",
    "cliche_template": "story",
    "inauthentic_company_ref": "story",
    "depth_collapse": "mechanism",
}

INTENT_TEMPLATES: dict[str, tuple[str, str]] = {
    "timeline_conflict": (
        "두 주장 사이 기간 충돌을 해소한다",
        "병행 가능성 또는 한쪽 주장의 정정",
    ),
    "metric_unsupported": (
        "수치 성과의 측정 방법과 기준을 확인한다",
        "측정 도구·환경·표본의 구체 명시",
    ),
    "suspected_exaggeration": (
        "스코프와 실제 기여 범위를 좁혀 확인한다",
        "혼자/함께 여부와 책임 영역의 경계",
    ),
    "technical_probe_needed": (
        "기술 선택 이유와 대안·트레이드오프를 묻는다",
        "선택 근거, 대안 비교, 단점 인지",
    ),
    "cliche_template": (
        "추상 수사 뒤의 구체 사례를 끌어낸다",
        "특정 사건·행동·결과의 디테일",
    ),
    "inauthentic_company_ref": (
        "회사·조직 맥락의 실제 경험 흔적을 확인한다",
        "제품/팀/업무 프로세스의 구체 언급",
    ),
    "depth_collapse": (
        "추상화된 주장에 대한 작동 원리를 확인한다",
        "어떻게 동작하는지 설명 가능 여부",
    ),
}


def _calc_priority(flag: SuspicionFlag) -> int:
    severity = int(flag.get("severity", 1))
    tier = CATEGORY_TIER.get(flag["category"], 0)
    return severity * 10 + tier


def _flag_to_item(flag: SuspicionFlag, idx: int) -> ProbeItem:
    category: FlagCategory = flag["category"]
    intent, expected = INTENT_TEMPLATES.get(
        category, ("이 주장을 구체적으로 파고든다", "구체 디테일")
    )
    profile = PROFILE_MAP.get(category, "story")
    return {
        "id": f"q{idx}",
        "target_claim_ids": list(flag["claim_ids"]),
        "intent": intent,
        "expected_signal": expected,
        "profile": profile,
        "attempts": 0,
        "target_flag_id": flag["id"],
        "priority": _calc_priority(flag),
        "source": "plan",
    }


def plan_probe(state: GraphState) -> GraphState:
    flags = list(state.get("suspicion_flags") or [])
    existing = list(state.get("probe_queue") or [])
    taken_flag_ids = {
        it.get("target_flag_id") for it in existing if it.get("target_flag_id")
    }

    start_idx = len(existing)
    new_items: list[ProbeItem] = []
    for flag in flags:
        if flag.get("resolved"):
            continue
        if flag.get("id") in taken_flag_ids:
            continue
        new_items.append(_flag_to_item(flag, start_idx + len(new_items)))

    queue = existing + new_items
    # priority 내림차순 (동일 priority는 기존 순서 유지 — Python sort는 stable)
    queue.sort(key=lambda it: -int(it.get("priority", 0)))
    return {"probe_queue": queue}
