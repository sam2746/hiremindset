from hiremindset.graph.nodes.flag import (
    _merge_flags_same_paragraph_category,
    _rule_depth_collapse,
    _rule_technical_probe,
    _rule_timeline_conflict,
    flag_suspicion,
)


def _claim(cid, text, type_, entities, pid="p0"):
    return {
        "id": cid,
        "text": text,
        "type": type_,
        "source_paragraph_id": pid,
        "entities": entities,
        "confidence": 1.0,
    }


def test_rule_timeline_conflict_only_for_timeline_contradicts():
    claims = {
        "c0": _claim("c0", "A사 인턴 2021-2022", "timeline", ["A사"]),
        "c1": _claim("c1", "A사 풀타임 2020-2023", "timeline", ["A사"]),
        "c2": _claim("c2", "A사에서 성과", "achievement", ["A사"]),
    }
    pairs = [
        {"claim_ids": ("c0", "c1"), "verdict": "contradict", "rationale": "기간 겹침"},
        {"claim_ids": ("c0", "c2"), "verdict": "contradict", "rationale": "역할 차이"},
        {"claim_ids": ("c1", "c2"), "verdict": "ok", "rationale": ""},
    ]
    flags = _rule_timeline_conflict(pairs, claims)
    assert len(flags) == 1
    assert flags[0]["claim_ids"] == ["c0", "c1"]
    assert flags[0]["category"] == "timeline_conflict"
    assert flags[0]["severity"] == 4


def test_rule_technical_probe_only_when_two_or_more_tokens():
    """토큰 1개는 단순 언급이라 스킵, 2개 이상 뭉쳐 있을 때만 probe 대상."""
    claims = [
        _claim("c0", "Redis 사용", "factual", ["Redis"]),
        _claim("c1", "FastAPI + PostgreSQL 조합으로 구축", "factual", ["FastAPI", "PostgreSQL"]),
        _claim("c2", "책임감 있게 일함", "value", ["책임감"]),
    ]
    flags = _rule_technical_probe(claims)
    assert len(flags) == 1
    assert flags[0]["claim_ids"] == ["c1"]
    assert flags[0]["category"] == "technical_probe_needed"


def test_rule_depth_collapse_flags_paragraph_with_low_entity_diversity():
    claims = [
        _claim("c0", "성장했다", "value", [], pid="p1"),
        _claim("c1", "책임감 있게 임했다", "value", [], pid="p1"),
        _claim("c2", "Redis 도입", "factual", ["Redis"], pid="p2"),
    ]
    flags = _rule_depth_collapse(claims)
    assert len(flags) == 1
    assert flags[0]["category"] == "depth_collapse"
    assert set(flags[0]["claim_ids"]) == {"c0", "c1"}


def test_merge_flags_same_paragraph_category_unions_claims_and_max_severity():
    claims_by_id = {
        "c0": _claim("c0", "전략 A", "factual", [], "p0"),
        "c1": _claim("c1", "전략 B", "factual", [], "p0"),
    }
    flags_in = [
        {
            "id": "",
            "claim_ids": ["c0"],
            "category": "cliche_template",
            "severity": 3,
            "evidence": "e1",
            "strikes": 0,
            "fallback_attempts": 0,
            "resolved": False,
        },
        {
            "id": "",
            "claim_ids": ["c1"],
            "category": "cliche_template",
            "severity": 2,
            "evidence": "e2",
            "strikes": 0,
            "fallback_attempts": 0,
            "resolved": False,
        },
    ]
    out = _merge_flags_same_paragraph_category(flags_in, claims_by_id)
    assert len(out) == 1
    assert set(out[0]["claim_ids"]) == {"c0", "c1"}
    assert out[0]["severity"] == 3
    assert "e1" in out[0]["evidence"] and "e2" in out[0]["evidence"]


def test_flag_suspicion_merges_rules_and_llm_and_assigns_ids():
    claims = [
        _claim(
            "c0",
            "Redis + Kafka 조합으로 응답시간 개선",
            "achievement",
            ["Redis", "Kafka"],
        ),
    ]

    def fake_detector(claims_in, paragraphs):
        return [
            {
                "id": "",
                "claim_ids": ["c0"],
                "category": "metric_unsupported",
                "severity": 3,
                "evidence": "측정 기준 없음",
                "strikes": 0,
                "fallback_attempts": 0,
                "resolved": False,
            }
        ]

    state = {
        "claims": claims,
        "documents": {"kind": "resume", "raw": "", "paragraphs": []},
        "cross_check": [],
        "suspicion_flags": [],
    }
    out = flag_suspicion(state, detector=fake_detector)
    flags = out["suspicion_flags"]
    ids = [f["id"] for f in flags]
    assert ids == ["f0", "f1"]
    categories = {f["category"] for f in flags}
    assert categories == {"technical_probe_needed", "metric_unsupported"}


def test_flag_suspicion_collapses_two_llm_flags_same_paragraph_same_category():
    # 엔티티를 달리해 depth_collapse 룰이 동시에 안 뜨게 함 (그렇지 않으면 플래그 2종)
    claims = [
        _claim("c0", "End-to-Start로 멤버 우선", "factual", ["멤버"], "p0"),
        _claim("c1", "매칭 다음 개발", "factual", ["매칭"], "p0"),
    ]

    def fake_detector(claims_in, paragraphs):
        return [
            {
                "id": "",
                "claim_ids": ["c0"],
                "category": "cliche_template",
                "severity": 3,
                "evidence": "클리셰1",
                "strikes": 0,
                "fallback_attempts": 0,
                "resolved": False,
            },
            {
                "id": "",
                "claim_ids": ["c1"],
                "category": "cliche_template",
                "severity": 2,
                "evidence": "클리셰2",
                "strikes": 0,
                "fallback_attempts": 0,
                "resolved": False,
            },
        ]

    state = {
        "claims": claims,
        "documents": {"kind": "essay", "raw": "", "paragraphs": []},
        "cross_check": [],
        "suspicion_flags": [],
    }
    out = flag_suspicion(state, detector=fake_detector)
    flags = out["suspicion_flags"]
    assert len(flags) == 1
    assert flags[0]["id"] == "f0"
    assert set(flags[0]["claim_ids"]) == {"c0", "c1"}


def test_flag_suspicion_appends_after_existing_flags():
    existing = [
        {
            "id": "f0",
            "claim_ids": ["c0"],
            "category": "cliche_template",
            "severity": 3,
            "evidence": "prev",
            "strikes": 0,
            "fallback_attempts": 0,
            "resolved": False,
        }
    ]
    claims = [_claim("c0", "Redis + Kafka 도입", "factual", ["Redis", "Kafka"])]
    state = {
        "claims": claims,
        "documents": {"kind": "resume", "raw": "", "paragraphs": []},
        "cross_check": [],
        "suspicion_flags": existing,
    }
    out = flag_suspicion(state, detector=lambda c, p: [])
    assert [f["id"] for f in out["suspicion_flags"]] == ["f0", "f1"]
