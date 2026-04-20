from hiremindset.graph.nodes.flag import (
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


def test_rule_technical_probe_detects_tech_tokens():
    claims = [
        _claim("c0", "Redis 사용", "factual", ["Redis"]),
        _claim("c1", "책임감 있게 일함", "value", ["책임감"]),
    ]
    flags = _rule_technical_probe(claims)
    assert len(flags) == 1
    assert flags[0]["claim_ids"] == ["c0"]
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


def test_flag_suspicion_merges_rules_and_llm_and_assigns_ids():
    claims = [
        _claim("c0", "Redis 도입으로 응답시간 개선", "achievement", ["Redis"]),
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
    claims = [_claim("c0", "Redis 도입", "factual", ["Redis"])]
    state = {
        "claims": claims,
        "documents": {"kind": "resume", "raw": "", "paragraphs": []},
        "cross_check": [],
        "suspicion_flags": existing,
    }
    out = flag_suspicion(state, detector=lambda c, p: [])
    assert [f["id"] for f in out["suspicion_flags"]] == ["f0", "f1"]
