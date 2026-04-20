from hiremindset.graph.nodes.cross_check import (
    candidate_pairs,
    cross_check_claims,
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


def test_candidate_pairs_picks_shared_entity():
    claims = [
        _claim("c0", "A사에서 API 마이그레이션", "achievement", ["A사", "API"]),
        _claim("c1", "A사 인턴 2021-2022", "timeline", ["A사", "2021", "2022"]),
        _claim("c2", "무관한 수상 경력", "factual", ["상장"]),
    ]
    pairs = candidate_pairs(claims)
    assert ("c0", "c1") in pairs
    assert ("c0", "c2") not in pairs
    assert ("c1", "c2") not in pairs


def test_candidate_pairs_picks_both_timeline_even_without_shared_entity():
    claims = [
        _claim("c0", "A사 인턴 2021-2022", "timeline", ["A사"]),
        _claim("c1", "B사 풀타임 2020-2023", "timeline", ["B사"]),
    ]
    pairs = candidate_pairs(claims)
    assert pairs == [("c0", "c1")]


def test_cross_check_returns_empty_when_less_than_two_claims():
    out = cross_check_claims({"claims": [_claim("c0", "x", "factual", [])]})
    assert out == {"cross_check": []}


def test_cross_check_does_not_invoke_verifier_when_no_candidate_pairs():
    calls = {"n": 0}

    def fake_verifier(claims, pairs, paragraphs):
        calls["n"] += 1
        return []

    # 서로 공유 entity 없고 둘 다 timeline 아님 → 후보 없음
    state = {
        "claims": [
            _claim("c0", "x", "factual", ["alpha"]),
            _claim("c1", "y", "achievement", ["beta"]),
        ]
    }
    out = cross_check_claims(state, verifier=fake_verifier)
    assert out == {"cross_check": []}
    assert calls["n"] == 0


def test_cross_check_invokes_verifier_and_returns_pairs():
    claims = [
        _claim("c0", "A사 인턴 2021-2022", "timeline", ["A사"]),
        _claim("c1", "A사 풀타임 2020-2023", "timeline", ["A사"]),
    ]

    def fake_verifier(claims_in, pairs, paragraphs):
        return [
            {
                "claim_ids": pairs[0],
                "verdict": "contradict",
                "rationale": "기간 겹침",
            }
        ]

    out = cross_check_claims({"claims": claims}, verifier=fake_verifier)
    assert out["cross_check"][0]["verdict"] == "contradict"
    assert out["cross_check"][0]["claim_ids"] == ("c0", "c1")
