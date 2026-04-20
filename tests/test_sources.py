"""build_source_excerpts / flag_evidence 유틸 단위 테스트."""

from hiremindset.graph.sources import build_source_excerpts, flag_evidence


def _state():
    return {
        "claims": [
            {
                "id": "c0",
                "text": "응답시간 40% 개선",
                "type": "achievement",
                "source_paragraph_id": "p0",
            },
            {
                "id": "c1",
                "text": "Spring Boot로 백엔드 개발",
                "type": "factual",
                # paragraph id가 없는 claim
            },
        ],
        "documents": {
            "paragraphs": [
                {"id": "p0", "text": "대규모 트래픽 테스트를 통해 응답시간을 개선했다"},
                {"id": "p1", "text": "팀 프로젝트에서 백엔드를 맡았다"},
            ]
        },
        "suspicion_flags": [
            {"id": "f0", "evidence": "수치 근거가 제시되지 않음"},
        ],
    }


def test_build_source_excerpts_joins_claim_and_paragraph():
    pq = {"target_claim_ids": ["c0"]}
    out = build_source_excerpts(_state(), pq)
    assert len(out) == 1
    assert out[0]["claim_text"] == "응답시간 40% 개선"
    assert out[0]["paragraph_id"] == "p0"
    assert out[0]["paragraph_text"].startswith("대규모")


def test_build_source_excerpts_handles_missing_paragraph():
    pq = {"target_claim_ids": ["c1"]}
    out = build_source_excerpts(_state(), pq)
    assert len(out) == 1
    assert "paragraph_id" not in out[0]
    assert "paragraph_text" not in out[0]


def test_build_source_excerpts_ignores_unknown_claim_ids():
    pq = {"target_claim_ids": ["c0", "nope"]}
    out = build_source_excerpts(_state(), pq)
    assert [e["claim_id"] for e in out] == ["c0"]


def test_build_source_excerpts_empty_when_no_targets():
    assert build_source_excerpts(_state(), {}) == []


def test_flag_evidence_lookup():
    assert flag_evidence(_state(), "f0") == "수치 근거가 제시되지 않음"
    assert flag_evidence(_state(), "missing") is None
    assert flag_evidence(_state(), None) is None
