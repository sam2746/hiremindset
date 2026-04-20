from hiremindset.graph.nodes.extract import (
    extract_claims_essay,
    extract_claims_resume,
)


def _fake_extractor(paragraphs, jd):
    # 앞 두 문단에 대해 가짜 claim 하나씩 반환 (id는 노드에서 부여)
    return [
        {
            "id": "",
            "text": f"fake:{p['id']}",
            "type": "factual",
            "source_paragraph_id": p["id"],
            "entities": [],
            "confidence": 0.5,
        }
        for p in paragraphs[:2]
    ]


def _state_with_paragraphs(kind, n):
    return {
        "documents": {
            "kind": kind,
            "raw": "",
            "paragraphs": [{"id": f"p{i}", "text": f"t{i}"} for i in range(n)],
        },
        "jd": "",
        "claims": [],
    }


def test_extract_resume_assigns_sequential_ids():
    out = extract_claims_resume(_state_with_paragraphs("resume", 3), extractor=_fake_extractor)
    assert [c["id"] for c in out["claims"]] == ["c0", "c1"]
    assert out["claims"][0]["source_paragraph_id"] == "p0"


def test_extract_essay_appends_after_existing_claims():
    state = _state_with_paragraphs("essay", 1)
    state["claims"] = [
        {
            "id": "c0",
            "text": "prev",
            "type": "factual",
            "source_paragraph_id": "p0",
            "entities": [],
            "confidence": 1.0,
        }
    ]
    out = extract_claims_essay(state, extractor=_fake_extractor)
    assert [c["id"] for c in out["claims"]] == ["c0", "c1"]
    assert out["claims"][-1]["text"] == "fake:p0"


def test_extract_with_empty_paragraphs_returns_existing():
    out = extract_claims_resume(
        {"documents": {"kind": "resume", "raw": "", "paragraphs": []}, "claims": []},
        extractor=_fake_extractor,
    )
    assert out["claims"] == []
