from hiremindset.graph.nodes.ingest import ingest_normalize
from hiremindset.graph.routers import route_doc_profile


def test_ingest_splits_paragraphs_on_any_newline():
    text = "첫 줄\n두 번째 줄\n\n네 번째 줄"
    out = ingest_normalize(
        {"documents": {"kind": "essay", "raw": text, "paragraphs": []}}
    )
    paragraphs = out["documents"]["paragraphs"]
    assert [p["text"] for p in paragraphs] == ["첫 줄", "두 번째 줄", "네 번째 줄"]
    assert [p["id"] for p in paragraphs] == ["p0", "p1", "p2"]


def test_ingest_defaults_meta_and_strategy_with_hitl_on():
    out = ingest_normalize({"documents": {"kind": "resume", "raw": "한 줄", "paragraphs": []}})
    meta = out["meta"]
    strategy = out["strategy"]
    assert meta["allow_hitl"] is True
    assert meta["max_rounds"] >= 1
    assert strategy["round"] == 0
    assert strategy["fallbacks_used"] == 0
    assert strategy["max_rounds"] == meta["max_rounds"]


def test_ingest_accepts_legacy_resume_text_input():
    out = ingest_normalize({"resume_text": "백엔드 3년\n스택: Python", "jd_text": "Python"})
    assert out["documents"]["kind"] == "resume"
    assert [p["text"] for p in out["documents"]["paragraphs"]] == ["백엔드 3년", "스택: Python"]
    assert out["jd"] == "Python"


def test_route_doc_profile_returns_kind_or_default_resume():
    assert route_doc_profile({"documents": {"kind": "essay", "raw": "", "paragraphs": []}}) == "essay"
    assert route_doc_profile({"documents": {"kind": "resume", "raw": "", "paragraphs": []}}) == "resume"
    assert route_doc_profile({}) == "resume"
