from hiremindset.graph.builder import build_graph


def test_stub_graph_invokes():
    g = build_graph()
    out = g.invoke({"resume_text": "hello", "jd_text": "world"})
    assert "report_markdown" in out
    assert "opening_questions" in out
    assert len(out["opening_questions"]) >= 1
    assert "Stub report" in out["report_markdown"]
