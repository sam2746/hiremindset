"""LangGraph assembly. Stub pipeline: echo-style report until nodes are implemented."""

from langgraph.graph import END, START, StateGraph

from hiremindset.graph.state import GraphState


def _stub_report(state: GraphState) -> GraphState:
    resume = (state.get("resume_text") or "").strip()
    jd = (state.get("jd_text") or "").strip()
    md = (
        "## Stub report\n\n"
        f"- Resume length: {len(resume)} chars\n"
        f"- JD length: {len(jd)} chars\n\n"
        "Replace this node with opening questions → simulator → extract → … per plan."
    )
    return {"report_markdown": md, "opening_questions": ["(stub) Tell me about your latest project."]}


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("stub_report", _stub_report)
    g.add_edge(START, "stub_report")
    g.add_edge("stub_report", END)
    return g.compile()
