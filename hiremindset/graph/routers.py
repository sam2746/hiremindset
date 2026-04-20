"""
그래프 조건부 엣지에서 쓰는 순수 함수 라우터.
"""

from __future__ import annotations

from hiremindset.graph.state import DocKind, GraphState


def route_doc_profile(state: GraphState) -> DocKind:
    """사용자가 명시한 문서 종류를 다음 추출 노드의 키로 돌려준다.

    정책: 사용자는 resume / essay 중 반드시 하나를 선택한다.
    값이 없으면 보수적으로 resume으로 본다.
    """
    documents = state.get("documents")
    if documents and documents.get("kind"):
        return documents["kind"]
    return "resume"
