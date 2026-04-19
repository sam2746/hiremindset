"""Graph state schema (expand as nodes are implemented)."""

from typing import TypedDict


class GraphState(TypedDict, total=False):
    """Minimal state for scaffolding; extend for claims, flags, probing questions."""

    resume_text: str
    jd_text: str
    # Placeholders for upcoming nodes:
    opening_questions: list[str]
    report_markdown: str
    error: str
