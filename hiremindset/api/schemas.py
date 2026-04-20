"""FastAPI <-> Streamlit 간 주고받는 세션 API 스키마.

그래프가 HITL interrupt 기반이라 한 번의 요청으로 끝나지 않는다.
- /session/start : 문서를 넣고 첫 질문이 나올 때까지 돌린 뒤 interrupt
- /session/resume: 면접관의 답변/액션으로 그래프 재개 → 다음 질문 또는 종료
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class SessionStartRequest(BaseModel):
    kind: Literal["resume", "essay"]
    text: str = Field(..., min_length=1, description="후보자 이력서/자기소개서 원문")
    jd: str = Field(default="", description="선택. 채용 JD 원문")


class SessionResumeRequest(BaseModel):
    thread_id: str = Field(..., min_length=1)
    action: Literal["accept", "fallback", "inject"]
    answer_text: str = Field(default="", description="현재 질문에 대한 후보자 답변")
    injected_question: str | None = Field(
        default=None,
        description="action='inject'일 때 면접관이 직접 적은 다음 질문",
    )


class PendingQuestion(BaseModel):
    question_id: str
    queue_id: str
    text: str
    profile: str | None = None
    target_flag_id: str | None = None
    asked_round: int = 0


class SessionStepResponse(BaseModel):
    thread_id: str
    done: bool
    pending_question: PendingQuestion | None = None
    summary: dict[str, Any] | None = None
    error: str | None = None
