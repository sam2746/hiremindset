"""FastAPI <-> Streamlit 간 주고받는 세션 API 스키마.

Phase 1부터 한 라운드는 두 번의 interrupt로 나뉜다.
- phase="collect_answer" : 후보자 답변 텍스트만 수집
- phase="decide_action"  : AI 평가를 본 뒤 면접관이 accept/fallback/drill/pass/inject 결정

/session/resume 요청에는 클라이언트가 어느 phase로 응답하는지 싣는다.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

Phase = Literal["collect_answer", "decide_action"]


class SessionStartRequest(BaseModel):
    kind: Literal["resume", "essay"]
    text: str = Field(..., min_length=1, description="후보자 이력서/자기소개서 원문")
    jd: str = Field(default="", description="선택. 채용 JD 원문")
    max_rounds: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="해당 세션의 최대 라운드 수. 미지정 시 그래프 기본값 사용.",
    )


class SessionResumeRequest(BaseModel):
    thread_id: str = Field(..., min_length=1)
    phase: Phase = Field(
        default="collect_answer",
        description="직전 pending_question의 phase를 그대로 넣는다.",
    )
    answer_text: str = Field(
        default="",
        description="phase=collect_answer일 때 후보자 답변.",
    )
    action: (
        Literal["accept", "fallback", "drill", "pass", "inject"] | None
    ) = Field(
        default=None,
        description="phase=decide_action일 때 면접관 결정.",
    )
    injected_question: str | None = Field(
        default=None,
        description="action='inject'일 때 면접관이 직접 적은 다음 질문",
    )


class AnswerEvalSnapshot(BaseModel):
    specificity: float
    consistency: float
    epistemic: float
    refusal_or_hedge: bool
    suggest: list[str] = Field(default_factory=list)


class SourceExcerpt(BaseModel):
    """질문이 겨냥한 claim과 그 원문 문단 스니펫."""

    claim_id: str
    claim_text: str
    paragraph_id: str | None = None
    paragraph_text: str | None = None


class PendingQuestion(BaseModel):
    phase: Phase
    question_id: str
    queue_id: str
    text: str
    profile: str | None = None
    target_flag_id: str | None = None
    target_claim_ids: list[str] = Field(default_factory=list)
    asked_round: int = 0

    source_excerpts: list[SourceExcerpt] = Field(default_factory=list)
    flag_evidence: str | None = None

    # phase=decide_action일 때만 채워짐: UI가 평가 카드 + 답변 원문을 보여주는 데 사용.
    answer_text: str | None = None
    ai_eval: AnswerEvalSnapshot | None = None


class SessionStepResponse(BaseModel):
    thread_id: str
    done: bool
    pending_question: PendingQuestion | None = None
    summary: dict[str, Any] | None = None
    error: str | None = None
