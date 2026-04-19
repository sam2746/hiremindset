from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    resume_text: str = Field(..., min_length=1, description="Candidate resume or profile text")
    jd_text: str = Field(default="", description="Optional job description")


class RunResponse(BaseModel):
    opening_questions: list[str] = Field(default_factory=list)
    report_markdown: str = ""
    error: str | None = None
