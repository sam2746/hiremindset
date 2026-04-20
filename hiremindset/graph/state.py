"""
Graph state schema for the cross-check interview agent.

Uses TypedDict(total=False) so nodes can update only the fields they own.
Nested TypedDicts describe the shape of list items / compound objects.

Design notes:
- HITL is wired at `collect_answer` (turns[].role == "human" via interrupt),
  not at the report stage.
- Scoring enforces the "3-strikes → severe penalty" rule; strikes live on
  each suspicion_flags[] item.
"""

from typing import Literal, NotRequired, TypedDict

# ---------- shared literals ----------

DocKind = Literal["resume", "essay"] #리터럴은 자바의 enum 같은 느낌
InterviewMode = Literal["technical", "behavioral", "mixed"]

FlagCategory = Literal[
    "timeline_conflict",        # 날짜 모순
    "metric_unsupported",       # 수치 근거 없음
    "cliche_template",          # 클리셰·STAR 외운 티
    "depth_collapse",           # follow-up에서 무너지는 깊이
    "suspected_exaggeration",   # 스코프·역할 과장
    "technical_probe_needed",   # 왜?·대안·트레이드오프 누락
    "inauthentic_company_ref",  # 다른 회사명 혼용 등 성의 문제
]

ProbeProfile = Literal["numeric", "mechanism", "story", "consistency", "context"]
ProbeSource = Literal["plan", "fallback", "hitl"]
ClaimType = Literal["factual", "achievement", "timeline", "value"]
TurnRole = Literal["simulator", "human"]
EvalSuggestion = Literal["mechanism", "drill", "done"]
CrossVerdict = Literal["contradict", "weak", "ok"]
ControlSignal = Literal["continue", "fallback", "hitl", "done"]


# ---------- inputs / documents ----------

class Paragraph(TypedDict):
    id: str
    text: str
    section: NotRequired[str]


class Documents(TypedDict):
    kind: DocKind
    raw: str
    paragraphs: list[Paragraph]


class Meta(TypedDict, total=False):
    interview_mode: InterviewMode
    max_rounds: int
    max_fallbacks: int
    allow_hitl: bool


# ---------- claims & verification ----------

class Claim(TypedDict):
    id: str
    text: str
    type: ClaimType
    source_paragraph_id: NotRequired[str]
    entities: NotRequired[list[str]]
    ts_range: NotRequired[tuple[str, str]]
    confidence: NotRequired[float]


class CrossCheckPair(TypedDict):
    claim_ids: tuple[str, str]
    verdict: CrossVerdict
    rationale: str


class SuspicionFlag(TypedDict):
    id: str
    claim_ids: list[str]
    category: FlagCategory #플래그 리터럴에서 갖고 옴
    severity: int  # 1..5
    evidence: str
    strikes: int
    fallback_attempts: int
    resolved: bool


# ---------- probing (plan vs emit split) ---------- 

class ProbeItem(TypedDict):
    """계획 단위. emit_question에서 문장화되어 probing_questions로 이동."""

    id: str
    target_claim_ids: list[str]
    intent: str
    expected_signal: str
    profile: ProbeProfile
    attempts: int
    target_flag_id: NotRequired[str]
    priority: NotRequired[int]        # 우선순위 큐 정렬 키 (클수록 먼저)
    source: NotRequired[ProbeSource]  # plan | fallback | hitl
    pre_generated_text: NotRequired[str]  # 주입/재생성된 질문 문장 (emit_question이 LLM 생략)


class ProbingQuestion(TypedDict):
    id: str
    queue_id: str
    text: str
    asked_round: int
    target_flag_id: NotRequired[str]
    target_claim_ids: NotRequired[list[str]]
    profile: NotRequired[ProbeProfile]


# ---------- turns & evaluation ----------

class Turn(TypedDict):
    q_id: str
    role: TurnRole
    answer_text: str


class AnswerEval(TypedDict):
    q_id: str
    specificity: float        # 0.0 ~ 1.0
    consistency: float        # 0.0 ~ 1.0 (이전 claim/턴과의 일치도)
    epistemic: float          # 0.0 ~ 1.0 (정직한 '모른다'인지)
    refusal_or_hedge: bool
    suggest: list[EvalSuggestion]


# ---------- control / strategy / scoring ----------

class DecisionLogEntry(TypedDict):
    round: int
    why: str
    fallback_used: bool
    chosen_probe_id: NotRequired[str]


class Strategy(TypedDict, total=False):
    round: int
    max_rounds: int
    fallbacks_used: int
    max_fallbacks: int
    last_profile: ProbeProfile


class Deduction(TypedDict):
    flag_id: str
    points: int
    reason: str


class Scoring(TypedDict):
    base: int
    deductions: list[Deduction]
    total: int
    severe_penalty_triggered: bool


# ---------- graph state ----------

class GraphState(TypedDict, total=False):
    """Canonical state passed between nodes. Fields are optional by design."""

    # Legacy/stub inputs — 현재 스텁 그래프와 테스트 호환용
    resume_text: str
    jd_text: str
    opening_questions: list[str]

    # Canonical inputs
    documents: Documents
    jd: str
    meta: Meta

    # Extraction & verification
    claims: list[Claim]
    cross_check: list[CrossCheckPair]
    suspicion_flags: list[SuspicionFlag]

    # Probing pipeline
    probe_queue: list[ProbeItem]
    probing_questions: list[ProbingQuestion]

    # Dialogue & evaluation
    turns: list[Turn]
    answer_eval: list[AnswerEval]

    # Control / bookkeeping
    strategy: Strategy
    control: ControlSignal
    decision_log: list[DecisionLogEntry]

    # Output
    scoring: Scoring
    report_markdown: str
    error: str
