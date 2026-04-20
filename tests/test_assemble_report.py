"""assemble_report: 3-strikes 감점 규칙 + markdown 생성."""

from hiremindset.graph.nodes.assemble_report import (
    BASE_SCORE,
    _compute_scoring,
    assemble_report,
)


def _flag(
    fid: str,
    severity: int = 3,
    resolved: bool = False,
    fallback_attempts: int = 0,
    category: str = "cliche_template",
    evidence: str = "",
):
    return {
        "id": fid,
        "claim_ids": ["c0"],
        "category": category,
        "severity": severity,
        "evidence": evidence,
        "strikes": 0,
        "fallback_attempts": fallback_attempts,
        "resolved": resolved,
    }


def test_all_resolved_gives_full_score():
    flags = [_flag("f0", resolved=True), _flag("f1", resolved=True)]
    sc = _compute_scoring(flags)
    assert sc["total"] == BASE_SCORE
    assert sc["deductions"] == []
    assert sc["severe_penalty_triggered"] is False


def test_unresolved_flag_deducts_severity_scaled():
    flags = [_flag("f0", severity=3, resolved=False)]
    sc = _compute_scoring(flags)
    # severity 3 → 3*3 = 9점 감점
    assert sc["total"] == BASE_SCORE - 9
    assert sc["deductions"][0]["points"] == 9
    assert sc["severe_penalty_triggered"] is False


def test_three_strikes_triggers_severe_penalty():
    flags = [_flag("f0", severity=3, fallback_attempts=2, resolved=False)]
    sc = _compute_scoring(flags)
    # base 9점 + 강감점 severity*5 = 15점 → 총 24점 감점
    assert sc["severe_penalty_triggered"] is True
    total_deduction = sum(d["points"] for d in sc["deductions"])
    assert total_deduction == 9 + 15
    assert sc["total"] == BASE_SCORE - total_deduction


def test_score_clamped_at_zero():
    # 아주 큰 severity flag 여럿 → total이 음수가 되지 않아야 함
    flags = [
        _flag(f"f{i}", severity=5, fallback_attempts=3, resolved=False)
        for i in range(5)
    ]
    sc = _compute_scoring(flags)
    assert sc["total"] == 0


def test_assemble_report_populates_state():
    state = {
        "documents": {"kind": "resume", "raw": "", "paragraphs": []},
        "strategy": {"round": 4, "max_rounds": 20},
        "claims": [{"id": "c0", "text": "x", "type": "factual"}],
        "suspicion_flags": [
            _flag(
                "f0",
                severity=4,
                fallback_attempts=2,
                evidence="수치 근거 없음",
            )
        ],
        "probing_questions": [
            {
                "id": "pq0",
                "queue_id": "q0",
                "text": "응답시간 40% 어떻게 측정했나요?",
                "asked_round": 1,
                "target_flag_id": "f0",
                "profile": "numeric",
            }
        ],
        "decision_log": [
            {
                "round": 1,
                "question_id": "pq0",
                "question": "응답시간 40% 어떻게 측정했나요?",
                "answer_text": "기억이 안 나요",
                "action": "fallback",
                "why": "답변 부족 — 폴백 시드",
                "fallback_used": True,
                "flag_id": "f0",
                "profile": "numeric",
                "ai_specificity": 0.1,
                "ai_consistency": 0.9,
                "ai_epistemic": 0.8,
                "ai_hedge": True,
                "ai_suggest": ["drill"],
            }
        ],
    }
    out = assemble_report(state)
    assert "scoring" in out
    assert "report_markdown" in out
    md = out["report_markdown"]
    assert "# HireMindset 인터뷰 리포트" in md
    assert "종합 점수" in md
    assert "3-strikes" in md
    assert "라운드별 결정 로그" in md
    assert "pq0" not in md  # 질문 id 대신 텍스트가 보여야 한다
    assert "응답시간 40%" in md
    assert "fallback" in md
