"""
assemble_report: 세션 종료 직전에 실행되는 순수 계산 노드.

책임:
- 3-strikes 규칙 기반으로 ``Scoring``을 계산해 ``state.scoring``에 저장
- 감점 내역·결정 로그·플래그 현황을 요약한 markdown 리포트를
  ``state.report_markdown``에 저장

점수 규칙(간결 MVP):
- base = 100
- 미해결 flag 하나당 severity 비례 감점 (severity × 3점, min 3, max 15)
- fallback_attempts ≥ 2 인데도 미해결(= 3-strikes) → 추가 강감점 severity × 5점
- 모든 flag가 resolved면 추가 감점 없음
- total = max(0, base - sum(deductions))
- severe_penalty_triggered 는 3-strikes 항목이 1개 이상이면 True
"""

from __future__ import annotations

from typing import Any

from hiremindset.graph.state import (
    DecisionLogEntry,
    Deduction,
    GraphState,
    Scoring,
    SuspicionFlag,
)

BASE_SCORE = 100
STRIKE_THRESHOLD = 2  # fallback_attempts가 이 이상이면 3-strikes 로 본다


# ---------- scoring ----------

def _compute_scoring(flags: list[SuspicionFlag]) -> Scoring:
    deductions: list[Deduction] = []
    severe = False
    for f in flags:
        if f.get("resolved"):
            continue
        severity = max(1, min(5, int(f.get("severity", 1))))
        base_pts = max(3, min(15, severity * 3))
        reason = f"미해결 flag: {f.get('category')} (severity={severity})"
        deductions.append(
            {"flag_id": f["id"], "points": base_pts, "reason": reason}
        )
        if int(f.get("fallback_attempts", 0)) >= STRIKE_THRESHOLD:
            severe = True
            strike_pts = severity * 5
            deductions.append(
                {
                    "flag_id": f["id"],
                    "points": strike_pts,
                    "reason": (
                        f"3-strikes 강감점: fallback {f.get('fallback_attempts')}회 "
                        f"뒤에도 해소되지 않음"
                    ),
                }
            )

    total = max(0, BASE_SCORE - sum(d["points"] for d in deductions))
    return {
        "base": BASE_SCORE,
        "deductions": deductions,
        "total": total,
        "severe_penalty_triggered": severe,
    }


# ---------- markdown ----------

def _fmt_pct(n: int, d: int) -> str:
    if d <= 0:
        return "-"
    return f"{n}/{d} ({n * 100 // d}%)"


def _render_decision_row(entry: DecisionLogEntry) -> str:
    action = entry.get("action", "?")
    q = (entry.get("question") or "").replace("|", "\\|").replace("\n", " ")
    if len(q) > 80:
        q = q[:77] + "…"
    a = (entry.get("answer_text") or "").replace("|", "\\|").replace("\n", " ")
    if len(a) > 80:
        a = a[:77] + "…"
    spec = entry.get("ai_specificity")
    cons = entry.get("ai_consistency")
    epi = entry.get("ai_epistemic")
    hedge = entry.get("ai_hedge")
    scores = (
        f"S{spec:.2f}/C{cons:.2f}/E{epi:.2f}"
        if isinstance(spec, float)
        else "-"
    )
    hedge_str = (
        ("있음" if hedge else "없음") if isinstance(hedge, bool) else "-"
    )
    return (
        f"| R{entry.get('round', '?')} "
        f"| {action} "
        f"| {entry.get('flag_id') or '-'} "
        f"| {q} "
        f"| {a} "
        f"| {scores} "
        f"| {hedge_str} |"
    )


def _render_markdown(state: GraphState, scoring: Scoring) -> str:
    flags = list(state.get("suspicion_flags") or [])
    resolved = [f for f in flags if f.get("resolved")]
    unresolved = [f for f in flags if not f.get("resolved")]
    strikes = [
        f
        for f in unresolved
        if int(f.get("fallback_attempts", 0)) >= STRIKE_THRESHOLD
    ]

    strategy = state.get("strategy") or {}
    docs = state.get("documents") or {}

    lines: list[str] = []
    lines.append("# HireMindset 인터뷰 리포트")
    lines.append("")
    lines.append(
        f"- 문서 종류: `{docs.get('kind', '?')}`  "
        f"· 라운드: {strategy.get('round', 0)} / {strategy.get('max_rounds', '?')}"
    )
    lines.append(
        f"- Claims: {len(state.get('claims') or [])}  "
        f"· Flags: {len(flags)} "
        f"(resolved {_fmt_pct(len(resolved), len(flags))})  "
        f"· 3-strikes: {len(strikes)}"
    )
    lines.append("")

    lines.append("## 종합 점수")
    lines.append(
        f"- **{scoring['total']} / {scoring['base']}**"
        + (" · ⚠️ 강감점 발동" if scoring["severe_penalty_triggered"] else "")
    )
    if scoring["deductions"]:
        lines.append("")
        lines.append("| flag | 감점 | 사유 |")
        lines.append("|------|------:|------|")
        for d in scoring["deductions"]:
            reason = d["reason"].replace("|", "\\|")
            lines.append(f"| {d['flag_id']} | {d['points']} | {reason} |")
    else:
        lines.append("- 감점 없음")
    lines.append("")

    if strikes:
        lines.append("## 3-strikes 미해결 플래그")
        for f in strikes:
            lines.append(
                f"- `{f['id']}` · {f.get('category')} "
                f"· severity {f.get('severity')} "
                f"· fallback {f.get('fallback_attempts')}회"
            )
            evi = (f.get("evidence") or "").strip()
            if evi:
                lines.append(f"  - 근거: {evi}")
        lines.append("")

    if unresolved:
        lines.append("## 미해결 플래그")
        lines.append("| id | category | severity | fallback | 근거 |")
        lines.append("|----|----------|---------:|---------:|------|")
        for f in unresolved:
            evi = (f.get("evidence") or "").replace("|", "\\|").replace("\n", " ")
            if len(evi) > 60:
                evi = evi[:57] + "…"
            lines.append(
                f"| {f['id']} | {f.get('category')} | {f.get('severity')} "
                f"| {f.get('fallback_attempts', 0)} | {evi} |"
            )
        lines.append("")

    decision_log = list(state.get("decision_log") or [])
    if decision_log:
        lines.append("## 라운드별 결정 로그")
        lines.append(
            "| R | action | flag | 질문 | 답변 | S/C/E | 회피 |"
        )
        lines.append(
            "|---:|--------|------|------|------|-------|------|"
        )
        for entry in decision_log:
            lines.append(_render_decision_row(entry))
        lines.append("")

    pqs = list(state.get("probing_questions") or [])
    if pqs:
        lines.append("## 질문 목록 (순서대로)")
        for pq in pqs:
            flag_tag = (
                f" · flag `{pq.get('target_flag_id')}`"
                if pq.get("target_flag_id")
                else ""
            )
            lines.append(
                f"- R{pq.get('asked_round', '?')} "
                f"[{pq.get('profile', '-')}]{flag_tag}  \n"
                f"  {pq.get('text', '')}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------- node entrypoint ----------

def assemble_report(state: GraphState) -> dict[str, Any]:
    flags = list(state.get("suspicion_flags") or [])
    scoring = _compute_scoring(flags)
    markdown = _render_markdown(state, scoring)
    out: dict[str, Any] = {"scoring": scoring, "report_markdown": markdown}
    if state.get("skip_evaluate_decide"):
        out["skip_evaluate_decide"] = False
    return out
