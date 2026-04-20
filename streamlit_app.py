"""
Streamlit UI.

원칙: 그래프를 import하지 않고 FastAPI(/session/start, /session/resume)만 호출한다.

Phase 1 이후 한 라운드는 두 개의 HITL 단계로 나뉜다.
  1) phase="collect_answer" : 후보자 답변 제출 — (선택) 답변+즉시 통과로 AI 평가 생략
  2) phase="decide_action"  : AI 평가 카드를 본 뒤 면접관이 결정
     - accept   : 통과 (해당 flag resolved)
     - drill    : AI가 심층 기술 질문을 생성해 큐에 투입
     - fallback : AI가 주변 디테일 폴백 질문을 생성
     - pass     : 답변이 추궁할 가치 없음 — 다음으로
     - skip     : 이 추천 질문은 쓰지 않음 — 다음으로
     - inject   : 면접관이 직접 다음 질문 주입
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

DEFAULT_API = os.environ.get("API_PUBLIC_URL", "http://127.0.0.1:8000").rstrip("/")
GA_ID = os.environ.get("GA_MEASUREMENT_ID", "").strip()

ACTIONS = ("accept", "drill", "fallback", "pass", "skip", "inject")
ACTION_LABELS: dict[str, str] = {
    "accept": "✅ 통과",
    "drill": "🔬 심층 질문 생성하기",
    "fallback": "🩹 폴백 (주변 디테일)",
    "pass": "⏭️ 답변 스킵 (추궁 불가)",
    "skip": "⏭️ 추천 질문 건너뛰기",
    "inject": "✍️ 직접 질문 주입",
}
ACTION_HELP = (
    "통과 — 답변이 충분, 다음 질문으로 (해당 flag resolved)\n"
    "심층 질문 생성하기 — 답변을 기반으로 신입에게 요구될 만한 기술적 심층 질문을 AI가 생성해 큐에 투입\n"
    "폴백 — 답이 부족해 보일 때 AI가 주변 디테일을 캐묻는 질문을 생성\n"
    "답변 스킵 — 답변이 너무 빈약·무의미해 더 물어볼 가치가 없음. flag는 그대로, 다음 큐로\n"
    "추천 질문 건너뛰기 — 답변은 괜찮지만 이 시스템 추천 질문 라인은 쓰지 않고 다음으로\n"
    "직접 질문 주입 — 면접관이 직접 이어갈 질문을 써서 큐 최상단에 투입"
)

# AI 평가 지표 라벨 한글화 (내부 필드명은 그대로 유지하고 표시만 바꿈).
METRIC_LABELS: dict[str, str] = {
    "specificity": "구체성",
    "consistency": "정합성",
    "epistemic": "정직성",
    "hedge": "회피",
}

PROFILE_BADGE = {
    "context": "🧭",
    "numeric": "🔢",
    "mechanism": "⚙️",
    "story": "📖",
    "consistency": "⚖️",
}


# ---------- session state ----------

def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("thread_id", None)
    ss.setdefault("done", False)
    ss.setdefault("pending", None)
    ss.setdefault("summary", None)
    ss.setdefault("history", [])       # [{role, text, meta}]
    ss.setdefault("last_error", None)
    ss.setdefault("kind", "resume")
    ss.setdefault("resume_text", "")
    ss.setdefault("jd_text", "")
    ss.setdefault("max_rounds", 20)
    ss.setdefault("action", "accept")
    ss.setdefault("_decide_for_qid", None)


def _reset_session() -> None:
    ss = st.session_state
    ss.thread_id = None
    ss.done = False
    ss.pending = None
    ss.summary = None
    ss.history = []
    ss.last_error = None


# ---------- API ----------

def _post(api_base: str, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        r = httpx.post(f"{api_base}{path}", json=payload, timeout=180.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.session_state.last_error = f"{path} {e.response.status_code}: {e.response.text}"
    except Exception as e:  # noqa: BLE001
        st.session_state.last_error = f"{path} 호출 실패: {e}"
    return None


def _apply_step(data: dict[str, Any]) -> None:
    ss = st.session_state
    ss.thread_id = data.get("thread_id") or ss.thread_id
    ss.summary = data.get("summary")
    if data.get("done"):
        ss.done = True
        ss.pending = None
    else:
        ss.done = False
        ss.pending = data.get("pending_question")


def _start_session(api_base: str) -> None:
    ss = st.session_state
    if not ss.resume_text.strip():
        st.warning("이력서/자소서 원문을 입력해 주세요.")
        return
    ss.last_error = None
    data = _post(
        api_base,
        "/session/start",
        {
            "kind": ss.kind,
            "text": ss.resume_text,
            "jd": ss.jd_text,
            "max_rounds": int(ss.max_rounds),
        },
    )
    if not data:
        return
    _apply_step(data)
    if ss.pending and ss.pending.get("phase") == "collect_answer":
        ss.history.append(
            {"role": "simulator", "text": ss.pending["text"], "meta": ss.pending}
        )


def _submit_answer(
    api_base: str, answer: str, *, immediate_accept: bool = False
) -> None:
    ss = st.session_state
    if not ss.thread_id or not ss.pending:
        return
    ss.last_error = None
    meta: dict[str, Any] = {}
    if immediate_accept:
        meta["action"] = "accept"
        meta["immediate_accept"] = True
    ss.history.append({"role": "human", "text": answer, "meta": meta})
    payload: dict[str, Any] = {
        "thread_id": ss.thread_id,
        "phase": "collect_answer",
        "answer_text": answer,
    }
    if immediate_accept:
        payload["immediate_action"] = "accept"
    data = _post(api_base, "/session/resume", payload)
    if not data:
        return
    _apply_step(data)
    # 즉시 통과(immediate accept)면 evaluate/decide를 건너뛰고 곧바로 다음 collect_answer로
    # 이어지므로, 여기서도 새 질문을 히스토리에 올려야 한다 (decide_action 경로와 동일).
    # decide_action 단계로 가는 경우엔 아래 조건이 거짓이라 시뮬레이터 턴을 추가하지 않음.
    if ss.pending and ss.pending.get("phase") == "collect_answer":
        ss.history.append(
            {"role": "simulator", "text": ss.pending["text"], "meta": ss.pending}
        )


def _submit_decision(api_base: str, action: str, injected: str) -> None:
    ss = st.session_state
    if not ss.thread_id or not ss.pending:
        return
    ss.last_error = None
    # 결정 자체를 히스토리의 직전 human turn 메타에 덧붙여 기록
    if ss.history and ss.history[-1]["role"] == "human":
        ss.history[-1]["meta"] = {**(ss.history[-1].get("meta") or {}), "action": action}
    payload = {
        "thread_id": ss.thread_id,
        "phase": "decide_action",
        "action": action,
        "injected_question": injected if action == "inject" else None,
    }
    data = _post(api_base, "/session/resume", payload)
    if not data:
        return
    _apply_step(data)
    if ss.pending and ss.pending.get("phase") == "collect_answer":
        ss.history.append(
            {"role": "simulator", "text": ss.pending["text"], "meta": ss.pending}
        )


# ---------- rendering ----------

def _inject_ga4(measurement_id: str) -> None:
    if not measurement_id:
        return
    st.components.v1.html(
        f"""
<script async src="https://www.googletagmanager.com/gtag/js?id={measurement_id}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{measurement_id}');
</script>
        """,
        height=0,
    )


def _render_setup(api_base: str) -> None:
    ss = st.session_state
    st.subheader("1. 후보자 문서 입력")
    ss.kind = st.radio(
        "문서 종류",
        options=["resume", "essay"],
        horizontal=True,
        index=0 if ss.kind == "resume" else 1,
    )
    ss.resume_text = st.text_area(
        "이력서 / 자기소개서 원문",
        value=ss.resume_text,
        height=220,
        placeholder="후보자가 제출한 원문을 그대로 붙여 넣으세요. 줄바꿈 단위로 문단이 구분됩니다.",
    )
    ss.jd_text = st.text_area(
        "JD (선택)", value=ss.jd_text, height=100, placeholder="채용 공고/JD 원문"
    )
    if st.button("세션 시작", type="primary"):
        with st.spinner("문서를 분석하고 첫 질문을 생성하는 중…"):
            _start_session(api_base)
        st.rerun()


def _chip(label: str) -> str:
    return f"<span style='background:#eef;padding:2px 8px;border-radius:10px;font-size:0.8em;margin-right:4px'>{label}</span>"


def _render_history() -> None:
    for turn in st.session_state.history:
        role = turn["role"]
        meta = turn.get("meta") or {}
        if role == "simulator":
            with st.chat_message("assistant"):
                profile = meta.get("profile") or ""
                tag = meta.get("target_flag_id")
                rnd = meta.get("asked_round")
                chips = []
                if profile:
                    chips.append(_chip(f"{PROFILE_BADGE.get(profile, '')} {profile}"))
                if tag:
                    chips.append(_chip(f"flag {tag}"))
                if rnd is not None:
                    chips.append(_chip(f"R{rnd}"))
                if chips:
                    st.markdown("".join(chips), unsafe_allow_html=True)
                st.write(turn["text"])
        else:
            with st.chat_message("user"):
                action = (meta or {}).get("action")
                if action:
                    label = ACTION_LABELS.get(action, action)
                    st.markdown(_chip(label), unsafe_allow_html=True)
                st.write(turn["text"] or "_(빈 답변)_")


def _render_source_excerpts(pq: dict[str, Any]) -> None:
    excerpts = pq.get("source_excerpts") or []
    evidence = pq.get("flag_evidence")
    if not excerpts and not evidence:
        return
    with st.expander("이 질문의 출처 원문", expanded=True):
        for e in excerpts:
            st.markdown(
                f"**claim `{e.get('claim_id')}`**  \n> {e.get('claim_text') or ''}"
            )
            p_text = e.get("paragraph_text")
            if p_text:
                st.caption(f"문단 `{e.get('paragraph_id')}`")
                st.text(p_text)
        if evidence:
            st.markdown("**flag 근거**")
            st.warning(evidence)


def _render_collect_answer(api_base: str) -> None:
    ss = st.session_state
    pq = ss.pending or {}

    st.divider()
    st.markdown(
        f"**현재 질문** · R{pq.get('asked_round')} · "
        f"profile `{pq.get('profile')}` · "
        f"flag `{pq.get('target_flag_id') or '-'}`"
    )
    _render_source_excerpts(pq)

    turn_idx = len(ss.history)
    answer = st.text_area(
        "후보자 답변",
        key=f"answer_{turn_idx}",
        height=140,
        placeholder="면접관이 후보자의 답변을 대신 입력/요약해 주세요.",
    )
    b1, b2 = st.columns(2)
    with b1:
        if st.button("답변 제출 (AI 평가)", type="primary", use_container_width=True):
            with st.spinner("AI 평가 중…"):
                _submit_answer(api_base, answer, immediate_accept=False)
            st.rerun()
    with b2:
        if st.button("답변 + 통과 (평가 생략)", type="secondary", use_container_width=True):
            with st.spinner("반영 중…"):
                _submit_answer(api_base, answer, immediate_accept=True)
            st.rerun()


def _fmt_score(v: Any) -> str:
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "-"


def _render_decide_action(api_base: str) -> None:
    ss = st.session_state
    pq = ss.pending or {}
    ai = pq.get("ai_eval") or {}

    st.divider()
    st.markdown(
        f"**AI 평가** · R{pq.get('asked_round')} · profile `{pq.get('profile')}`"
    )
    _render_source_excerpts(pq)
    st.caption("질문")
    st.info(pq.get("text") or "")
    st.caption("후보자 답변")
    st.success(pq.get("answer_text") or "_(빈 답변)_")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(METRIC_LABELS["specificity"], _fmt_score(ai.get("specificity")))
    c2.metric(METRIC_LABELS["consistency"], _fmt_score(ai.get("consistency")))
    c3.metric(METRIC_LABELS["epistemic"], _fmt_score(ai.get("epistemic")))
    c4.metric(
        METRIC_LABELS["hedge"],
        "있음" if ai.get("refusal_or_hedge") else "없음",
    )
    suggest = ai.get("suggest") or []
    if suggest:
        st.markdown(
            "**AI 제안**: " + " ".join(_chip(s) for s in suggest),
            unsafe_allow_html=True,
        )

    # AI가 drill을 제안하면 기본 선택을 drill로 유도한다 (질문이 바뀔 때만).
    qid = pq.get("question_id") or ""
    last_qid = ss.get("_decide_for_qid")
    if last_qid != qid:
        ss.action = "drill" if "drill" in suggest else "accept"
        ss._decide_for_qid = qid

    recommended = "drill" if "drill" in suggest else None
    if recommended:
        st.info(f"AI 추천: **{ACTION_LABELS[recommended]}**")

    def _fmt_action(a: str) -> str:
        label = ACTION_LABELS.get(a, a)
        if recommended and a == recommended:
            return f"{label}  ⭐"
        return label

    ss.action = st.radio(
        "면접관 결정",
        options=ACTIONS,
        format_func=_fmt_action,
        horizontal=True,
        index=ACTIONS.index(ss.action if ss.action in ACTIONS else "accept"),
        help=ACTION_HELP,
        key=f"decide_action_{qid}",
    )
    injected = ""
    if ss.action == "inject":
        injected = st.text_area(
            "주입할 다음 질문",
            key=f"inject_{qid}",
            height=80,
            placeholder="면접관이 직접 이어갈 질문을 입력하세요. 이 질문이 큐 최상단에 들어갑니다.",
        )

    if st.button("결정 제출", type="primary"):
        if ss.action == "inject" and not injected.strip():
            st.warning("직접 질문 주입에는 다음 질문 내용이 필요합니다.")
        else:
            with st.spinner("그래프 재개 중…"):
                _submit_decision(api_base, ss.action, injected)
            st.rerun()


def _render_pending(api_base: str) -> None:
    phase = (st.session_state.pending or {}).get("phase")
    if phase == "decide_action":
        _render_decide_action(api_base)
    else:
        _render_collect_answer(api_base)


def _render_summary() -> None:
    summary = st.session_state.summary or {}
    strategy = summary.get("strategy") or {}
    scoring = summary.get("scoring") or {}
    report_md = summary.get("report_markdown") or ""

    st.success(
        f"세션 종료 · round {strategy.get('round', '?')} / "
        f"max {strategy.get('max_rounds', '?')}"
    )

    c1, c2, c3, c4 = st.columns(4)
    total = scoring.get("total")
    base = scoring.get("base", 100)
    c1.metric(
        "종합 점수",
        f"{total} / {base}" if total is not None else "-",
        delta=(
            "⚠️ 강감점" if scoring.get("severe_penalty_triggered") else None
        ),
        delta_color="inverse",
    )
    flags = summary.get("suspicion_flags") or []
    resolved = sum(1 for f in flags if f.get("resolved"))
    c2.metric("Flag 해소", f"{resolved} / {len(flags)}")
    c3.metric("질문 수", len(summary.get("probing_questions") or []))
    c4.metric("답변 수", len(summary.get("answer_eval") or []))

    if report_md:
        st.markdown("### 📝 리포트")
        st.markdown(report_md)
        st.download_button(
            "리포트 Markdown 다운로드",
            data=report_md,
            file_name="hiremindset_report.md",
            mime="text/markdown",
        )

    decisions = summary.get("decision_log") or []
    if decisions:
        with st.expander("결정 로그 (표)", expanded=False):
            st.dataframe(decisions, use_container_width=True)


def _render_dashboard() -> None:
    summary = st.session_state.summary or {}
    if not summary:
        st.caption("세션이 시작되면 분석 결과가 여기에 누적됩니다.")
        return

    strategy = summary.get("strategy") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Round", strategy.get("round", 0))
    c2.metric("Claims", len(summary.get("claims") or []))
    c3.metric("Flags", len(summary.get("suspicion_flags") or []))
    c4.metric("Questions", len(summary.get("probing_questions") or []))

    with st.expander("Suspicion flags", expanded=True):
        flags = summary.get("suspicion_flags") or []
        if flags:
            st.dataframe(flags, use_container_width=True)
        else:
            st.caption("아직 플래그 없음")

    with st.expander("Probing questions"):
        pqs = summary.get("probing_questions") or []
        if pqs:
            st.dataframe(pqs, use_container_width=True)
        else:
            st.caption("아직 질문 없음")

    with st.expander("Claims"):
        claims = summary.get("claims") or []
        if claims:
            st.dataframe(claims, use_container_width=True)

    with st.expander("Answer evaluations"):
        evals = summary.get("answer_eval") or []
        if evals:
            st.dataframe(evals, use_container_width=True)

    with st.expander("Cross-check pairs"):
        pairs = summary.get("cross_check") or []
        if pairs:
            st.dataframe(pairs, use_container_width=True)

    with st.expander("Raw state"):
        st.json(summary)


# ---------- main ----------

def main() -> None:
    st.set_page_config(page_title="HireMindset", layout="wide")
    _inject_ga4(GA_ID)
    _init_state()

    st.title("HireMindset")
    st.caption("LangGraph 기반 Cross-check 인터뷰 에이전트")

    with st.sidebar:
        api_base = st.text_input("API URL", value=DEFAULT_API).rstrip("/")
        st.divider()
        st.session_state.max_rounds = st.slider(
            "최대 라운드 수",
            min_value=3,
            max_value=40,
            value=int(st.session_state.max_rounds),
            step=1,
            help=(
                "세션당 허용되는 최대 라운드. context probe + flag 질문 "
                "+ 폴백/심층/주입 질문을 전부 포함한 숫자입니다. "
                "세션 시작 이후에는 새로 시작할 때 반영됩니다."
            ),
            disabled=bool(st.session_state.thread_id),
        )
        st.divider()
        if st.session_state.thread_id:
            st.code(f"thread = {st.session_state.thread_id[:12]}…")
            if st.button("세션 리셋", use_container_width=True):
                _reset_session()
                st.rerun()
        else:
            st.caption("새 세션을 시작할 준비가 되었습니다.")
        if st.session_state.last_error:
            st.error(st.session_state.last_error)

    tab_interview, tab_dashboard = st.tabs(["인터뷰", "대시보드"])

    with tab_interview:
        if st.session_state.thread_id is None:
            _render_setup(api_base)
        else:
            _render_history()
            if st.session_state.done:
                _render_summary()
            elif st.session_state.pending:
                _render_pending(api_base)
            else:
                st.info("다음 단계를 기다리는 중…")

    with tab_dashboard:
        _render_dashboard()


if __name__ == "__main__":
    main()
