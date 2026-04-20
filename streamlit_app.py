"""
Streamlit UI.

원칙: 그래프를 import하지 않고 FastAPI(/session/start, /session/resume)만 호출한다.

Phase 1 이후 한 라운드는 두 개의 HITL 단계로 나뉜다.
  1) phase="collect_answer" : 후보자 답변 텍스트만 제출
  2) phase="decide_action"  : AI 평가 카드를 본 뒤 accept/fallback/inject 결정
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

DEFAULT_API = os.environ.get("API_PUBLIC_URL", "http://127.0.0.1:8000").rstrip("/")
GA_ID = os.environ.get("GA_MEASUREMENT_ID", "").strip()

ACTIONS = ("accept", "fallback", "inject")
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
    ss.setdefault("action", "accept")


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
        {"kind": ss.kind, "text": ss.resume_text, "jd": ss.jd_text},
    )
    if not data:
        return
    _apply_step(data)
    if ss.pending and ss.pending.get("phase") == "collect_answer":
        ss.history.append(
            {"role": "simulator", "text": ss.pending["text"], "meta": ss.pending}
        )


def _submit_answer(api_base: str, answer: str) -> None:
    ss = st.session_state
    if not ss.thread_id or not ss.pending:
        return
    ss.last_error = None
    ss.history.append({"role": "human", "text": answer, "meta": {}})
    data = _post(
        api_base,
        "/session/resume",
        {
            "thread_id": ss.thread_id,
            "phase": "collect_answer",
            "answer_text": answer,
        },
    )
    if not data:
        return
    _apply_step(data)
    # decide_action phase면 채팅 히스토리에는 기록하지 않고 카드로만 보여준다.


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
                    st.markdown(_chip(f"action: {action}"), unsafe_allow_html=True)
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
    if st.button("답변 제출", type="primary"):
        with st.spinner("AI 평가 중…"):
            _submit_answer(api_base, answer)
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
    c1.metric("specificity", _fmt_score(ai.get("specificity")))
    c2.metric("consistency", _fmt_score(ai.get("consistency")))
    c3.metric("epistemic", _fmt_score(ai.get("epistemic")))
    c4.metric("hedge", "Yes" if ai.get("refusal_or_hedge") else "No")
    suggest = ai.get("suggest") or []
    if suggest:
        st.markdown(
            "**AI suggest**: "
            + " ".join(_chip(s) for s in suggest),
            unsafe_allow_html=True,
        )

    ss.action = st.radio(
        "면접관 결정",
        options=ACTIONS,
        horizontal=True,
        index=ACTIONS.index(ss.action if ss.action in ACTIONS else "accept"),
        help=(
            "accept — 답변이 충분, 다음 질문으로 (해당 flag resolved) / "
            "fallback — 답이 부족, AI가 폴백 질문을 생성해 큐 최상단에 투입 / "
            "inject — 면접관이 직접 다음 질문을 지정"
        ),
        key=f"decide_action_{pq.get('question_id')}",
    )
    injected = ""
    if ss.action == "inject":
        injected = st.text_area(
            "inject할 다음 질문",
            key=f"inject_{pq.get('question_id')}",
            height=80,
            placeholder="면접관이 직접 이어갈 질문을 입력하세요. 이 질문이 큐 최상단에 들어갑니다.",
        )

    if st.button("결정 제출", type="primary"):
        if ss.action == "inject" and not injected.strip():
            st.warning("inject 액션에는 다음 질문이 필요합니다.")
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
    st.success(
        f"세션 종료 · round {strategy.get('round', '?')} / "
        f"max {strategy.get('max_rounds', '?')}"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Claims", len(summary.get("claims") or []))
    c2.metric("Suspicion", len(summary.get("suspicion_flags") or []))
    c3.metric("Questions", len(summary.get("probing_questions") or []))
    c4.metric("Answers", len(summary.get("answer_eval") or []))


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
