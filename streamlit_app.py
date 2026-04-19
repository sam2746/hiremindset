"""
Streamlit UI: calls FastAPI /run (no direct graph import — keeps UI thin).
Run API first: uvicorn hiremindset.api.main:app --reload
"""

import os

import httpx
import streamlit as st

DEFAULT_API = os.environ.get("API_PUBLIC_URL", "http://127.0.0.1:8000").rstrip("/")
GA_ID = os.environ.get("GA_MEASUREMENT_ID", "").strip()


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


def main() -> None:
    st.set_page_config(page_title="HireMindset", layout="wide")
    _inject_ga4(GA_ID)

    st.title("HireMindset")
    st.caption("Cross-check interview agent (stub pipeline)")

    if "turns" not in st.session_state:
        st.session_state.turns = []

    api_base = st.sidebar.text_input("API base URL", value=DEFAULT_API)

    resume = st.text_area("Resume / profile", height=200, placeholder="Paste resume text…")
    jd = st.text_area("Job description (optional)", height=120, placeholder="Paste JD…")

    if st.button("Run cross-check pipeline", type="primary"):
        with st.spinner("Calling API…"):
            try:
                r = httpx.post(
                    f"{api_base}/run",
                    json={"resume_text": resume, "jd_text": jd},
                    timeout=120.0,
                )
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                st.error(f"API error: {e}")
                return

        st.session_state.last_result = data
        st.session_state.turns.append({"role": "system", "content": str(data)})

    if st.session_state.get("last_result"):
        data = st.session_state.last_result
        st.subheader("Opening questions (stub)")
        for q in data.get("opening_questions") or []:
            st.write(f"- {q}")
        st.subheader("Report")
        st.markdown(data.get("report_markdown") or "_empty_")

    with st.expander("Session debug (session_state)"):
        st.json(dict(st.session_state))


if __name__ == "__main__":
    main()
