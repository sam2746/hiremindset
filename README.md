# HireMindset

교차 검증(Cross-check) 면접 에이전트 — **LangGraph** + **FastAPI** + **Streamlit**.  

## 로컬 설정

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .

cp .env.example .env
```

### API 실행

```bash
uvicorn hiremindset.api.main:app --reload --host 127.0.0.1 --port 8000
```

### Streamlit 실행 (다른 터미널)

```bash
streamlit run streamlit_app.py
```

### Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

- API 문서: http://localhost:8000/docs  
- Streamlit: http://localhost:8501 (Compose 네트워크 안에서는 API를 `http://api:8000` 으로 호출하도록 맞춰 둠)

## 프로젝트 구조

- `hiremindset/graph/` — LangGraph `GraphState`, `build_graph()`
- `hiremindset/api/` — FastAPI (`/health`, `/run`)
- `streamlit_app.py` — UI; HTTP로만 API 호출 (그래프 직접 import 없음)
- `samples/` — 예시 이력서·JD 텍스트

## 테스트

개발 의존성 포함 설치 후 `pytest` 로 API·그래프 스모크 테스트를 돌릴 수 있습니다.

```bash
pip install -e ".[dev]"
pytest -q
```

## GA4 (선택)

`.env`에 `GA_MEASUREMENT_ID=G-...` 를 넣으면 Streamlit에서 gtag를 불러옵니다.
