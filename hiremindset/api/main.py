import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from hiremindset.api.schemas import RunRequest, RunResponse
from hiremindset.graph.builder import build_graph

app = FastAPI(title="HireMindset API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def get_graph():
    return build_graph()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
def run_pipeline(body: RunRequest) -> RunResponse:
    graph = get_graph()
    try:
        out = graph.invoke(
            {
                "resume_text": body.resume_text,
                "jd_text": body.jd_text,
            }
        )
    except Exception as e:  # noqa: BLE001 — surface to client during dev
        raise HTTPException(status_code=500, detail=str(e)) from e

    return RunResponse(
        opening_questions=list(out.get("opening_questions") or []),
        report_markdown=str(out.get("report_markdown") or ""),
        error=out.get("error"),
    )
