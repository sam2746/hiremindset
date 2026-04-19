from fastapi.testclient import TestClient

from hiremindset.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_run_returns_stub_payload():
    r = client.post(
        "/run",
        json={"resume_text": "백엔드 3년", "jd_text": "Python"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "opening_questions" in data
    assert "report_markdown" in data
    assert len(data["opening_questions"]) >= 1
    assert "Stub report" in data["report_markdown"]


def test_run_rejects_empty_resume():
    r = client.post("/run", json={"resume_text": "", "jd_text": ""})
    assert r.status_code == 422
