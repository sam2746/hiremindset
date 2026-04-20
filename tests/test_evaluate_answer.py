"""evaluate_answer: AI 채점 결과가 누적되는지 확인. fallback 시드 로직은 별도 노드."""

from hiremindset.graph.nodes.evaluate_answer import evaluate_answer


def _base_state(answer="응답시간 40% 개선했습니다"):
    return {
        "probing_questions": [
            {
                "id": "pq0",
                "queue_id": "q0",
                "text": "측정 기준은?",
                "asked_round": 0,
                "target_flag_id": "f0",
                "target_claim_ids": ["c0"],
                "profile": "numeric",
            }
        ],
        "turns": [{"q_id": "pq0", "role": "human", "answer_text": answer}],
        "claims": [
            {
                "id": "c0",
                "text": "응답시간 40% 개선",
                "type": "achievement",
                "source_paragraph_id": "p0",
                "entities": ["40%"],
                "confidence": 1.0,
            }
        ],
        "documents": {"paragraphs": [{"id": "p0", "text": "p0 본문"}]},
        "suspicion_flags": [
            {
                "id": "f0",
                "claim_ids": ["c0"],
                "category": "metric_unsupported",
                "severity": 3,
                "evidence": "",
                "strikes": 0,
                "fallback_attempts": 0,
                "resolved": False,
            }
        ],
        "answer_eval": [],
    }


def _fake_evaluator(last_q, answer_text, target_claims, flag, paragraphs):
    return {
        "q_id": "",
        "specificity": 0.6,
        "consistency": 0.8,
        "epistemic": 0.5,
        "refusal_or_hedge": False,
        "suggest": ["drill"],
    }


def test_appends_answer_eval_with_q_id():
    out = evaluate_answer(_base_state(), evaluator=_fake_evaluator)
    assert len(out["answer_eval"]) == 1
    assert out["answer_eval"][0]["q_id"] == "pq0"
    assert out["answer_eval"][0]["specificity"] == 0.6
    # control·queue는 건드리지 않는다.
    assert "probe_queue" not in out
    assert "control" not in out


def test_empty_answer_skips_evaluation():
    out = evaluate_answer(_base_state(answer=""), evaluator=_fake_evaluator)
    assert out == {}


def test_no_state_returns_empty():
    assert evaluate_answer({}, evaluator=_fake_evaluator) == {}
