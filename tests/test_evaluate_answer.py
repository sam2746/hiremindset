from hiremindset.graph.nodes.evaluate_answer import evaluate_answer


def _base_state(control="continue", answer="답변"):
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
        "probe_queue": [],
        "answer_eval": [],
        "control": control,
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


def test_evaluate_writes_answer_eval_and_stops_when_not_fallback():
    state = _base_state(control="continue")
    out = evaluate_answer(state, evaluator=_fake_evaluator)
    assert len(out["answer_eval"]) == 1
    assert out["answer_eval"][0]["q_id"] == "pq0"
    assert out["answer_eval"][0]["specificity"] == 0.6
    assert "probe_queue" not in out
    assert "control" not in out


def test_evaluate_returns_empty_when_answer_is_empty():
    state = _base_state(answer="")
    out = evaluate_answer(state, evaluator=_fake_evaluator)
    assert out == {}


def test_evaluate_on_fallback_pushes_new_probe_item_with_inherited_profile():
    state = _base_state(control="fallback")
    state["probe_queue"] = [
        {
            "id": "q5",
            "target_claim_ids": ["cX"],
            "intent": "",
            "expected_signal": "",
            "profile": "story",
            "attempts": 0,
            "priority": 30,
            "source": "plan",
        }
    ]

    def fake_seeder(last_q, answer_text, target_claims, flag):
        return "어떤 도구로, 얼마 동안 측정한 값인가요?"

    out = evaluate_answer(
        state, evaluator=_fake_evaluator, seeder=fake_seeder
    )
    assert len(out["answer_eval"]) == 1
    queue = out["probe_queue"]
    assert len(queue) == 2
    new_item = [it for it in queue if it["source"] == "fallback"][0]
    assert new_item["priority"] == 40  # 30 + 10
    assert new_item["profile"] == "numeric"  # 직전 질문 profile 승계
    assert new_item["target_flag_id"] == "f0"
    assert new_item["pre_generated_text"].startswith("어떤 도구")
    # fallback 처리 후 control은 continue로 복귀
    assert out["control"] == "continue"


def test_evaluate_no_state_returns_empty():
    assert evaluate_answer({}, evaluator=_fake_evaluator) == {}
