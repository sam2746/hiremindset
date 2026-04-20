"""collect_answer 순수 로직 (answer-only)."""

from hiremindset.graph.nodes.collect_answer import apply_answer_response


def _base_state():
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
        "turns": [{"q_id": "pq0", "role": "simulator", "answer_text": ""}],
    }


def test_records_answer_text_and_promotes_role_to_human():
    state = _base_state()
    out = apply_answer_response(state, {"answer_text": "k6로 1시간 측정했습니다"})
    assert out["turns"][-1]["answer_text"] == "k6로 1시간 측정했습니다"
    assert out["turns"][-1]["role"] == "human"
    # collect_answer는 control/suspicion_flags를 건드리지 않는다.
    assert "control" not in out
    assert "suspicion_flags" not in out


def test_empty_answer_is_allowed():
    out = apply_answer_response(_base_state(), {"answer_text": ""})
    assert out["turns"][-1]["answer_text"] == ""


def test_no_questions_or_turns_returns_empty_patch():
    assert apply_answer_response({}, {"answer_text": "x"}) == {}
