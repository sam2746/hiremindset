"""collect_answer 순수 로직 (answer-only 및 즉시 통과)."""

from hiremindset.graph.nodes.collect_answer import (
    apply_answer_response,
    apply_collect_response,
)


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
        "strategy": {"round": 1},
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


def test_immediate_accept_merges_accept_and_sets_skip_flag():
    state = _base_state()
    out = apply_collect_response(
        state,
        {"answer_text": "k6로 측정했어요", "immediate_action": "accept"},
    )
    assert out["turns"][-1]["answer_text"] == "k6로 측정했어요"
    assert out["skip_evaluate_decide"] is True
    assert out["suspicion_flags"][0]["resolved"] is True
    log = out.get("decision_log") or []
    assert log[-1]["immediate"] is True
    assert "즉시 통과" in log[-1]["why"]


def test_without_immediate_delegates_to_answer_only():
    state = _base_state()
    out = apply_collect_response(state, {"answer_text": "답"})
    assert "skip_evaluate_decide" not in out
    assert "decision_log" not in out
