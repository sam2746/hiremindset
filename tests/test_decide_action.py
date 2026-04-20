"""decide_action 순수 로직 (accept/fallback/drill/pass/skip/inject 분기)."""

import pytest

from hiremindset.graph.nodes.decide_action import apply_decision_response


def _base_state(queue_item=None):
    state = {
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
        "turns": [{"q_id": "pq0", "role": "human", "answer_text": "측정했어요"}],
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
        "answer_eval": [
            {
                "q_id": "pq0",
                "specificity": 0.3,
                "consistency": 0.4,
                "epistemic": 0.7,
                "refusal_or_hedge": True,
                "suggest": ["drill"],
            }
        ],
    }
    if queue_item:
        state["probe_queue"] = [queue_item]
    return state


def test_accept_marks_flag_resolved_and_sets_continue():
    out = apply_decision_response(_base_state(), {"action": "accept"})
    assert out["suspicion_flags"][0]["resolved"] is True
    assert out["control"] == "continue"
    assert "probe_queue" not in out


def test_fallback_bumps_fallback_attempts_and_sets_control():
    out = apply_decision_response(_base_state(), {"action": "fallback"})
    assert out["control"] == "fallback"
    assert out["suspicion_flags"][0]["fallback_attempts"] == 1
    assert out["suspicion_flags"][0]["resolved"] is False


def test_inject_pushes_new_item_with_highest_priority():
    existing = {
        "id": "q9",
        "target_claim_ids": ["c1"],
        "intent": "",
        "expected_signal": "",
        "profile": "story",
        "attempts": 0,
        "priority": 50,
        "source": "plan",
    }
    out = apply_decision_response(
        _base_state(queue_item=existing),
        {
            "action": "inject",
            "injected_question": "그 때 .gitignore에 뭐를 넣었죠?",
        },
    )
    queue = out["probe_queue"]
    assert len(queue) == 2
    injected = [it for it in queue if it["source"] == "hitl"][0]
    assert injected["priority"] == 60  # 50 + 10
    assert injected["pre_generated_text"] == "그 때 .gitignore에 뭐를 넣었죠?"
    assert injected["profile"] == "numeric"  # 직전 질문 profile 승계
    assert injected["target_flag_id"] == "f0"
    assert injected["id"] != "q9"
    assert out["control"] == "continue"


def test_drill_sets_control_without_touching_flag_or_queue():
    out = apply_decision_response(_base_state(), {"action": "drill"})
    assert out["control"] == "drill"
    assert "suspicion_flags" not in out
    assert "probe_queue" not in out


def test_pass_sets_continue_without_touching_flag_or_queue():
    out = apply_decision_response(_base_state(), {"action": "pass"})
    assert out["control"] == "continue"
    assert "suspicion_flags" not in out
    assert "probe_queue" not in out


def test_skip_sets_continue_like_pass_but_distinct_log():
    out = apply_decision_response(_base_state(), {"action": "skip"})
    assert out["control"] == "continue"
    assert "suspicion_flags" not in out
    log = out.get("decision_log") or []
    assert log[-1]["action"] == "skip"
    assert log[-1]["why"] == "추천 질문 미사용 — 다음 큐로"


def test_inject_requires_injected_question():
    with pytest.raises(ValueError):
        apply_decision_response(
            _base_state(), {"action": "inject", "injected_question": ""}
        )


def test_unknown_action_raises():
    with pytest.raises(ValueError):
        apply_decision_response(_base_state(), {"action": "nope"})


def test_no_questions_returns_empty_patch():
    assert apply_decision_response({}, {"action": "accept"}) == {}


def test_decision_log_appended_with_ai_eval_snapshot():
    out = apply_decision_response(_base_state(), {"action": "fallback"})
    log = out.get("decision_log") or []
    assert len(log) == 1
    entry = log[0]
    assert entry["action"] == "fallback"
    assert entry["question_id"] == "pq0"
    assert entry["flag_id"] == "f0"
    assert entry["profile"] == "numeric"
    assert entry["ai_suggest"] == ["drill"]
    assert entry["ai_specificity"] == 0.3
    assert entry["ai_hedge"] is True
    assert entry["fallback_used"] is True


def test_decision_log_preserves_prior_entries():
    state = _base_state()
    state["decision_log"] = [{"round": 1, "action": "accept", "why": "prev"}]
    out = apply_decision_response(state, {"action": "pass"})
    log = out["decision_log"]
    assert len(log) == 2
    assert log[0]["round"] == 1  # 기존 항목 보존
    assert log[1]["action"] == "pass"
