from hiremindset.graph.nodes.collect_answer import apply_interrupt_response


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
        "suspicion_flags": [
            {
                "id": "f0",
                "claim_ids": ["c0"],
                "category": "metric_unsupported",
                "severity": 3,
                "evidence": "e",
                "strikes": 0,
                "fallback_attempts": 0,
                "resolved": False,
            }
        ],
        "probe_queue": [],
    }


def test_accept_marks_flag_resolved_and_records_answer():
    state = _base_state()
    out = apply_interrupt_response(
        state, {"answer_text": "k6로 1시간 측정했습니다", "action": "accept"}
    )
    assert out["turns"][-1]["answer_text"] == "k6로 1시간 측정했습니다"
    assert out["turns"][-1]["role"] == "human"
    assert out["suspicion_flags"][0]["resolved"] is True
    assert out["control"] == "continue"
    assert "probe_queue" not in out


def test_fallback_bumps_fallback_attempts_and_sets_control():
    state = _base_state()
    out = apply_interrupt_response(
        state, {"answer_text": "정확히 기억이 안 납니다", "action": "fallback"}
    )
    assert out["control"] == "fallback"
    assert out["suspicion_flags"][0]["fallback_attempts"] == 1
    assert out["suspicion_flags"][0]["resolved"] is False


def test_inject_pushes_new_item_with_highest_priority():
    state = _base_state()
    state["probe_queue"] = [
        {
            "id": "q9",
            "target_claim_ids": ["c1"],
            "intent": "",
            "expected_signal": "",
            "profile": "story",
            "attempts": 0,
            "priority": 50,
            "source": "plan",
        }
    ]
    out = apply_interrupt_response(
        state,
        {
            "answer_text": "",
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
    # id 충돌 방지: 기존 q9와 다른 번호
    assert injected["id"] != "q9"


def test_inject_requires_injected_question():
    state = _base_state()
    try:
        apply_interrupt_response(
            state, {"answer_text": "x", "action": "inject", "injected_question": ""}
        )
    except ValueError:
        return
    raise AssertionError("inject without injected_question must raise")


def test_unknown_action_raises():
    state = _base_state()
    try:
        apply_interrupt_response(state, {"answer_text": "x", "action": "nope"})
    except ValueError:
        return
    raise AssertionError("unknown action must raise")


def test_no_questions_or_turns_returns_empty_patch():
    assert apply_interrupt_response({}, {"answer_text": "x", "action": "accept"}) == {}
