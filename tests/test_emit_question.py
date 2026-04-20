from hiremindset.graph.nodes.emit_question import emit_question


def _item(qid, priority, profile="mechanism", flag_id="f0", claim_ids=("c0",)):
    return {
        "id": qid,
        "target_claim_ids": list(claim_ids),
        "intent": "",
        "expected_signal": "",
        "profile": profile,
        "attempts": 0,
        "target_flag_id": flag_id,
        "priority": priority,
        "source": "plan",
    }


def _claim(cid, pid="p0"):
    return {
        "id": cid,
        "text": f"claim {cid}",
        "type": "factual",
        "source_paragraph_id": pid,
        "entities": [],
        "confidence": 1.0,
    }


def test_emit_question_no_queue_returns_empty_patch():
    out = emit_question({"probe_queue": []})
    assert out == {}


def test_emit_question_picks_highest_priority():
    state = {
        "probe_queue": [
            _item("q0", 10, profile="story", flag_id="fA"),
            _item("q1", 50, profile="consistency", flag_id="fB"),
            _item("q2", 30, profile="numeric", flag_id="fC"),
        ],
        "claims": [_claim("c0")],
        "documents": {"paragraphs": []},
        "strategy": {"round": 2},
    }
    calls = {}

    def fake_gen(item, claims_by_id, flag, paragraphs):
        calls["item_id"] = item["id"]
        return f"Q for {item['id']}"

    out = emit_question(state, generator=fake_gen)
    assert calls["item_id"] == "q1"
    assert out["probing_questions"][-1]["text"] == "Q for q1"
    assert out["probing_questions"][-1]["queue_id"] == "q1"
    assert out["probing_questions"][-1]["asked_round"] == 2
    remaining_ids = [it["id"] for it in out["probe_queue"]]
    assert remaining_ids == ["q0", "q2"]


def test_emit_question_advances_strategy_round_and_last_profile():
    state = {
        "probe_queue": [_item("q0", 10, profile="numeric")],
        "claims": [_claim("c0")],
        "documents": {"paragraphs": []},
    }
    out = emit_question(state, generator=lambda *a, **kw: "질문")
    assert out["strategy"]["round"] == 1
    assert out["strategy"]["last_profile"] == "numeric"
    assert out["turns"][-1]["q_id"] == out["probing_questions"][-1]["id"]
    assert out["turns"][-1]["answer_text"] == ""


def test_emit_question_appends_to_existing_questions_and_turns():
    state = {
        "probe_queue": [_item("q0", 10)],
        "claims": [_claim("c0")],
        "documents": {"paragraphs": []},
        "probing_questions": [
            {"id": "pq0", "queue_id": "qx", "text": "prev", "asked_round": 0}
        ],
        "turns": [{"q_id": "pq0", "role": "simulator", "answer_text": "past"}],
        "strategy": {"round": 1},
    }
    out = emit_question(state, generator=lambda *a, **kw: "새 질문")
    assert [pq["id"] for pq in out["probing_questions"]] == ["pq0", "pq1"]
    assert out["probing_questions"][-1]["asked_round"] == 1
    assert len(out["turns"]) == 2
