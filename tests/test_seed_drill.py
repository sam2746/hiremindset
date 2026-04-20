"""seed_drill_probe: control=="drill"일 때만 심층 질문을 큐에 투입."""

from hiremindset.graph.nodes.seed_drill import seed_drill_probe


def _base_state(control="drill"):
    return {
        "probing_questions": [
            {
                "id": "pq0",
                "queue_id": "q0",
                "text": "어떤 인증 방식을 썼어요?",
                "asked_round": 1,
                "target_flag_id": "f0",
                "target_claim_ids": ["c0"],
                "profile": "story",
            }
        ],
        "turns": [{"q_id": "pq0", "role": "human", "answer_text": "JWT 썼어요"}],
        "claims": [
            {
                "id": "c0",
                "text": "JWT 인증 도입",
                "type": "factual",
                "source_paragraph_id": "p0",
                "entities": ["JWT"],
                "confidence": 1.0,
            }
        ],
        "suspicion_flags": [
            {
                "id": "f0",
                "claim_ids": ["c0"],
                "category": "technical_probe_needed",
                "severity": 2,
                "evidence": "",
                "strikes": 0,
                "fallback_attempts": 0,
                "resolved": False,
            }
        ],
        "probe_queue": [
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
        ],
        "control": control,
    }


def _fake_drill_seeder(last_q, answer_text, target_claims):
    return "JWT 서명은 어떻게 검증하나요?"


def test_drill_pushes_mechanism_item_with_boosted_priority():
    out = seed_drill_probe(_base_state(), seeder=_fake_drill_seeder)
    queue = out["probe_queue"]
    assert len(queue) == 2
    new_item = [it for it in queue if it["source"] == "drill"][0]
    assert new_item["priority"] == 40  # 30 + 10
    # drill은 profile을 무조건 mechanism으로 고정
    assert new_item["profile"] == "mechanism"
    # flag와 무관하므로 target_flag_id는 붙지 않는다
    assert "target_flag_id" not in new_item
    assert new_item["pre_generated_text"].startswith("JWT")
    assert out["control"] == "continue"


def test_non_drill_control_is_noop():
    out = seed_drill_probe(_base_state(control="continue"), seeder=_fake_drill_seeder)
    assert out == {}


def test_empty_state_returns_continue_reset():
    out = seed_drill_probe({"control": "drill"}, seeder=_fake_drill_seeder)
    assert out == {"control": "continue"}
