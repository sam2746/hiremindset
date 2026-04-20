"""seed_fallback_probe: control=="fallback"일 때만 꼬리질문을 큐에 투입."""

from hiremindset.graph.nodes.seed_fallback import seed_fallback_probe


def _base_state(control="fallback"):
    return {
        "probing_questions": [
            {
                "id": "pq0",
                "queue_id": "q0",
                "text": "측정 기준은?",
                "asked_round": 1,
                "target_flag_id": "f0",
                "target_claim_ids": ["c0"],
                "profile": "numeric",
            }
        ],
        "turns": [{"q_id": "pq0", "role": "human", "answer_text": "기억이 안 납니다"}],
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
        "suspicion_flags": [
            {
                "id": "f0",
                "claim_ids": ["c0"],
                "category": "metric_unsupported",
                "severity": 3,
                "evidence": "",
                "strikes": 0,
                "fallback_attempts": 1,
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


def _fake_seeder(last_q, answer_text, target_claims, flag):
    return "그 프로젝트의 .gitignore에 뭐를 넣었나요?"


def test_fallback_pushes_new_probe_item_with_inherited_profile():
    out = seed_fallback_probe(_base_state(), seeder=_fake_seeder)
    queue = out["probe_queue"]
    assert len(queue) == 2
    new_item = [it for it in queue if it["source"] == "fallback"][0]
    assert new_item["priority"] == 40  # 30 + 10
    assert new_item["profile"] == "numeric"  # 직전 질문 승계
    assert new_item["target_flag_id"] == "f0"
    assert new_item["pre_generated_text"].startswith("그 프로젝트")
    assert out["control"] == "continue"


def test_non_fallback_control_is_noop():
    out = seed_fallback_probe(_base_state(control="continue"), seeder=_fake_seeder)
    assert out == {}


def test_empty_state_returns_continue_reset():
    out = seed_fallback_probe({"control": "fallback"}, seeder=_fake_seeder)
    assert out == {"control": "continue"}
