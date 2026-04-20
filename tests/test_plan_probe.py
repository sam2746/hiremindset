from hiremindset.graph.nodes.plan_probe import (
    CATEGORY_TIER,
    PROFILE_MAP,
    plan_probe,
)


def _flag(fid, category, severity, claim_ids, resolved=False):
    return {
        "id": fid,
        "claim_ids": claim_ids,
        "category": category,
        "severity": severity,
        "evidence": "",
        "strikes": 0,
        "fallback_attempts": 0,
        "resolved": resolved,
    }


def test_plan_probe_orders_by_priority_desc():
    flags = [
        _flag("f0", "cliche_template", 3, ["c0"]),
        _flag("f1", "timeline_conflict", 4, ["c1", "c2"]),
        _flag("f2", "technical_probe_needed", 2, ["c3"]),
    ]
    out = plan_probe({"suspicion_flags": flags})
    ids = [it["target_flag_id"] for it in out["probe_queue"]]
    assert ids[0] == "f1"  # severity 4 * 10 + tier 6 = 46
    assert out["probe_queue"][0]["priority"] == 46
    assert out["probe_queue"][0]["profile"] == PROFILE_MAP["timeline_conflict"]
    prios = [it["priority"] for it in out["probe_queue"]]
    assert prios == sorted(prios, reverse=True)


def test_plan_probe_skips_resolved_flags():
    flags = [
        _flag("f0", "metric_unsupported", 3, ["c0"], resolved=True),
        _flag("f1", "metric_unsupported", 3, ["c1"]),
    ]
    out = plan_probe({"suspicion_flags": flags})
    ids = [it["target_flag_id"] for it in out["probe_queue"]]
    assert ids == ["f1"]


def test_plan_probe_is_idempotent_on_existing_queue():
    flags = [_flag("f0", "metric_unsupported", 3, ["c0"])]
    state = {"suspicion_flags": flags}
    first = plan_probe(state)
    state["probe_queue"] = first["probe_queue"]
    second = plan_probe(state)
    assert len(second["probe_queue"]) == 1
    assert second["probe_queue"][0]["target_flag_id"] == "f0"


def test_plan_probe_priority_formula():
    flags = [_flag("f0", "depth_collapse", 5, ["c0"])]
    out = plan_probe({"suspicion_flags": flags})
    assert out["probe_queue"][0]["priority"] == 5 * 10 + CATEGORY_TIER["depth_collapse"]
    assert out["probe_queue"][0]["source"] == "plan"
