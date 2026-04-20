"""
ingest → extract → cross_check → flag → plan_probe → emit_question 전체 파이프라인 러너.

사용:
  python scripts/try_probe.py resume samples/resume_example.txt
  python scripts/try_probe.py essay  samples/essay_example.txt samples/jd_example.txt [N]

N은 emit_question을 몇 번 반복할지(기본 3). 실제 턴에서는 HITL로 답변이 들어와야 다음으로
넘어가지만, 여기서는 큐에서 top을 순차로 소비해 샘플 질문 리포트를 확인한다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from hiremindset.graph.nodes.cross_check import cross_check_claims
from hiremindset.graph.nodes.emit_question import emit_question
from hiremindset.graph.nodes.extract import (
    extract_claims_essay,
    extract_claims_resume,
)
from hiremindset.graph.nodes.flag import flag_suspicion
from hiremindset.graph.nodes.ingest import ingest_normalize
from hiremindset.graph.nodes.plan_probe import plan_probe


def _usage() -> int:
    print(
        "사용: python scripts/try_probe.py <resume|essay> <input.txt> [jd.txt] [반복N]"
    )
    return 2


def _merge(state: dict, patch: dict) -> dict:
    out = dict(state)
    out.update(patch)
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] not in {"resume", "essay"}:
        return _usage()

    kind = argv[1]
    text = Path(argv[2]).read_text(encoding="utf-8")
    jd = ""
    repeat = 3
    if len(argv) >= 4 and argv[3].isdigit():
        repeat = int(argv[3])
    elif len(argv) >= 4:
        jd = Path(argv[3]).read_text(encoding="utf-8")
        if len(argv) >= 5 and argv[4].isdigit():
            repeat = int(argv[4])

    load_dotenv()

    state: dict = ingest_normalize(
        {"documents": {"kind": kind, "raw": text, "paragraphs": []}, "jd": jd}
    )
    node = extract_claims_resume if kind == "resume" else extract_claims_essay
    state = _merge(state, node(state))
    state = _merge(state, cross_check_claims(state))
    state = _merge(state, flag_suspicion(state))
    state = _merge(state, plan_probe(state))

    print("── initial probe_queue (priority desc) ──")
    for it in state.get("probe_queue", []):
        print(
            f"  {it['id']} | prio={it.get('priority')} | {it['profile']:>11} "
            f"| flag={it.get('target_flag_id')} | claims={it['target_claim_ids']}"
        )

    print(f"\n── emit_question x{repeat} ──")
    for i in range(repeat):
        if not state.get("probe_queue"):
            print(f"[{i}] 큐 비어있음. 종료.")
            break
        state = _merge(state, emit_question(state))
        pq = state["probing_questions"][-1]
        print(f"\n[Q{i+1}] (queue_id={pq['queue_id']}, round={pq['asked_round']})")
        print(f"  > {pq['text']}")

    print("\n── residual queue ──")
    print(
        json.dumps(
            [
                {"id": it["id"], "priority": it.get("priority"), "profile": it["profile"]}
                for it in state.get("probe_queue", [])
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
