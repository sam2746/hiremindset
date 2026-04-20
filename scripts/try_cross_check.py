"""
ingest → extract → cross_check → flag 파이프라인 수동 확인용 CLI.

사용:
  python scripts/try_cross_check.py resume samples/resume_example.txt
  python scripts/try_cross_check.py essay  samples/essay_example.txt samples/jd_example.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from hiremindset.graph.nodes.cross_check import candidate_pairs, cross_check_claims
from hiremindset.graph.nodes.extract import (
    extract_claims_essay,
    extract_claims_resume,
)
from hiremindset.graph.nodes.flag import flag_suspicion
from hiremindset.graph.nodes.ingest import ingest_normalize


def _usage() -> int:
    print(
        "사용: python scripts/try_cross_check.py <resume|essay> <input.txt> [jd.txt]"
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
    jd = Path(argv[3]).read_text(encoding="utf-8") if len(argv) >= 4 else ""

    load_dotenv()

    state = ingest_normalize(
        {"documents": {"kind": kind, "raw": text, "paragraphs": []}, "jd": jd}
    )

    extract_node = extract_claims_resume if kind == "resume" else extract_claims_essay
    state = _merge(state, extract_node(state))

    print("── claims ──")
    print(json.dumps(state.get("claims", []), ensure_ascii=False, indent=2))

    print("\n── candidate pairs (룰 필터링 후) ──")
    pairs = candidate_pairs(state.get("claims") or [])
    print(pairs)

    state = _merge(state, cross_check_claims(state))
    print("\n── cross_check ──")
    print(json.dumps(state.get("cross_check", []), ensure_ascii=False, indent=2))

    state = _merge(state, flag_suspicion(state))
    print("\n── suspicion_flags ──")
    print(json.dumps(state.get("suspicion_flags", []), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
