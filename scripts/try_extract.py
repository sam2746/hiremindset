"""
노드 단위 수동 확인용 CLI.

사용:
  python scripts/try_extract.py resume samples/resume_example.txt
  python scripts/try_extract.py essay  samples/essay_example.txt samples/jd_example.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from hiremindset.graph.nodes.extract import (
    extract_claims_essay,
    extract_claims_resume,
)
from hiremindset.graph.nodes.ingest import ingest_normalize


def _usage() -> int:
    print("사용: python scripts/try_extract.py <resume|essay> <input.txt> [jd.txt]")
    return 2


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] not in {"resume", "essay"}:
        return _usage()

    kind = argv[1]
    text = Path(argv[2]).read_text(encoding="utf-8")
    jd = Path(argv[3]).read_text(encoding="utf-8") if len(argv) >= 4 else ""

    load_dotenv()

    ingested = ingest_normalize(
        {"documents": {"kind": kind, "raw": text, "paragraphs": []}, "jd": jd}
    )
    state = dict(ingested)

    print("── paragraphs ──")
    for p in state["documents"]["paragraphs"]:
        print(f"  {p['id']} | {p['text']}")

    node = extract_claims_resume if kind == "resume" else extract_claims_essay
    out = node(state)

    print("\n── claims ──")
    print(json.dumps(out.get("claims", []), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
