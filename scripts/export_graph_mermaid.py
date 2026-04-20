"""
LangGraph 컴파일 결과를 Mermaid flowchart 문자열로 덤프한다.

사용:
  python scripts/export_graph_mermaid.py
  python scripts/export_graph_mermaid.py --stdout   # 파일 대신 표준출력만

기본 출력: 프로젝트 루트 기준 ``docs/graph.mmd``
(README나 Notion, GitHub, mermaid.live 등에 붙여 넣기용)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hiremindset.graph.builder import build_graph


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export LangGraph as Mermaid")
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="파일에 쓰지 않고 Mermaid만 출력",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="출력 경로 (기본: docs/graph.mmd)",
    )
    args = parser.parse_args()

    app = build_graph()
    mermaid = app.get_graph().draw_mermaid()

    if args.stdout:
        print(mermaid)
        return 0

    out = args.output or (_repo_root() / "docs" / "graph.mmd")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(mermaid, encoding="utf-8")
    print(f"Wrote {out} ({len(mermaid)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
