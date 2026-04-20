"""
실 Gemini로 세션 한 바퀴 돌려보는 대화형 CLI.

사용:
  python scripts/try_session.py resume samples/resume_example.txt [samples/jd_example.txt]
  python scripts/try_session.py essay  samples/essay_example.txt

흐름:
  1) build_graph()로 실제 그래프 컴파일 (Gemini 호출 포함)
  2) /session/start에 해당하는 invoke → 첫 interrupt까지 실행
  3) 면접관 역할로 답변과 액션(a=accept / f=fallback / i=inject)을 CLI에 입력
  4) Command(resume=...)로 재개. 큐가 마르거나 max_rounds 도달까지 반복
  5) 종료 시 요약 출력
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.types import Command

from hiremindset.graph.builder import build_graph


def _usage() -> int:
    print(
        "사용: python scripts/try_session.py <resume|essay> <input.txt> [jd.txt] [--max-rounds N]",
        file=sys.stderr,
    )
    return 2


def _parse_max_rounds(argv: list[str]) -> tuple[list[str], int | None]:
    """argv에서 --max-rounds N 플래그만 분리. 양수 아니면 None."""
    pos, i, max_rounds = [], 0, None
    while i < len(argv):
        if argv[i] == "--max-rounds" and i + 1 < len(argv):
            try:
                max_rounds = int(argv[i + 1])
            except ValueError:
                max_rounds = None
            i += 2
            continue
        pos.append(argv[i])
        i += 1
    return pos, max_rounds


def _print_interrupt(result: dict, state_values: dict) -> dict:
    interrupts = result.get("__interrupt__") or []
    payload = {}
    if interrupts:
        raw = interrupts[0]
        payload = getattr(raw, "value", None) or {}
    strategy = state_values.get("strategy") or {}
    print("\n" + "=" * 60)
    print(f"라운드 {strategy.get('round', '?')} | profile={payload.get('profile')}")
    print(f"Q({payload.get('question_id')}): {payload.get('question')}")
    if payload.get("target_flag_id"):
        flag = next(
            (
                f
                for f in state_values.get("suspicion_flags") or []
                if f["id"] == payload["target_flag_id"]
            ),
            None,
        )
        if flag:
            print(
                f"  ↳ flag={flag['id']} "
                f"category={flag['category']} severity={flag['severity']} "
                f"evidence={flag.get('evidence', '')[:80]}"
            )
    print("=" * 60)
    return payload


def _prompt_action() -> dict:
    action = ""
    while action not in {"a", "f", "i"}:
        action = input("action [a=accept / f=fallback / i=inject] > ").strip().lower()
    answer = input("answer_text > ").strip()
    injected = None
    if action == "i":
        while not (injected or "").strip():
            injected = input("injected_question > ").strip()
    mapping = {"a": "accept", "f": "fallback", "i": "inject"}
    return {
        "answer_text": answer,
        "action": mapping[action],
        "injected_question": injected,
    }


def _print_summary(values: dict) -> None:
    print("\n" + "#" * 60)
    print("세션 종료")
    print("#" * 60)
    print(f"- claims        : {len(values.get('claims') or [])}")
    print(f"- cross_check   : {len(values.get('cross_check') or [])}")
    print(f"- suspicion     : {len(values.get('suspicion_flags') or [])}")
    print(f"- probe_queue   : {len(values.get('probe_queue') or [])}")
    print(f"- pqs           : {len(values.get('probing_questions') or [])}")
    print(f"- turns         : {len(values.get('turns') or [])}")
    print(f"- answer_eval   : {len(values.get('answer_eval') or [])}")
    print(f"- strategy      : {json.dumps(values.get('strategy') or {}, ensure_ascii=False)}")
    print("\n-- probing_questions --")
    for pq in values.get("probing_questions") or []:
        print(f"  [{pq['id']}|r{pq.get('asked_round')}] {pq['text']}")


def main(argv: list[str]) -> int:
    pos, max_rounds = _parse_max_rounds(argv)
    if len(pos) < 3 or pos[1] not in {"resume", "essay"}:
        return _usage()

    kind = pos[1]
    text = Path(pos[2]).read_text(encoding="utf-8")
    jd = Path(pos[3]).read_text(encoding="utf-8") if len(pos) >= 4 else ""

    load_dotenv()

    graph = build_graph()
    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    print(f"thread_id = {thread_id}")

    initial: dict = {
        "documents": {"kind": kind, "raw": text, "paragraphs": []},
        "jd": jd,
    }
    if max_rounds is not None:
        initial["meta"] = {"max_rounds": max_rounds}

    result = graph.invoke(initial, config)

    while "__interrupt__" in result:
        state = graph.get_state(config).values
        _print_interrupt(result, state)
        resp = _prompt_action()
        result = graph.invoke(Command(resume=resp), config)

    _print_summary(graph.get_state(config).values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
