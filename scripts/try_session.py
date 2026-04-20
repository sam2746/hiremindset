"""
실 Gemini로 세션 한 바퀴 돌려보는 대화형 CLI.

사용:
  python scripts/try_session.py resume samples/resume_example.txt [samples/jd_example.txt]
  python scripts/try_session.py essay  samples/essay_example.txt

Phase 1 이후 한 라운드는 두 번의 interrupt로 나뉜다:
  1) collect_answer  : 답변 텍스트만 입력
  2) decide_action   : AI 평가를 본 뒤 a/f/i 결정
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


def _interrupt_payload(result: dict) -> dict:
    interrupts = result.get("__interrupt__") or []
    if not interrupts:
        return {}
    raw = interrupts[0]
    value = getattr(raw, "value", None) or {}
    if isinstance(value, tuple):
        value = value[0] if value else {}
    return value if isinstance(value, dict) else {}


def _print_question(payload: dict, state_values: dict) -> None:
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


def _print_ai_eval(payload: dict) -> None:
    ev = payload.get("ai_eval") or {}
    print("\n" + "-" * 60)
    print("AI 평가")
    print(f"  answer      : {payload.get('answer_text')}")
    print(
        f"  specificity={ev.get('specificity')} consistency={ev.get('consistency')} "
        f"epistemic={ev.get('epistemic')} hedge={ev.get('refusal_or_hedge')}"
    )
    print(f"  suggest     : {ev.get('suggest')}")
    print("-" * 60)


def _prompt_answer() -> dict:
    answer = input("answer_text > ").strip()
    return {"answer_text": answer}


def _prompt_decision() -> dict:
    action = ""
    while action not in {"a", "f", "i"}:
        action = input("action [a=accept / f=fallback / i=inject] > ").strip().lower()
    mapping = {"a": "accept", "f": "fallback", "i": "inject"}
    payload: dict = {"action": mapping[action]}
    if action == "i":
        injected = ""
        while not injected.strip():
            injected = input("injected_question > ").strip()
        payload["injected_question"] = injected
    return payload


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
        payload = _interrupt_payload(result)
        phase = payload.get("type")

        if phase == "collect_answer":
            _print_question(payload, state)
            resp = _prompt_answer()
        elif phase == "decide_action":
            _print_ai_eval(payload)
            resp = _prompt_decision()
        else:
            print(f"(알 수 없는 phase: {phase}) — 빈 resume으로 진행")
            resp = {}

        result = graph.invoke(Command(resume=resp), config)

    _print_summary(graph.get_state(config).values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
