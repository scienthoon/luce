"""
demo.py — jevlocal v0 동작 확인.

실행:
    python demo.py
    python demo.py --model Qwen/Qwen2.5-1.5B-Instruct --n-perm 4
"""

from __future__ import annotations

import argparse
import json
import time

from jevlocal import Choice, JevLocal, Noul, Score, answers_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="jevlocal v0 demo")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="HuggingFace 모델 이름")
    parser.add_argument("--device", default=None, help="cuda | cpu (기본: 자동)")
    parser.add_argument("--temperature", type=float, default=1.0, help="캘리브레이션 temperature")
    parser.add_argument("--n-perm", type=int, default=1, help="선택지 순서를 섞어 평균낼 횟수 (위치 편향 완화)")
    args = parser.parse_args()

    print(f"loading {args.model} ...")
    t0 = time.time()
    jev = JevLocal(
        model_name=args.model,
        device=args.device,
        temperature=args.temperature,
        n_perm=args.n_perm,
    )
    print(f"loaded in {time.time() - t0:.1f}s on {jev.device}")

    # ---- 예제 1: 스마트홈 (Jev 클라이언트 README의 예제) -------------------
    state = "The front door has been unlocked for 40 minutes and nobody is home."
    questions = {
        "warn": Noul("Someone should be warned about this."),
        "area": Choice(
            "Which area is this about?",
            {
                "security": "Doors, locks, alarms",
                "climate": "Heating and ventilation",
            },
        ),
        "urgency": Score(
            "How urgent is it?",
            ["Ignore", "Today", "Right now"],
        ),
    }

    t0 = time.time()
    answers = jev.ask(state, questions)
    elapsed_ms = (time.time() - t0) * 1000.0

    print("\n=== example 1: smart home ===")
    print(json.dumps(answers_to_dict(answers), ensure_ascii=False, indent=2))
    print(f"({elapsed_ms:.0f} ms for {len(questions)} questions)")

    # ---- 예제 2: 고객 문의 라우팅 (구조화 state) -------------------------
    state2 = {
        "channel": "email",
        "customer_tier": "gold",
        "subject": "Charged twice for my order #4821",
        "body": (
            "Hi, I placed one order last week but my card shows two charges of $89.99. "
            "Please fix this as soon as possible."
        ),
    }
    questions2 = {
        "queue": Choice(
            "Which support queue should handle this?",
            {
                "billing": "Payments, refunds, duplicate charges",
                "shipping": "Delivery status, lost packages",
                "technical": "App or website bugs",
                "general": "Anything else",
            },
        ),
        "priority": Score(
            "How should this ticket be prioritized?",
            ["Low", "Normal", "High", "Critical"],
        ),
        "refund_review": Noul("This ticket requires a manual refund review by a human."),
        "angry": Noul("The customer sounds angry."),
    }

    t0 = time.time()
    answers2 = jev.ask(state2, questions2)
    elapsed_ms = (time.time() - t0) * 1000.0

    print("\n=== example 2: support routing ===")
    print(json.dumps(answers_to_dict(answers2), ensure_ascii=False, indent=2))
    print(f"({elapsed_ms:.0f} ms for {len(questions2)} questions)")

    # ---- 예제 3: 다지선다 검증 (수학 객관식, 역추론 감 잡기용) -------------
    state3 = "Problem: A rectangle has perimeter 20 and area 24. What is the length of its longer side?"
    questions3 = {
        "answer": Choice(
            "Which option is the correct answer?",
            {
                "a": "4",
                "b": "5",
                "c": "6",
                "d": "8",
            },
        ),
        "is_hard": Noul("This problem requires multi-step reasoning to solve."),
    }

    t0 = time.time()
    answers3 = jev.ask(state3, questions3)
    elapsed_ms = (time.time() - t0) * 1000.0

    print("\n=== example 3: math MCQ (correct: c = 6) ===")
    print(json.dumps(answers_to_dict(answers3), ensure_ascii=False, indent=2))
    print(f"({elapsed_ms:.0f} ms for {len(questions3)} questions)")


if __name__ == "__main__":
    main()
