"""새 홀드아웃 15문항 — 튜닝에 쓰지 않은 영역(조직·절차·행정).

채점은 자동 토큰 매칭이 아니라 **사람이 답변과 인용 근거를 대조**한다.
결과는 answer_holdout2_results.jsonl 에 남긴다.
"""
import json
import pathlib
import time

import answer

import sys
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
OUT = pathlib.Path(__file__).with_name(f"answer_holdout2_results{TAG}.jsonl")

QUESTIONS = [
    "HACCP 팀장의 책무는 무엇인가요?",
    "자동 기록관리 시스템 등록은 어떻게 하나요?",
    "글로벌 해썹 등록 요건은 무엇인가요?",
    "인증받은 사항을 변경하려면 어떻게 해야 하나요?",
    "인증서는 언제 반납해야 하나요?",
    "조사·평가 결과 부적합이면 어떤 조치를 받나요?",
    "지도관은 무슨 일을 하나요?",
    "교육훈련기관으로 지정받으려면 어떻게 신청하나요?",
    "교육훈련기관은 어떻게 평가받나요?",
    "교육훈련기관 지정이 취소되는 경우는 언제인가요?",
    "HACCP 인증업소가 받는 우대조치는 무엇이 있나요?",
    "HACCP 팀은 어떻게 구성하나요?",
    "글로벌 해썹이란 무엇인가요?",
    "인증서에는 어떤 사항이 적히나요?",
    "HACCP 적용업소 표시·광고는 어떻게 할 수 있나요?",
]


def main():
    rows = []
    with OUT.open("w", encoding="utf-8") as f:
        for i, q in enumerate(QUESTIONS, 1):
            t0 = time.time()
            r = answer.ask(q)
            row = {"q": q, "answer": r["answer"],
                   "hits": [(h["source"], h["label"]) for h in r["hits"]],
                   "coverage": r.get("coverage"),
                   "sec": round(time.time() - t0, 1)}
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{i:2d}/15 {row['sec']:5.1f}s | {q}")
            print("   ", r["answer"].replace("\n", " / ")[:180])
    print("\n평균", round(sum(r["sec"] for r in rows) / len(rows), 1), "초")
    print("결과:", OUT)


if __name__ == "__main__":
    main()
