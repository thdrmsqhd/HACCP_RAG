"""생성 변동 + 응답 시간 측정: 새 홀드아웃 15문항을 같은 조건으로 N회 반복.

- 같은 프롬프트(temperature=0, seed 고정)에 답이 글자 단위로 같은지 문항별로 센다.
- 다른 작업을 같이 돌리지 않은 상태에서 잰 시간이 ‘깨끗한’ 응답 시간이다.
결과: answer_repeat_results.jsonl (run, q, answer, sec, timing)
"""
import json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import answer as A
from answer_holdout2 import QUESTIONS

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OUT = pathlib.Path(__file__).with_name("answer_repeat_results.jsonl")

with OUT.open("w", encoding="utf-8") as f:
    for run in range(1, N + 1):
        for q in QUESTIONS:
            A._CACHE.clear()          # 앱의 답변 캐시를 비워야 매번 실제로 생성한다(첫 시도는 2·3회차가 캐시로 0초)
            t = time.time()
            r = A.ask(q)
            assert not r["timing"].get("cache"), "cache hit"
            row = {"run": run, "q": q, "answer": r["answer"], "sec": round(time.time() - t, 1), "timing": r["timing"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(run, row["sec"], q[:30], flush=True)

rows = [json.loads(l) for l in OUT.open(encoding="utf-8")]
by_q = {}
for r in rows:
    by_q.setdefault(r["q"], []).append(r)
stable = sum(len({x["answer"] for x in v}) == 1 for v in by_q.values())
print(f"\n문항별 답 일치: {stable}/{len(by_q)} 문항이 {N}회 모두 같은 답")
for q, v in by_q.items():
    print(f"  {len({x['answer'] for x in v})}종 | " + " / ".join(f"{x['sec']}" for x in v) + f" | {q[:28]}")
for run in range(1, N + 1):
    s = [r["sec"] for r in rows if r["run"] == run]
    print(f"run {run}: 평균 {sum(s) / len(s):.1f}초")
