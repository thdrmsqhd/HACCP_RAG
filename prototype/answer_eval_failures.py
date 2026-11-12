"""answer_eval 결과에서 실패 사례만 뽑아 사람이 검토할 수 있게 출력."""
import json
import pathlib
import sys

P = pathlib.Path(__file__).with_name("answer_eval_results.jsonl")
rows = [json.loads(l) for l in P.read_text(encoding="utf-8").splitlines() if l.strip()]
bad = [r for r in rows if r["verdict"] != "정답" or not r["cite_ok"]]

print(f"전체 {len(rows)} · 실패 {len(bad)}\n")
for r in bad:
    print("=" * 70)
    print("Q:", r["q"])
    print(f"판정: {r['verdict']} | 필수 {r['keys']} | 발견 {r['found']} | 출처기대 {r['cite_expected']} 일치={r['cite_ok']}")
    print("검색:", [(h["source"], h["label"]) for h in r["hits"]])
    print("답변:", r["answer"].replace("\n", " / ")[:500])
