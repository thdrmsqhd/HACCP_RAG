"""매뉴얼 검색 단위 비교: 쪽(pages) vs 평가항목(items). LLM 없이 검색만 본다.

정답: answer_eval.py 매뉴얼 16문항의 기대 쪽. 상위 2개 매뉴얼 결과의 쪽 범위가 기대 쪽을 덮으면 적중.
"""
import os, re, subprocess, sys, json

CODE = r'''
import sys, json, time, re
sys.path.insert(0, ".")
from answer_eval import CASES
import search_all as S
t = time.time(); u = S.Unified(); build = time.time() - t
hit = 0; rows = []; lens = []; t2 = time.time()
man = [(q, int(src[2:])) for q, _, src in CASES if src.startswith("p.")]
for q, page in man:
    res = u.rank_pages(q, 2)
    ok = False; got = []
    for _, d in res:
        m = re.search(r"p\.(\d+)(?:-(\d+))?", d["label"])
        a = int(m.group(1)); b = int(m.group(2) or a)
        got.append(d["label"]); lens.append(len(d["text"]))
        ok = ok or a <= page <= b
    hit += ok
    rows.append((ok, page, got, q[:24]))
q_ms = (time.time() - t2) / len(man) * 1000
print(json.dumps({"mode": S.MANUAL_MODE, "manual_docs": len(u.page_idx), "build_s": round(build, 2),
                  "query_ms": round(q_ms, 1), "top2_hit": f"{hit}/{len(man)}",
                  "avg_doc_chars": sum(lens) // len(lens)}, ensure_ascii=False))
for r in rows:
    if not r[0]: print("   MISS", r[1], r[2], r[3])
'''
for mode in ["pages", "items"]:
    env = dict(os.environ, HACCP_MANUAL_MODE=mode, PYTHONIOENCODING="utf-8")
    out = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, encoding="utf-8", env=env,
                         cwd=r"C:/Users/gbsong/Documents/HACCP_RAG/prototype")
    print(out.stdout.strip(), out.stderr[-600:])
