"""검색 결과를 로컬 LLM(Ollama)으로 3단 답변으로 정리한다.

LLM은 문장화만 담당하고, 근거는 반드시 검색된 출처를 인용하도록 프롬프트에서 강제한다.
"""
import json
import urllib.request

from search_all import Unified

OLLAMA = "http://127.0.0.1:11434/api/generate"
MODEL = "hermes-test-qwen38-9b:q4"
KEEP_ALIVE = "30m"          # 모델 언로드로 인한 30~40초 콜드스타트 방지
NUM_PREDICT = 160           # 3줄 답변은 약 100~120토큰

PROMPT = """/no_think
너는 HACCP을 처음 접하는 품질 담당자를 돕는 안내 도우미다.
아래 [근거]만 사용해 한국어로 답한다. 근거에 없는 내용은 쓰지 않는다.

출력은 정확히 아래 세 줄만 쓴다. 다른 말은 절대 쓰지 않는다.
[설명] 쉽게 2~3문장
[근거] 출처만 나열 (예: 제10조, 평가매뉴얼 p.22)
[다음 할 일] 지금 할 일 1~2개

질문: {q}

[근거]
{ctx}
"""

_U = None


def index():
    """인덱스는 프로세스당 한 번만 만든다(609쪽 파싱 비용 제거)."""
    global _U
    if _U is None:
        _U = Unified()
    return _U


def retrieve(q, per_source=2, rel=0.45):
    """소스별 상위 k개. 단, 해당 소스 1위의 45% 미만이면 버린다(무관 근거 방지)."""
    u = index()
    hits = u.search(q, len(u.docs))
    picked, count, top = [], {"고시": 0, "매뉴얼": 0}, {}
    for s, d in hits:
        src = d["source"]
        top.setdefault(src, s)
        if s < top[src] * rel:
            continue
        if count[src] < per_source:
            picked.append((s, d))
            count[src] += 1
        if all(v >= per_source for v in count.values()):
            break
    return picked


def build_prompt(q, hits):
    ctx = "\n".join(f"- ({d['source']} {d['label']}) " + " ".join(d["text"].split())[:600] for _, d in hits)
    return PROMPT.format(q=q, ctx=ctx)


def _payload(prompt, stream):
    return json.dumps({
        "model": MODEL, "prompt": prompt, "stream": stream, "raw": False,
        "think": False, "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0.1, "num_predict": NUM_PREDICT},
    }).encode()


def _clean(raw: str) -> str:
    raw = raw.strip()
    if "[설명]" in raw:
        raw = raw[raw.index("[설명]"):]
    return raw


def hits_meta(hits):
    return [{"source": d["source"], "label": d["label"], "score": round(s, 3)} for s, d in hits]


def ask(q):
    hits = retrieve(q)
    if not hits:
        return {"answer": "[설명] 근거를 찾지 못했습니다.\n[근거] 없음\n[다음 할 일] 질문을 업무 단계로 나눠 다시 물어보세요.", "hits": []}
    req = urllib.request.Request(OLLAMA, data=_payload(build_prompt(q, hits), False),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        out = json.loads(r.read().decode())
    return {"answer": _clean(out.get("response", "")), "hits": hits_meta(hits)}


def ask_stream(q):
    """(첫 조각 전 메타, 텍스트 조각 제너레이터)를 돌려준다. 첫 토큰까지의 지연을 줄인다."""
    hits = retrieve(q)
    meta = {"hits": hits_meta(hits)}
    if not hits:
        meta["answer"] = "[설명] 근거를 찾지 못했습니다.\n[근거] 없음\n[다음 할 일] 질문을 업무 단계로 나눠 다시 물어보세요."
        return meta, iter(())

    def gen():
        req = urllib.request.Request(OLLAMA, data=_payload(build_prompt(q, hits), True),
                                     headers={"Content-Type": "application/json"})
        started, buf = False, ""
        with urllib.request.urlopen(req, timeout=240) as r:
            for line in r:
                if not line.strip():
                    continue
                obj = json.loads(line.decode())
                piece = obj.get("response", "")
                if not piece:
                    continue
                buf += piece
                if not started:
                    if "[설명]" not in buf:
                        continue
                    buf = buf[buf.index("[설명]"):]
                    started = True
                out, buf = buf, ""
                yield out
    return meta, gen()


if __name__ == "__main__":
    import sys
    for q in sys.argv[1:] or ["인증 후 교육은 언제까지 받아야 하나요?", "작업장 출입구에 꼭 있어야 하는 설비는?"]:
        print("=" * 60)
        print("Q:", q)
        meta, gen = ask_stream(q)
        for piece in gen:
            print(piece, end="", flush=True)
        print("\n출처:", [(h["source"], h["label"]) for h in meta["hits"]])
