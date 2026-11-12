"""검색 결과를 로컬 LLM(Ollama)으로 3단 답변으로 정리한다.

LLM은 문장화만 담당하고, 근거는 반드시 검색된 출처를 인용하도록 프롬프트에서 강제한다.
"""
import json
import re
import time
import urllib.request

from search_all import Unified

OLLAMA = "http://127.0.0.1:11434/api/generate"
MODEL = "hermes-test-qwen38-9b:q4"
KEEP_ALIVE = "30m"          # 모델 언로드로 인한 30~40초 콜드스타트 방지
NUM_PREDICT = 110           # 3줄 답변 목표. 길게 쓰면 느려지고 잘린다

PROMPT = """/no_think
너는 HACCP을 처음 접하는 품질 담당자를 돕는 안내 도우미다.
아래 [근거]만 사용해 한국어로 답한다. 근거에 없는 내용은 쓰지 않는다.

출력은 정확히 아래 세 줄만 쓴다. 다른 말은 절대 쓰지 않는다.
[설명] 쉽게 1~2문장. 120자 이내.
[근거] 출처 이름만. 본문을 인용하지 말 것. 예: 제10조, 평가매뉴얼 p.22
[다음 할 일] 한 문장.

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


def retrieve(q, per_source=2, max_articles=2):
    """근거 선별 — 2단계.

    1) 조문 단위 BM25로 후보 조문 최대 2개 (탐색형 질문에서 후보가 좁아지지 않게)
    2) 각 조문 안에서 항 단위 BM25로 최대 2개 (숫자 규정이 여러 항에 흩어져 있어서)
    매뉴얼은 페이지 단위 상위 2쪽.
    """
    u = index()
    picked = []
    for _, art in u.rank_articles(q, max_articles):
        picked.extend(u.chunks_of(art, q, 2))
    picked.extend(u.rank_pages(q, per_source))
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
    # 모델이 [근거] 줄에 조문 본문을 길게 인용하면 잘라낸다(토큰·가독성)
    out = []
    for line in raw.split("\n"):
        if line.startswith("[근거]"):
            body = " ".join(line[4:].split())
            if len(body) > 100:
                body = body[:100].rstrip() + "…"
            line = "[근거] " + body
        out.append(line)
    return "\n".join(out).strip()


def hits_meta(hits):
    return [{"source": d["source"], "label": d["label"], "score": round(s, 3)} for s, d in hits]


def _context_text(hits):
    return " ".join(" ".join(d["text"].split()) for _, d in hits)


NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(년|개월|일|시간|회|점|%|퍼센트|명|개|쪽|차)")


def unsupported_numbers(answer_text: str, ctx: str) -> list[str]:
    """답변에 나온 '숫자+단위' 중 근거 문장에 없는 것. 환각 탐지용."""
    flat_ctx = re.sub(r"\s+", "", ctx)
    out = []
    for num, unit in NUM_RE.findall(answer_text):
        pat = f"{num}{unit}"
        alt = f"{num}{'퍼센트' if unit == '%' else '%' if unit == '퍼센트' else unit}"
        if pat.replace(" ", "") not in flat_ctx and alt not in flat_ctx:
            out.append(pat)
    return out


GUARD = """
주의: 아래 [근거]에 없는 숫자는 절대 쓰지 않는다. 근거에 숫자가 없으면 "근거에 없음"이라고 쓴다.
"""


def ask(q):
    t0 = time.time()
    hits = retrieve(q)
    t1 = time.time()
    if not hits:
        return {"answer": "[설명] 근거를 찾지 못했습니다.\n[근거] 없음\n[다음 할 일] 질문을 업무 단계로 나눠 다시 물어보세요.", "hits": []}
    ctx = build_prompt(q, hits)
    text, gen1 = _generate(ctx)
    t2 = time.time()
    bad = unsupported_numbers(text, _context_text(hits))
    gen2 = 0.0
    retried = False
    if bad:  # 근거에 없는 수치 → 한 번 더 엄격 모드로 재생성
        retried = True
        text, gen2 = _generate(GUARD + ctx)
        bad = unsupported_numbers(text, _context_text(hits))
    return {"answer": text, "hits": hits_meta(hits), "unsupported_numbers": bad,
            "timing": {"retrieve": round(t1 - t0, 2), "gen1": round(gen1, 2),
                       "gen2": round(gen2, 2), "retried": retried,
                       "total": round(time.time() - t0, 2)}}


def _generate(prompt):
    t0 = time.time()
    req = urllib.request.Request(OLLAMA, data=_payload(prompt, False),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        out = json.loads(r.read().decode())
    return _clean(out.get("response", "")), round(time.time() - t0, 2)


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
