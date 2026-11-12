"""검색 결과를 로컬 LLM(Ollama)으로 3단 답변으로 정리한다.

LLM은 문장화만 담당하고, 근거는 반드시 검색된 출처를 인용하도록 프롬프트에서 강제한다.
"""
import json
import re
import time
import urllib.request

from retrieve import tokens
from search_all import FAMILY_RE, Unified, unit_family

OLLAMA = "http://127.0.0.1:11434/api/generate"
MODEL = "hermes-test-qwen38-9b:q4"
KEEP_ALIVE = "30m"          # 모델 언로드로 인한 30~40초 콜드스타트 방지
NUM_PREDICT = 110           # 3줄 답변 목표. 길게 쓰면 느려지고 잘린다

PROMPT = """/no_think
너는 HACCP을 처음 접하는 품질 담당자를 돕는 안내 도우미다.
아래 [근거]만 사용해 한국어로 답한다. 근거에 없는 내용은 쓰지 않는다.
고시 조문과 평가매뉴얼의 내용이 다르면 **고시를 따른다**.
근거에 나온 용어를 그대로 쓴다(임의로 바꿔 말하지 않는다).
[설명]에는 **누가 무엇을 하는지**를 포함한다. “별표/조문을 확인하세요”처럼 출처만 가리키는 설명은 쓰지 않는다.

출력은 정확히 아래 세 줄만 쓴다. 다른 말은 절대 쓰지 않는다.
[설명] 쉽게 1~2문장. 120자 이내.
[근거] 출처 이름만. 본문을 인용하지 말 것. 예: 제10조, 평가매뉴얼 p.22
[다음 할 일] 한 문장.

질문: {q}

[근거]
{ctx}
"""

_U = None
_CACHE = {}
CACHE_MAX = 200


def index():
    """인덱스는 프로세스당 한 번만 만든다(609쪽 파싱 비용 제거)."""
    global _U
    if _U is None:
        _U = Unified()
    return _U


SUBJECT_NOUNS = {"영업자", "팀장", "팀원", "종업원", "지도관", "검사관", "농업인", "영업장"}

MANUAL_CUE = re.compile(r"실시상황평가표|평가항목|세부항목|현장|조도|위생복|출입구|창|유리|해충|폐기물|게시물|작업장")


def _numbers_by_family(text: str) -> dict[str, set[str]]:
    fam = {"기간": r"(\d+)\s*(년|개월|일)", "시간": r"(\d+)\s*시간", "횟수": r"(\d+)\s*회",
           "점수": r"(\d+)\s*점", "비율": r"(\d+)\s*(?:%|퍼센트)"}
    out = {}
    for k, pat in fam.items():
        nums = {m.group(1) for m in re.finditer(pat, text)}
        if nums:
            out[k] = nums
    return out


def _drop_conflicting_manual(q, picked):
    """법령(고시)과 매뉴얼의 숫자가 충돌하면 매뉴얼 쪽을 근거에서 뺀다.

    예: “검증 결과 보완 기한” → 고시 제18조③ 1개월 vs 매뉴얼 p.18 3개월.
    해설서를 함께 주면 모델이 해설서를 따라 법령과 다른 답을 낸다.
    """
    gosi = [d for _, d in picked if d["source"] == "고시"]
    if not gosi:
        return picked
    gnum = _numbers_by_family(" ".join(d["text"] for d in gosi))
    out = []
    for s, d in picked:
        if d["source"] == "매뉴얼" and gnum:
            mnum = _numbers_by_family(d["text"])
            if any(k in gnum and k in mnum and gnum[k] != mnum[k] for k in gnum):
                continue
        out.append((s, d))
    return out


def retrieve(q, per_source=2, max_articles=2):
    """근거 선별 — 2단계.

    1) 조문 단위 BM25로 후보 조문 최대 2개 (탐색형 질문에서 후보가 좁아지지 않게)
    2) 각 조문 안에서 항 단위 BM25로 최대 2개 (숫자 규정이 여러 항에 흩어져 있어서)
    매뉴얼은 페이지 단위 상위 2쪽. 고시와 숫자가 충돌하는 매뉴얼은 제외.
    """
    u = index()
    gosi, pages = [], []
    for _, art in u.rank_articles(q, max_articles):
        gosi.extend(u.chunks_of(art, q, 2))
    pages = u.rank_pages(q, per_source)

    # 현장·평가표 용어가 질문에 있으면 매뉴얼 우선. 조문이 끼면 모델이 조문을 따라간다.
    # 단, 조문 점수가 확실히 강하면(0.6 이상) 유지한다.
    if gosi and pages and MANUAL_CUE.search(q):
        if gosi[0][0] < 0.75:
            gosi = []

    return _drop_conflicting_manual(q, gosi + pages)


def _snippet(text: str, q: str, limit: int = 150) -> str:
    """근거에서 질문어와 겹치는 문장만 골라 자른다.

    통짜로 넣으면 프롬프트가 2,000~3,000토큰이 되고, 이 모델은 프롬프트 처리도
    ~32 tok/s라 첫 질의가 60초 이상 걸린다(캐시가 없을 때).
    수량 질문은 **질문이 묻는 단위**의 숫자가 든 문장을 우선한다
    (예: “언제까지” → 6개월, “몇 시간” → 16시간).
    """
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    qs = {t for t in tokens(q) if len(t) > 1}
    subjects = {t for t in qs if t in SUBJECT_NOUNS}
    quantity = bool(re.search(r"몇|얼마|며칠|기한|최대|최소|언제까지", q))
    if subjects:
        limit = max(limit, 220)   # 영업자/팀장/팀원별 항목이 한 문장에 몰려 있어 잘리기 쉽다
    family = unit_family(q)
    fam_re = FAMILY_RE.get(family) if family else None
    # "70% 미만"처럼 숫자+부등호가 질문에 있으면 그 표현이 든 문장을 강하게 우선
    q_phrases = [re.sub(r"\s+", "", m.group(0)) for m in re.finditer(r"\d+\s*%\s*(?:미만|이상)", q)]
    sents = re.split(r"(?<=[.다])\s+", flat)
    scored = []
    for i, s in enumerate(sents):
        overlap = sum(1 for t in qs if t in s)
        has_num = bool(NUM_UNIT.search(s))
        flat_s = re.sub(r"\s+", "", s)
        phrase_hit = sum(1 for p in q_phrases if p in flat_s)
        fam_hit = 1 if (fam_re and fam_re.search(s)) else 0
        subj_hit = len(subjects & set(tokens(s)))
        score = (overlap * 2 + (2 if has_num else 0) + (3 if quantity and has_num else 0)
                 + phrase_hit * 4 + fam_hit * 4 + subj_hit * 5)
        scored.append((score, -i, i, s))
    scored.sort(reverse=True)
    out, used = "", set()
    for _, _, i, s in scored:
        if i in used:
            continue
        cand = (out + " " + s).strip()
        if len(cand) > limit:
            continue
        out = cand
        used.add(i)
        if len(out) > limit * 0.7:
            break
    if not out:                      # 질문어가 하나도 안 겹치면 앞부분
        out = flat[:limit]
    out = re.sub(r"\d+$", "", out).rstrip()   # 숫자 중간에서 잘리지 않게
    return out[:limit]


def build_prompt(q, hits):
    ctx = "\n".join(f"- ({d['source']} {d['label']}) " + _snippet(d["text"], q) for _, d in hits)
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
NUM_UNIT = re.compile(r"\d+\s*(년|개월|일|시간|회|점|%|퍼센트|명|개)")


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


def _cache_key(q: str) -> str:
    return re.sub(r"[\s?.,!~]+", "", q).lower()


def ask(q):
    t0 = time.time()
    key = _cache_key(q)
    if key in _CACHE:
        r = dict(_CACHE[key])
        r["timing"] = {"total": round(time.time() - t0, 2), "cache": True,
                       "retrieve": 0, "gen1": 0, "gen2": 0, "retried": False}
        return r
    hits = retrieve(q)
    t1 = time.time()
    if not hits:
        return {"answer": "[설명] 근거를 찾지 못했습니다.\n[근거] 없음\n[다음 할 일] 질문을 업무 단계로 나눠 다시 물어보세요.",
                "hits": [],
                "timing": {"total": round(time.time() - t0, 2), "retrieve": round(t1 - t0, 2),
                           "gen1": 0, "gen2": 0, "retried": False}}
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
    result = {"answer": text, "hits": hits_meta(hits), "unsupported_numbers": bad,
              "timing": {"retrieve": round(t1 - t0, 2), "gen1": round(gen1, 2),
                         "gen2": round(gen2, 2), "retried": retried,
                         "total": round(time.time() - t0, 2)}}
    if len(_CACHE) >= CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = result
    return result


def _generate(prompt):
    t0 = time.time()
    req = urllib.request.Request(OLLAMA, data=_payload(prompt, False),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        out = json.loads(r.read().decode())
    return _clean(out.get("response", "")), round(time.time() - t0, 2)


def ask_stream(q):
    """(첫 조각 전 메타, 텍스트 조각 제너레이터)를 돌려준다.

    캐시에 있으면 즉시 반환한다(스트리밍 경로도 캐시 사용).
    """
    key = _cache_key(q)
    if key in _CACHE:
        r = _CACHE[key]
        return {"hits": r["hits"], "cache": True}, iter([r["answer"]])

    hits = retrieve(q)
    meta = {"hits": hits_meta(hits)}
    if not hits:
        meta["answer"] = "[설명] 근거를 찾지 못했습니다.\n[근거] 없음\n[다음 할 일] 질문을 업무 단계로 나눠 다시 물어보세요."
        return meta, iter(())

    def gen():
        req = urllib.request.Request(OLLAMA, data=_payload(build_prompt(q, hits), True),
                                     headers={"Content-Type": "application/json"})
        started, acc, buf = False, "", ""
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
                acc += buf
                out, buf = buf, ""
                yield out
        text = _clean(acc)
        if len(_CACHE) >= CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = {"answer": text, "hits": meta["hits"], "unsupported_numbers": []}
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
