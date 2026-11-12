"""섹션 필터 적용 검색 (하이브리드 + 섹션 부스트)."""
import sys

from holdout import HOLDOUT
from retrieve import get_index
from sections import intent_section, section_of


def search_kw_section(q, topk=3):
    idx = get_index()
    sec = intent_section(q)
    scored = []
    for s, a in idx.search(q, topk=len(idx.arts)):
        boost = 1.0
        if sec:
            a_sec, a_type = section_of(a["num"])
            # 의도 섹션이면 부스트, 교육훈련기관 조항은 영업자 질의에서 감점
            if a_sec == sec:
                boost = 1.35
            elif a_type == "기관" and sec != "교육훈련":
                boost = 0.6
        scored.append((s * boost, a["num"]))
    scored.sort(key=lambda x: -x[0])
    return [n for _, n in scored[:topk]]


def search_hybrid_section(q, topk=3, alpha=0.6):
    from hybrid import get_hybrid
    h = get_hybrid(alpha=alpha)
    sec = intent_section(q)
    out = []
    for s, num, title, cos, kws in h.search(q, topk=len(h.items)):
        boost = 1.0
        if sec:
            a_sec, a_type = section_of(num)
            if a_sec == sec:
                boost = 1.35
            elif a_type == "기관" and sec != "교육훈련":
                boost = 0.6
        out.append((s * boost, num))
    out.sort(key=lambda x: -x[0])
    return [n for _, n in out[:topk]]


def search_hard(q, topk=3):
    """의도 섹션만 남기는 하드 필터."""
    idx = get_index()
    sec = intent_section(q)
    if not sec:
        return [a["num"] for _, a in idx.search(q, topk)]
    cand = [a for a in idx.arts if section_of(a["num"])[0] == sec]
    scored = []
    qv = idx._vec(__import__("retrieve").tokens(q))
    for a, vec in zip(idx.arts, idx.vecs):
        if a not in cand:
            continue
        sc = sum(w * vec.get(t, 0.0) for t, w in qv.items())
        scored.append((sc, a["num"]))
    scored.sort(key=lambda x: -x[0])
    return [n for _, n in scored[:topk]]


def report(search, label):
    t1 = t3 = 0
    for q, exp in HOLDOUT:
        nums = search(q)
        h1 = bool(nums) and nums[0] == exp
        h3 = exp in nums
        t1 += h1
        t3 += h3
        print(f"{q[:38]:<40} {exp:>5} {str(nums):<20} {'O' if h1 else ('△' if h3 else 'X')}")
    n = len(HOLDOUT)
    print(f"\n[{label}] Top-1 {t1}/{n} ({t1/n*100:.0f}%)  Top-3 {t3}/{n} ({t3/n*100:.0f}%)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "kw"
    if mode == "kw":
        report(search_kw_section, "키워드+섹션부스트")
    elif mode == "hard":
        report(search_hard, "키워드+섹션하드필터")
    else:
        report(search_hybrid_section, "하이브리드+섹션부스트")
