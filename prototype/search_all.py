"""조문 + 매뉴얼(OCR) 통합 검색.

- 조문은 **항(①②③…) 단위**로 쪼갠다. 한 조문에 숫자 규정이 여러 개 들어 있어
  (예: 제20조 교육시간) 조문 전체를 주면 모델이 첫 사실만 답하기 때문이다.
- 두 소스 모두 BM25. 소스별로 IDF를 따로 계산해 점수를 섞지 않는다.
- 조문은 **질문 용어 포함률 0.34 이상**일 때만 근거로 인정한다(무관 조문 차단).
"""
import math
import sys
from collections import Counter

from corpus import load_articles
from manual_index import load_pages
from retrieve import GOSI, tokens

K1, B = 1.5, 0.75
COV_MIN = 0.34      # 질문 용어가 조문에 등장하는 비율 하한
GOSI_FULL = 12.0    # 조문 BM25 정규화 상한(법령 근거를 우선 노출)
BM25_FULL = 15.0    # 매뉴얼 BM25 정규화 상한

MARKS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮"


def split_paragraphs(text: str) -> list[str]:
    """조문 본문을 항(①②③…) 경계로 분리한다. 항 표기가 없으면 원문 그대로."""
    idx = [i for i, ch in enumerate(text) if ch in MARKS]
    if len(idx) < 2:
        return [text]
    parts = []
    for j, start in enumerate(idx):
        end = idx[j + 1] if j + 1 < len(idx) else len(text)
        parts.append(text[start:end].strip())
    head = text[: idx[0]].strip()
    if head:
        parts.insert(0, head)
    return parts


def para_mark(text: str) -> str:
    return text[0] if text and text[0] in MARKS else ""


class _BM25:
    """한 소스 안에서만 IDF를 계산하는 BM25."""

    def __init__(self, docs, idx):
        self.idx = idx
        self.tfs = {i: Counter(docs[i]["toks"]) for i in idx}
        self.lens = {i: sum(self.tfs[i].values()) for i in idx}
        self.avg = sum(self.lens.values()) / max(1, len(self.lens))
        df = Counter()
        for i in idx:
            df.update(set(self.tfs[i]))
        n = len(idx)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, qtoks, i):
        s = 0.0
        for w in qtoks:
            f = self.tfs[i].get(w, 0)
            if not f:
                continue
            s += self.idf.get(w, 0) * f * (K1 + 1) / (
                f + K1 * (1 - B + B * self.lens[i] / self.avg))
        return s


class Unified:
    def __init__(self):
        self.docs = []
        for a in load_articles(GOSI):
            parts = split_paragraphs(a["text"])
            for k, body in enumerate(parts):
                if not body.strip():
                    continue
                label = f"제{a['num']}조({a['title']})"
                if len(parts) > 1:
                    label += f" {para_mark(body) or (str(k + 1) + '항')}"
                self.docs.append({
                    "source": "고시",
                    "label": label,
                    "article": f"제{a['num']}조",
                    "text": body,
                    "toks": tokens(f"제{a['num']}조 {a['title']} " + body),
                })
        for p in load_pages():
            self.docs.append({
                "source": "매뉴얼",
                "label": f"평가매뉴얼 p.{p['page']}",
                "article": "",
                "text": p["text"],
                "toks": tokens(p["text"]),
            })

        self.gosi_idx = [i for i, d in enumerate(self.docs) if d["source"] == "고시"]
        self.page_idx = [i for i, d in enumerate(self.docs) if d["source"] == "매뉴얼"]
        self.bm_gosi = _BM25(self.docs, self.gosi_idx)
        self.bm_page = _BM25(self.docs, self.page_idx)

    def search(self, q, topk=3):
        qtoks = tokens(q)
        qset = set(qtoks)

        gosi = []
        for i in self.gosi_idx:
            cov = len(qset & set(self.docs[i]["toks"])) / max(1, len(qset))
            if cov < COV_MIN:
                continue
            s = self.bm_gosi.score(qtoks, i)
            if s > 0:
                gosi.append((min(1.0, s / GOSI_FULL), i))

        pages = []
        for i in self.page_idx:
            s = self.bm_page.score(qtoks, i)
            if s > 0:
                pages.append((min(1.0, s / BM25_FULL), i))

        merged = gosi + pages
        merged.sort(key=lambda x: -x[0])
        return [(s, self.docs[i]) for s, i in merged[:topk]]


if __name__ == "__main__":
    u = Unified()
    qs = sys.argv[1:] or [
        "청결구역 출입 복장 기준",
        "선행요건 평가 점수 기준",
        "인증 신청 서류",
        "교육훈련기관 지정 요건",
    ]
    for q in qs:
        print("Q:", q)
        for s, d in u.search(q, 3):
            snippet = " ".join(d["text"].split())[:110]
            print(f"  {s:.3f} [{d['source']}] {d['label']} — {snippet}")
        print()
