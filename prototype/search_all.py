"""조문 + 매뉴얼(OCR) 통합 검색.

소스별로 검색기를 분리한다(측정 근거: 00_진행기록 §14, §16).
- 고시 36조문: TF-IDF 코사인 (짧고 정제된 법령 문장)
- 매뉴얼 609쪽: BM25 (OCR 잡음·업종별 반복에 강함)
"""
import math
import sys
from collections import Counter

from corpus import load_articles
from manual_index import load_pages
from retrieve import GOSI, tokens

K1, B = 1.5, 0.75
COS_MIN = 0.08      # 이보다 낮은 조문 유사도는 무관한 것으로 본다(실측 분포 기반)
COS_FULL = 0.25     # 조문 코사인 상한(정규화 기준)
BM25_FULL = 15.0    # 매뉴얼 BM25 상한(정규화 기준)


class Unified:
    def __init__(self):
        self.docs = []
        for a in load_articles(GOSI):
            self.docs.append({
                "source": "고시",
                "label": f"제{a['num']}조({a['title']})",
                "text": a["text"],
                "toks": tokens(f"제{a['num']}조 {a['title']} " + a["text"]),
            })
        for p in load_pages():
            self.docs.append({
                "source": "매뉴얼",
                "label": f"평가매뉴얼 p.{p['page']}",
                "text": p["text"],
                "toks": tokens(p["text"]),
            })

        # 고시: TF-IDF 코사인
        df = Counter()
        for d in self.docs:
            for t in set(d["toks"]):
                df[t] += 1
        n = len(self.docs)
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}
        self.vecs = [self._vec(d["toks"]) for d in self.docs]

        # 매뉴얼: BM25 (해당 소스만으로 통계 산출)
        self.page_idx = [i for i, d in enumerate(self.docs) if d["source"] == "매뉴얼"]
        self.tfs = {i: Counter(self.docs[i]["toks"]) for i in self.page_idx}
        self.lens = {i: sum(self.tfs[i].values()) for i in self.page_idx}
        self.avg = sum(self.lens.values()) / max(1, len(self.lens))
        pdf = Counter()
        for i in self.page_idx:
            pdf.update(set(self.tfs[i].keys()))
        m = len(self.page_idx)
        self.pidf = {t: math.log(1 + (m - c + 0.5) / (c + 0.5)) for t, c in pdf.items()}

    def _vec(self, toks):
        tf = Counter(toks)
        v = {t: (1 + math.log(c)) * self.idf.get(t, 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def _bm25(self, qtoks, i):
        s = 0.0
        for w in qtoks:
            f = self.tfs[i].get(w, 0)
            if not f:
                continue
            s += self.pidf.get(w, 0) * f * (K1 + 1) / (
                f + K1 * (1 - B + B * self.lens[i] / self.avg))
        return s

    def search(self, q, topk=3):
        qtoks = tokens(q)
        qv = self._vec(qtoks)

        gosi = []
        for i, d in enumerate(self.docs):
            if d["source"] != "고시":
                continue
            s = sum(w * self.vecs[i].get(t, 0.0) for t, w in qv.items())
            if s >= COS_MIN:                      # 무관한 조문 배제
                gosi.append((min(1.0, s / COS_FULL), i))
        pages = [(self._bm25(qtoks, i), i) for i in self.page_idx]
        pages = [(min(1.0, s / BM25_FULL), i) for s, i in pages if s > 0]

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
