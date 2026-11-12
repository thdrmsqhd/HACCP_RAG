"""조문 + 매뉴얼 통합 하이브리드 검색 (임베딩 + 키워드)."""
import json
import math
import pathlib

from corpus import load_articles
from manual_index import load_pages
from retrieve import GOSI, tokens

EMB_ART = pathlib.Path(__file__).with_name("embeddings.json")
EMB_PAGE = pathlib.Path(__file__).with_name("manual_embeddings.json")
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _cos(a, b):
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return s / (na * nb)


class UnifiedHybrid:
    def __init__(self, alpha: float = 0.6):
        from fastembed import TextEmbedding
        import json as _json

        self.model = TextEmbedding(model_name=MODEL)
        self.alpha = alpha
        self.docs, self.vecs = [], []

        art_emb = {i["num"]: i["vec"] for i in _json.loads(EMB_ART.read_text(encoding="utf-8"))["items"]}
        for a in load_articles(GOSI):
            self.docs.append({"source": "고시", "label": f"제{a['num']}조({a['title']})", "text": a["text"]})
            self.vecs.append(art_emb[a["num"]])

        page_emb = {i["page"]: i["vec"] for i in _json.loads(EMB_PAGE.read_text(encoding="utf-8"))["items"]}
        for p in load_pages():
            self.docs.append({"source": "매뉴얼", "label": f"평가매뉴얼 p.{p['page']}", "text": p["text"]})
            self.vecs.append(page_emb[p["page"]])

        # 키워드 인덱스
        self.toks = [tokens(d["text"]) for d in self.docs]
        df = {}
        for t in set(t for toks in self.toks for t in toks):
            df[t] = sum(1 for toks in self.toks if t in toks)
        n = len(self.docs)
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}
        self.kw = [self._vec(tk) for tk in self.toks]

    def _vec(self, toks):
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        v = {t: (1 + math.log(c)) * self.idf.get(t, 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def search(self, q, topk=3):
        qv = list(map(float, next(iter(self.model.embed([q])))))
        qk = self._vec(tokens(q))
        scored = []
        for i, d in enumerate(self.docs):
            cos = _cos(qv, self.vecs[i])
            kws = sum(w * self.kw[i].get(t, 0.0) for t, w in qk.items())
            scored.append((self.alpha * cos + (1 - self.alpha) * kws, d))
        scored.sort(key=lambda x: -x[0])
        return scored[:topk]


if __name__ == "__main__":
    import sys
    from manual_eval import MANUAL_Q

    u = UnifiedHybrid()
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        hit1 = hit3 = 0
        for q, exp in MANUAL_Q:
            hits = u.search(q, 5)
            pages = [int(d["label"].split("p.")[1]) for _, d in hits if d["source"] == "매뉴얼"]
            ok1 = pages and pages[0] == exp
            ok3 = exp in pages[:3]
            hit1 += ok1
            hit3 += ok3
            print(f"{q[:34]:<36} 정답 p{exp:<4} 검색 {pages[:3]}  {'O' if ok1 else ('△' if ok3 else 'X')}")
        n = len(MANUAL_Q)
        print(f"\n[통합 하이브리드] Top-1 {hit1}/{n} ({hit1/n*100:.0f}%), Top-3 {hit3}/{n} ({hit3/n*100:.0f}%)")
    else:
        for q in sys.argv[1:] or ["청결구역 출입 복장 기준", "필수항목 0점이면"]:
            print("Q:", q)
            for s, d in u.search(q, 3):
                print(f"  {s:.3f} [{d['source']}] {d['label']} — {' '.join(d['text'].split())[:100]}")
