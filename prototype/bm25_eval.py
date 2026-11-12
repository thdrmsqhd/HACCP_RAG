"""BM25 vs TF-IDF 비교 — 고시 홀드아웃 20문항 + 매뉴얼 8문항."""
import math
import re
from collections import Counter

from corpus import load_articles
from retrieve import GOSI
from manual_index import load_pages
from holdout import HOLDOUT
from manual_eval_phrase import CASES


from retrieve import tokens as _rtok
def tok(t):
    return _rtok(t)


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.tfs = [Counter(tok(d["text"])) for d in docs]
        self.lens = [sum(t.values()) for t in self.tfs]
        self.avg = sum(self.lens) / max(1, len(self.lens))
        df = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def score(self, q, i):
        s = 0.0
        for w in tok(q):
            f = self.tfs[i].get(w, 0)
            if not f:
                continue
            s += self.idf.get(w, 0) * f * (self.k1 + 1) / (
                f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
        return s

    def search(self, q, k=5):
        scored = [(self.score(q, i), d) for i, d in enumerate(self.docs)]
        scored.sort(key=lambda x: -x[0])
        return [(s, d) for s, d in scored[:k] if s > 0]


def eval_manual(docs, label):
    bm = BM25(docs)
    ok1 = ok3 = 0
    for q, keys in CASES:
        tops = bm.search(q, 3)
        texts = [" ".join(d["text"].split()) for _, d in tops]
        c = [sum(1 for k in keys if k in t) >= max(2, len(keys) - 1) for t in texts]
        ok1 += bool(c) and c[0]
        ok3 += any(c)
    print(f"[매뉴얼 {label}] Top-1 {ok1}/{len(CASES)}, Top-3 {ok3}/{len(CASES)}")


def eval_gosi(docs, label):
    bm = BM25(docs)
    ok1 = ok3 = 0
    for q, exp in HOLDOUT:
        tops = bm.search(q, 3)
        ids = [d.get("num") for _, d in tops]
        ok1 += bool(ids) and ids[0] == exp
        ok3 += exp in ids
    print(f"[고시 {label}] Top-1 {ok1}/{len(HOLDOUT)}, Top-3 {ok3}/{len(HOLDOUT)}")


if __name__ == "__main__":
    arts = load_articles(GOSI)
    pages = load_pages()
    eval_gosi(arts, "BM25")
    eval_manual(pages, "BM25")
