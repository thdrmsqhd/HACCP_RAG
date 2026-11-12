"""하이브리드 검색: 임베딩 코사인 + 키워드 TF-IDF.

임베딩은 embeddings.json(사전 계산)을 사용하고, 질의만 실시간 임베딩한다.
실행에는 fastembed가 필요하다: uv run --with fastembed python ...
"""
import json
import math
import pathlib

from retrieve import get_index

EMB = pathlib.Path(__file__).with_name("embeddings.json")


def _cos(a, b):
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return s / (na * nb)


# 교육훈련기관 지정·평가 조항: 영업자(사업장) 의무가 아니라 기관 요건 → 영업자 질의에서 감점
EDU_INST = {"21", "22", "22의2", "22의3", "23", "23의2", "23의3", "24", "25"}
OPERATOR_CUES = ["업소", "공장", "사업장", "인증 후", "인증받", "도입", "이수", "우리"]


class Hybrid:
    def __init__(self, alpha: float = 0.65):
        from fastembed import TextEmbedding

        data = json.loads(EMB.read_text(encoding="utf-8"))
        self.model_name = data["model"]
        self.items = data["items"]
        self.alpha = alpha
        self.kw = get_index()
        self.model = TextEmbedding(model_name=self.model_name)

    def search(self, query: str, topk: int = 3, scope_filter: bool = True):
        qv = list(map(float, next(iter(self.model.embed([query])))))
        kw_hits = {a["num"]: s for s, a in self.kw.search(query, topk=len(self.kw.arts))}
        kmax = max(kw_hits.values()) if kw_hits else 1.0
        scope_filter = scope_filter and any(c in query for c in OPERATOR_CUES)

        scored = []
        for it in self.items:
            cos = _cos(qv, it["vec"])
            kws = kw_hits.get(it["num"], 0.0) / kmax if kmax else 0.0
            s = self.alpha * cos + (1 - self.alpha) * kws
            if scope_filter and it["num"] in EDU_INST:
                s *= 0.5
            scored.append((s, it["num"], it["title"], cos, kws))
        scored.sort(key=lambda x: -x[0])
        return scored[:topk]


_h = None


def get_hybrid(alpha: float = 0.65) -> "Hybrid":
    global _h
    if _h is None:
        _h = Hybrid(alpha)
    return _h
