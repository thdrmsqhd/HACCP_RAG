"""조문 검색: 단어 TF-IDF + 제목 가중 + 동의어 확장 (외부 패키지 없이 동작)."""
import math
import re
from collections import Counter

from corpus import load_articles

GOSI = r"C:\Users\gbsong\Documents\HACCP_RAG\corpus\01_공개자료\식품및축산물안전관리인증기준_고시_2026-60호.txt"

STOP = set("이 그 저 것 수 등 및 또는 그러나 하지만 대한 위 한 하는 하여 있다 없 다 를 을 은 는 가 에 서 로 와 과 도 만 부터 까지 우리 회사 공장 어떻게 무엇 언제 누가".split())

# 질문 어휘 → 문서 어휘
SYN = {
    "서류": ["신청서", "제출", "첨부"],
    "신청": ["신청서", "제출"],
    "교육": ["교육훈련", "이수"],
    "심사": ["조사", "평가"],
    "검사": ["조사", "평가"],
    "의무": ["의무적용", "적용대상"],
    "대상": ["적용대상", "적용품목"],
    "주기": ["연 1회", "주기"],
    "기간": ["이내", "기한"],
    "벌칙": ["과태료", "벌칙"],
    "자동": ["자동 기록관리"],
    "스마트해썹": ["자동 기록관리", "스마트해썹"],
    "haccp": ["안전관리인증기준"],
    "haccp팀": ["안전관리인증기준팀", "팀장"],
    "팀": ["팀장", "팀원", "구성"],
    "7원칙": ["위해요소", "중요관리점", "한계기준", "모니터링"],
    "원칙": ["위해요소", "중요관리점", "한계기준"],
    "자주": ["연 1회", "주기"],
    "신규": ["신규교육"],
    "보완": ["보완", "3개월"],
}

SUFFIX = re.compile(r"(에서는|에게서|으로서|으로|에서|에게|까지|부터|보다|처럼|라고|라는|이나|거나|에는|에도|과의|와의|은|는|이|가|을|를|의|에|로|와|과|도|만)$")

# 서술어 어미 — 한국어는 '미흡하면/보완해야/필수항목인데'처럼 활용형이 붙어 색인이 갈라진다.
ENDING = re.compile(
    r"(하나요|까요|인가요|나요|습니까|합니다|했습니다|한다|된다|이다|하면|해야|하여|하며|하고|"
    r"되면|되는|이며|이고|인데|이면|여야|까지|정도|경우|한|할|된|될|이란|란|받)$"
)


def norm(word: str) -> str:
    """어미 → 조사 순으로 최대 두 번 벗긴다. 한 글자로 줄어들면 직전 형태를 유지."""
    w = word
    for _ in range(2):
        for rx in (ENDING, SUFFIX):
            w2 = rx.sub("", w)
            if w2 != w and len(w2) > 1:
                w = w2
                break
    return w


def tokens(text: str) -> list[str]:
    text = re.sub(r"[^\w가-힣]", " ", text.lower())
    out = []
    for w in text.split():
        n = norm(w)
        if len(n) > 1 and n not in STOP:
            out.append(n)
    return out


class Index:
    def __init__(self):
        self.arts = load_articles(GOSI)
        self.docs = [tokens(a["title"] * 3 + " " + a["text"]) for a in self.arts]  # 제목 가중
        df = Counter()
        for d in self.docs:
            for t in set(d):
                df[t] += 1
        self.n = len(self.docs)
        self.idf = {t: math.log((self.n + 1) / (c + 1)) + 1 for t, c in df.items()}
        self.vecs = [self._vec(d) for d in self.docs]

    def _vec(self, toks):
        tf = Counter(toks)
        v = {t: (1 + math.log(c)) * self.idf.get(t, 1.0) for t, c in tf.items()}
        n = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / n for t, x in v.items()}

    def search(self, query: str, topk: int = 3):
        q_toks = tokens(query)
        for w in list(q_toks):
            q_toks.extend(SYN.get(w, []))
        q = self._vec(q_toks)
        scored = []
        for art, vec in zip(self.arts, self.vecs):
            s = sum(w * vec.get(t, 0.0) for t, w in q.items())
            if s > 0:
                scored.append((s, art))
        scored.sort(key=lambda x: -x[0])
        return scored[:topk]


_index = None


def get_index() -> Index:
    global _index
    if _index is None:
        _index = Index()
    return _index


if __name__ == "__main__":
    idx = get_index()
    tests = ["우리 공장이 의무 적용인지 어떻게 판단하나요",
             "인증 신청 서류는 무엇인가요",
             "인증 후 교육은 언제까지 받아야 하나요",
             "팀장이 꼭 해야 하는 일",
             "정기 심사 주기"]
    for q in tests:
        print("Q:", q)
        for s, a in idx.search(q, 3):
            print(f"  {s:.3f} 제{a['num']}조({a['title']})")
        print()
