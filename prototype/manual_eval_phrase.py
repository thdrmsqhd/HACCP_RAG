"""매뉴얼 검색 품질 — 페이지 번호가 아니라 '정답 문구 포함' 기준.

업종별 반복이 많아 페이지 번호 일치만으로는 실제 유용성을 반영하지 못한다.
각 질문에 답이 되려면 포함되어야 하는 핵심 어구를 정의하고, 상위 페이지가 그 어구를 담고 있는지 본다.
"""
from search_all import Unified

CASES = [
    ("실시상황평가표 5점 척도는 어떻게 되나요?", ["5점", "4점", "일부"]),
    ("필수항목이 0점이면 어떻게 되나요?", ["필수항목", "0점", "부적합"]),
    ("청결구역과 일반구역은 어떻게 구분하나요?", ["청결구역", "일반구역", "분리"]),
    ("작업장 출입구에 꼭 있어야 하는 설비는?", ["세척", "건조", "소독"]),
    ("평가매뉴얼의 세부구성 3가지는 무엇인가요?", ["기준", "관리", "현장"]),
    ("청결구역 출입 복장 기준은 어떻게 정하나요?", ["청결구역", "복장", "기준"]),
    ("즉시 인증취소 항목을 위반하면 어떻게 되나요?", ["즉시", "인증취소", "부적합"]),
    ("작업장 출입구에 게시해야 하는 것은?", ["출입구", "게시", "복장"]),
]


def score(search, label, k=3, source=None):
    ok1 = ok3 = 0
    for q, keys in CASES:
        docs = search(q, k * 4 if source else k)
        if source:
            docs = [(s, d) for s, d in docs if d["source"] == source][:k]
        texts = [" ".join(d["text"].split()) for _, d in docs]
        contain = [sum(1 for key in keys if key in t) >= max(2, len(keys) - 1) for t in texts]
        ok1 += bool(contain) and contain[0]
        ok3 += any(contain)
        print(f"{q[:34]:<36} {['O' if x else 'X' for x in contain]}")
    n = len(CASES)
    print(f"\n[{label}] 문구 포함 Top-1 {ok1}/{n} ({ok1/n*100:.0f}%), Top-3 {ok3}/{n} ({ok3/n*100:.0f}%)")


if __name__ == "__main__":
    u = Unified()
    score(lambda q, k: u.search(q, k), "매뉴얼 BM25, 609쪽", source="매뉴얼")
