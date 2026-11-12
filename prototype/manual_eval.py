"""매뉴얼 전용 평가셋 (1~60쪽 범위 기준). 정답은 페이지 번호."""
from search_all import Unified

MANUAL_Q = [
    ("실시상황평가표 5점 척도는 어떻게 되나요?", 21),
    ("필수항목이 0점이면 어떻게 되나요?", 22),
    ("청결구역과 일반구역은 어떻게 구분하나요?", 29),
    ("작업장 출입구에 꼭 있어야 하는 설비는?", 32),
    ("평가매뉴얼의 세부구성 3가지는 무엇인가요?", 20),
    ("청결구역 출입 복장 기준은 어떻게 정하나요?", 41),
    ("즉시 인증취소 항목을 위반하면 어떻게 되나요?", 22),
    ("작업장 출입구에 게시해야 하는 것은?", 32),
]


def main():
    u = Unified()
    hit1 = hit3 = 0
    for q, exp in MANUAL_Q:
        hits = u.search(q, 5)
        pages = [int(d["label"].split("p.")[1]) for _, d in hits if d["source"] == "매뉴얼"]
        top = pages[0] if pages else None
        ok1 = top == exp
        ok3 = exp in pages[:3]
        hit1 += ok1
        hit3 += ok3
        print(f"{q[:34]:<36} 정답 p{exp:<4} 검색 {pages[:3]}  {'O' if ok1 else ('△' if ok3 else 'X')}")
    n = len(MANUAL_Q)
    print(f"\n매뉴얼 검색 Top-1 {hit1}/{n} ({hit1/n*100:.0f}%), Top-3 {hit3}/{n} ({hit3/n*100:.0f}%)")


if __name__ == "__main__":
    main()
