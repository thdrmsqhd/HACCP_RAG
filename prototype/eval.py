"""평가셋 자동 채점 (03_평가셋_초안.md 기반)."""
from retrieve import get_index

QSET = [
    ("우리 공장이 HACCP 의무 적용인지 어떻게 판단하나요?", "4"),
    ("신규 공장이라 1년 매출이 없으면 어떻게 하나요?", "4"),
    ("식품제조가공업의 선행요건 항목은 무엇인가요?", "5"),
    ("HACCP 7원칙 순서를 알려주세요", "6"),
    ("HACCP팀은 꼭 만들어야 하나요?", "9"),
    ("팀장이 반드시 해야 하는 일은 무엇인가요?", "9"),
    ("인증 신청 때 어떤 서류를 내나요?", "10"),
    ("현장조사 보완 요구를 받으면 기간은 얼마인가요?", "11"),
    ("인증 받은 뒤 얼마나 자주 심사를 받나요?", "15"),
    ("인증 후 교육은 언제까지 들어야 하나요?", "20"),
]


def evaluate(search, label=""):
    t1 = t3 = 0
    rows = []
    for q, exp in QSET:
        nums = search(q)
        hit1 = bool(nums) and nums[0] == exp
        hit3 = exp in nums
        t1 += hit1
        t3 += hit3
        rows.append((q, exp, nums[:3], "O" if hit1 else ("△" if hit3 else "X")))
    for q, exp, got, v in rows:
        print(f"{q[:40]:<42} {exp:>4} {str(got):<16} {v}")
    print(f"\n{label} Top-1 {t1}/{len(QSET)} ({t1/len(QSET)*100:.0f}%), "
          f"Top-3 {t3}/{len(QSET)} ({t3/len(QSET)*100:.0f}%)")
    return t1, t3


if __name__ == "__main__":
    idx = get_index()
    evaluate(lambda q: [a["num"] for _, a in idx.search(q, 3)], label="[키워드]")
