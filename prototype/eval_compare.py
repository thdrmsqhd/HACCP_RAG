"""하이브리드 검색 평가 (키워드 vs 임베딩+키워드)."""
import sys

from eval import QSET

from retrieve import get_index


def run(mod, alpha=None):
    idx = get_index()
    if mod == "kw":
        search = lambda q: [(s, a["num"]) for s, a in idx.search(q, 3)]
    else:
        from hybrid import get_hybrid
        h = get_hybrid(alpha=alpha or 0.65)
        search = lambda q: [(s, n) for s, n, *_ in h.search(q, 3)]

    t1 = t3 = 0
    for q, exp in QSET:
        nums = [n for _, n in search(q)]
        t1 += bool(nums) and nums[0] == exp
        t3 += exp in nums
    return t1, t3, len(QSET)


if __name__ == "__main__":
    mod = sys.argv[1] if len(sys.argv) > 1 else "kw"
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.65
    t1, t3, n = run(mod, alpha)
    print(f"{mod} alpha={alpha}: Top-1 {t1}/{n} ({t1/n*100:.0f}%), Top-3 {t3}/{n} ({t3/n*100:.0f}%)")
