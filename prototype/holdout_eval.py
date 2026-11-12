"""홀드아웃 20문항 채점: 키워드 vs 하이브리드."""
import sys

from holdout import HOLDOUT
from retrieve import get_index


def run(mode, alpha=0.6):
    idx = get_index()
    if mode == "kw":
        search = lambda q: [a["num"] for _, a in idx.search(q, 3)]
    else:
        from hybrid import get_hybrid
        h = get_hybrid(alpha=alpha)
        search = lambda q: [n for _, n, *_ in h.search(q, 3)]

    t1 = t3 = 0
    for q, exp in HOLDOUT:
        nums = search(q)
        h1 = bool(nums) and nums[0] == exp
        h3 = exp in nums
        t1 += h1
        t3 += h3
        print(f"{q[:38]:<40} {exp:>5} {str(nums[:3]):<18} {'O' if h1 else ('△' if h3 else 'X')}")
    n = len(HOLDOUT)
    print(f"\n[{mode}] Top-1 {t1}/{n} ({t1/n*100:.0f}%)  Top-3 {t3}/{n} ({t3/n*100:.0f}%)")
    return t1, t3


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "kw",
        float(sys.argv[2]) if len(sys.argv) > 2 else 0.6)
