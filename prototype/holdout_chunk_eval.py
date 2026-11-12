"""홀드아웃 20문항을 '조문 항 단위' 검색으로 채점 (실제 앱과 같은 경로).

앱은 근거로 **서로 다른 조문 2개**를 쓰므로, Top-3 대신 Top-1/Top-2(조문 중복 제거)로 본다.
"""
import answer
from holdout import HOLDOUT

TOP_N = 2


def art_of(label: str) -> str:
    return label.split("(")[0].strip()


def distinct_articles(hits):
    out = []
    for _, d in hits:
        if d["source"] != "고시":
            continue
        a = art_of(d["label"])
        if a not in out:
            out.append(a)
    return out


def main():
    ok1 = ok2 = 0
    for q, exp in HOLDOUT:
        arts = distinct_articles(answer.retrieve(q))
        hit1 = bool(arts) and arts[0] == f"제{exp}조"
        ok1 += hit1
        ok2 += f"제{exp}조" in arts[:TOP_N]
        print(f"{'O' if hit1 else 'X'} {q[:38]:<40} 기대 제{exp}조 / 실제 {arts[:TOP_N]}")
    n = len(HOLDOUT)
    print(f"\n[chunk-BM25] Top-1 {ok1}/{n} ({ok1/n*100:.0f}%)  Top-{TOP_N} {ok2}/{n} ({ok2/n*100:.0f}%)")


if __name__ == "__main__":
    main()
