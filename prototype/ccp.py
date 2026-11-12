"""CCP CSV 조회 (정형 질의)."""
import csv
import re
from collections import Counter

CSV_PATH = r"C:\Users\gbsong\Documents\HACCP_RAG\corpus\01_공개자료\HACCP_중요관리점정보_20250721.csv"


def load_rows():
    with open(CSV_PATH, encoding="cp949") as f:
        return list(csv.DictReader(f))


def norm_ccp(v: str) -> str:
    v = v.replace(" ", "").replace("-", "").replace("－", "").strip().upper()
    m = re.match(r"^CCP(\d*)([BCP])$", v)
    if m:
        return "CCP-" + m.group(1) + m.group(2)
    return v


def norm_hz(v: str) -> str:
    return re.sub(r"\s+", " ", v.replace("\t", "")).strip()


def query(업종: str = None, 공정: str = None, limit: int = 5):
    rows = load_rows()
    hit = [r for r in rows
           if (not 업종 or 업종 in r["업종"]) and (not 공정 or 공정 in r["공정"])]
    if not hit:
        return {"count": 0, "sample": [], "top_ccp": [], "top_hazard": []}
    ccp = Counter(norm_ccp(r["중요관리점"]) for r in hit).most_common(5)
    hz = Counter(norm_hz(r["위해요소"]) for r in hit).most_common(5)
    return {"count": len(hit), "sample": hit[:limit], "top_ccp": ccp, "top_hazard": hz}


if __name__ == "__main__":
    for args in [{"공정": "살균"}, {"공정": "금속검출"}, {"업종": "육가공"}]:
        r = query(**args)
        print(args, "→", r["count"], "건")
        print("  CCP:", r["top_ccp"])
        print("  위해요소:", r["top_hazard"])
