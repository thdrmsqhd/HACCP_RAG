"""고시 별표 원문 수집 (law.go.kr 공개 행정규칙 첨부).

흐름: 별표 본문 조회(admRulBylContentsInfoR.do, bylSeq) → hidden input pdfFlSeq/hanFlSeq
→ flDownload.do?flSeq=… 로 PDF 받기. 결과: corpus/01_공개자료/별표/별표_<번호>.pdf
"""
import json, pathlib, re, sys, time, urllib.parse, urllib.request

ADM = "2100000283858"   # 식품 및 축산물 안전관리인증기준 고시 제2026-60호
BASE = "https://www.law.go.kr/LSW/"
OUT = pathlib.Path(r"C:/Users/gbsong/Documents/HACCP_RAG/corpus/01_공개자료/별표")
UA = {"User-Agent": "Mozilla/5.0"}

# bylSeq는 고시 본문 페이지의 별표 목록 링크에서 확인(2026-11-19)
BYL = {
    "1": "3285449", "4": "3285455", "4의2": "3285457", "4의3": "3285459", "4의4": "3285461",
    "5": "3285463", "6의2": "3285467", "6의3": "3285469", "6의5": "3285473", "7": "3285475", "8": "3285477",
}


def post(url, data):
    req = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode(), headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "ignore")


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read(), r.headers.get("Content-Type", "")


def main(keys):
    OUT.mkdir(parents=True, exist_ok=True)
    log = []
    for no in keys:
        seq = BYL[no]
        html = post(BASE + "admRulBylContentsInfoR.do", {"bylSeq": seq})
        m = re.search(r'id="pdfFlSeq"[^>]*value="(\d+)"', html)
        if not m:
            log.append({"no": no, "bylSeq": seq, "status": "no pdf"})
            continue
        data, ctype = get(BASE + f"flDownload.do?flSeq={m.group(1)}")
        if not data.startswith(b"%PDF"):
            log.append({"no": no, "bylSeq": seq, "status": f"not pdf ({ctype})"})
            continue
        path = OUT / f"별표_{no}.pdf"
        path.write_bytes(data)
        log.append({"no": no, "bylSeq": seq, "pdfFlSeq": m.group(1), "bytes": len(data), "status": "ok"})
        time.sleep(1)
    (OUT / "수집_로그.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in log:
        print(r)


if __name__ == "__main__":
    main(sys.argv[1:] or list(BYL))
