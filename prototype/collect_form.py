"""식품위생법 시행규칙 별지 서식(HACCP 인증서 등) 수집 → 텍스트.

고시는 “별지 제53호서식의 인증서에 해당사항을 기재”라고만 쓰고, 무엇을 적는지는 서식에 있다.
흐름: 서식 본문 조회(lsBylContentsInfoR.do, bylSeq) → pdfFlSeq → flDownload.do
결과: corpus/01_공개자료/서식/서식_<번호>.pdf, .txt, form_chunks.jsonl
"""
import json, pathlib, re, time, urllib.parse, urllib.request

import pymupdf

LSI = "289161"          # 식품위생법 시행규칙 [시행 2026. 9. 1.] 총리령 제2150호
BASE = "https://www.law.go.kr/LSW/"
OUT = pathlib.Path(r"C:/Users/gbsong/Documents/HACCP_RAG/corpus/01_공개자료/서식")
UA = {"User-Agent": "Mozilla/5.0"}
# bylSeq: lsBylInfoR.do(lsiSeq, bylClsCd=110202) 서식 목록에서 확인(2026-11-19)
FORMS = {
    "53": ("18443303", "식품안전관리인증기준(HACCP)적용업소 인증서"),
    "54": ("18443305", "식품안전관리인증기준(HACCP)적용업소 인증사항 변경신청서"),
}


def _post(url, data):
    req = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode(), headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "ignore")


def collect():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for no, (seq, title) in FORMS.items():
        html = _post(BASE + "lsBylContentsInfoR.do", {"bylSeq": seq, "lsiSeq": LSI})
        fl = re.search(r'id="pdfFlSeq"[^>]*value="(\d+)"', html).group(1)
        with urllib.request.urlopen(urllib.request.Request(BASE + f"flDownload.do?flSeq={fl}", headers=UA), timeout=120) as r:
            data = r.read()
        assert data.startswith(b"%PDF"), no
        pdf = OUT / f"서식_{no}.pdf"
        pdf.write_bytes(data)
        doc = pymupdf.open(pdf)
        # 첫 쪽(한글 서식)만 쓴다. 뒤쪽은 영문 서식·작성 방법
        lines = [re.sub(r"\s+", " ", l).strip() for l in doc[0].get_text().split("\n")]
        lines = [l for l in lines if l and not re.fullmatch(r"\d+㎜×\d+㎜.*", l)]
        text = "\n".join(lines)
        (OUT / f"서식_{no}.txt").write_text(text, encoding="utf-8")
        rows.append({"form": no, "title": title, "bylSeq": seq, "pdfFlSeq": fl, "text": text})
        time.sleep(1)
    with (OUT / "form_chunks.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


if __name__ == "__main__":
    for r in collect():
        print(r["form"], r["title"], len(r["text"]))
