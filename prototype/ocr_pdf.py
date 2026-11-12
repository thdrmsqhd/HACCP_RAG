"""평가매뉴얼(스캔 PDF) OCR → 페이지별 텍스트.

사용: uv run --with easyocr --with pymupdf python ocr_pdf.py [시작페이지] [끝페이지]
출력: corpus/01_공개자료/manual_ocr/page_0001.txt, .../manual_ocr.jsonl
"""
import json
import pathlib
import sys
import time

import pymupdf

PDF = pathlib.Path(r"C:\Users\gbsong\Documents\HACCP_RAG\corpus\01_공개자료\HACCP_평가심사매뉴얼_20250130.pdf")
OUT = pathlib.Path(r"C:\Users\gbsong\Documents\HACCP_RAG\corpus\01_공개자료\manual_ocr")
TMP = pathlib.Path(r"C:\Users\gbsong\AppData\Local\Temp")
SCALE = 2.2


def main(start: int = 1, end: int | None = None):
    import easyocr

    OUT.mkdir(parents=True, exist_ok=True)
    reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
    doc = pymupdf.open(PDF)
    end = end or len(doc)
    jsonl = OUT / "manual_ocr.jsonl"
    t0 = time.time()
    done = 0

    with open(jsonl, "a", encoding="utf-8") as jf:
        for pno in range(start, end + 1):
            page = doc[pno - 1]
            png = TMP / f"ocr_p{pno:04d}.png"
            page.get_pixmap(matrix=pymupdf.Matrix(SCALE, SCALE)).save(png)
            boxes = reader.readtext(str(png), detail=1, paragraph=False)
            lines = [{"text": txt, "conf": round(float(conf), 3), "box": [[int(x), int(y)] for x, y in b]}
                     for b, txt, conf in boxes]
            txt_path = OUT / f"page_{pno:04d}.txt"
            txt_path.write_text("\n".join(l["text"] for l in lines), encoding="utf-8")
            jf.write(json.dumps({"page": pno, "lines": lines}, ensure_ascii=False) + "\n")
            jf.flush()
            png.unlink(missing_ok=True)
            done += 1
            if done % 10 == 0:
                el = time.time() - t0
                print(f"{pno}/{end} pages, {el:.0f}s, {el/done:.1f}s/page", flush=True)
    print(f"done {start}-{end} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    e = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(s, e)
