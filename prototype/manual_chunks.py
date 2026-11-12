"""OCR 결과를 '평가항목' 단위로 묶는다.

평가매뉴얼은 항목 번호(예: "15. 청결구역과 ...")로 시작해 세부항목/기준/관리/현장으로 이어진다.
항목 헤더를 경계로 chunk를 만든다.
"""
import json
import pathlib
import re

OCR_DIR = pathlib.Path(r"C:\Users\gbsong\Documents\HACCP_RAG\corpus\01_공개자료\manual_ocr")
OUT = OCR_DIR / "manual_chunks.jsonl"

ITEM_RE = re.compile(r"^\s*(\d{1,3})\s*[.．]\s*(.{4,90})$")
STOP_RE = re.compile(r"^(세부항목|기준\s*\(|관리\s*\(|현장\s*\(|평가\s*\(|PART\s)")

MIN_LEN = 40


def chunk_text(pages: list[tuple[int, str]]):
    chunks, cur = [], None
    for pno, text in pages:
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            m = ITEM_RE.match(line)
            # 항목 번호 + 제목 형태이고, 점수 표기나 페이지 번호가 아니면 새 항목
            if m and "점" not in line[:6]:
                if cur and len(cur["text"]) >= MIN_LEN:
                    chunks.append(cur)
                cur = {"item": m.group(1), "title": m.group(2).strip(), "page": pno, "text": line}
            else:
                if cur is None:
                    cur = {"item": None, "title": "", "page": pno, "text": line}
                else:
                    cur["text"] += "\n" + line
    if cur and len(cur["text"]) >= MIN_LEN:
        chunks.append(cur)
    return chunks


def main():
    pages = []
    for p in sorted(OCR_DIR.glob("page_*.txt")):
        pno = int(re.search(r"page_(\d+)", p.name).group(1))
        pages.append((pno, p.read_text(encoding="utf-8")))
    chunks = chunk_text(pages)
    with open(OUT, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"pages={len(pages)} chunks={len(chunks)} -> {OUT}")
    for c in chunks[:5]:
        print(f"  p{c['page']} item={c['item']} {c['title'][:40]} ({len(c['text'])}자)")


if __name__ == "__main__":
    main()
