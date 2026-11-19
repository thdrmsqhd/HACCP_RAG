"""고시 별표 PDF → 텍스트 → 검색 단위(byl_chunks.jsonl).

- PDF 텍스트 레이어는 글자 사이 공백이 빠져 있어("이기준은"), 글자 좌표 간격으로 띄어쓰기를 복원한다.
- 별표 안의 번호 제목("1. 목적", "가. …", "Ⅰ. …")에서 자른다. 너무 짧은 조각은 앞 조각에 붙인다.
- 별표 4(실시상황평가표 198쪽)는 평가매뉴얼이 같은 평가항목을 해설과 함께 담고 있어 제외한다.
"""
import json
import pathlib
import re

import pymupdf

DIR = pathlib.Path(r"C:/Users/gbsong/Documents/HACCP_RAG/corpus/01_공개자료/별표")
OUT = DIR / "byl_chunks.jsonl"
SKIP = {"4"}
MAX_CHARS = 1400
MIN_CHARS = 450   # 번호 제목에서 자르되, 이보다 짧으면 다음 문단과 묶는다

TITLE = {
    "1": "선행요건", "4의2": "자동 기록관리 시스템 등록 평가표", "4의3": "글로벌 해썹 등록 요건",
    "4의4": "글로벌 해썹 실시상황평가표", "5": "교육훈련기관 지정기준", "6의2": "교육훈련기관 지정 평가표",
    "6의3": "교육훈련기관 강사 전문성 평가표", "6의5": "교육훈련기관 사후관리 평가표",
    "7": "교육훈련기관 준수사항", "8": "적용품목 심벌",
}
HEAD = re.compile(r"^\s*(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.|\d{1,2}\.|[가-하]\.)\s*\S")


def page_lines(page):
    out = []
    for b in page.get_text("rawdict")["blocks"]:
        for l in b.get("lines", []):
            chars = [c for s in l["spans"] for c in s["chars"]]
            line, prev = "", None
            for c in chars:
                if prev is not None:
                    gap = c["bbox"][0] - prev["bbox"][2]
                    w = (prev["bbox"][2] - prev["bbox"][0]) or 1
                    if gap > 0.25 * w and not line.endswith(" "):
                        line += " "
                line += c["c"]
                prev = c
            if line.strip():
                out.append(line.strip())
    return out


def join_wrapped(lines):
    """PDF 줄바꿈을 문단으로 합친다. 번호 제목 줄에서만 새 문단."""
    paras = []
    for ln in lines:
        if not paras or HEAD.match(ln):
            paras.append(ln)
        else:
            paras[-1] += ("" if paras[-1].endswith(("(", "「")) else " ") + ln
    return [re.sub(r"\s+", " ", p).strip() for p in paras]


def chunk(no, paras):
    chunks, cur = [], []
    for p in paras:
        if cur and (HEAD.match(p) and sum(map(len, cur)) >= MIN_CHARS or sum(map(len, cur)) + len(p) > MAX_CHARS):
            chunks.append(cur)
            cur = []
        cur.append(p)
    if cur:
        if chunks and sum(map(len, cur)) < MIN_CHARS:
            chunks[-1].extend(cur)
        else:
            chunks.append(cur)
    out = []
    for k, c in enumerate(chunks, 1):
        text = "\n".join(c)
        head = next((p[:40] for p in c if HEAD.match(p)), c[0][:40])
        out.append({"byl": no, "title": TITLE.get(no, ""), "part": k, "head": head, "text": text})
    return out


# 평가표형 별표: 줄글로 풀면 “평가 기준 ↔ 비고(처분 등)” 칸 짝이 사라진다.
# (별표 6의5 ‘기타 사항’: 사유마다 지정취소 / 6개월 정지 / 시정이 다르다)
# 이 별표들은 PDF 표 칸을 그대로 읽어 행마다 “기준 → 비고”로 쓴다.
TABLE_BYL = {"6의2", "6의5"}
TABLE_MIN = 250


def _cell(c):
    return re.sub(r"\s+", " ", (c or "").replace("\n", " ")).strip()


def table_items(doc):
    """표 행 → 평가항목별 문단. 병합 셀(빈 칸)은 위 행의 번호·항목을 이어받는다."""
    rows = [[_cell(c) for c in r] for p in doc for t in p.find_tables().tables for r in t.extract()]
    items, cur, section = [], None, ""
    for r in rows:
        r = (r + [""] * 5)[:5]
        no, name, crit, _judge, note = r
        if no in ("구분", "") and name in ("평가내용", "항목"):
            continue                                   # 머리 행
        if no and not no.isdigit() and not any(r[1:]):
            section = no                               # 구분 행(예: 서류심사, 교육훈련기관 평가)
            continue
        if no.isdigit():
            cur = {"no": no, "name": name, "section": section, "lines": []}
            items.append(cur)
        if cur is None or not crit:
            continue
        cur["lines"].append((crit.lstrip("◦ ").strip(), note))
    out = []
    for it in items:
        has_note = any(n for _, n in it["lines"])
        body = []
        for crit, note in it["lines"]:
            if note:
                body.append(f"- {crit} → 비고: {note}")
            elif has_note:
                body.append(f"- {crit} → 비고: 표에 따로 적혀 있지 않음")
            else:
                body.append(f"- {crit}")
        head = f"[{it['section']}] {it['no']}. {it['name']}".strip()
        out.append(head + "\n" + "\n".join(body))
    return out


def table_chunks(no, items):
    chunks, cur = [], []
    for it in items:
        cur.append(it)
        if sum(map(len, cur)) >= TABLE_MIN:
            chunks.append(cur)
            cur = []
    if cur:
        (chunks[-1].extend(cur) if chunks and sum(map(len, cur)) < TABLE_MIN // 2 else chunks.append(cur))
    return [{"byl": no, "title": TITLE.get(no, ""), "part": k, "head": c[0].split("\n")[0][:40],
             "text": "\n".join(c)} for k, c in enumerate(chunks, 1)]


def build():
    rows = []
    for pdf in sorted(DIR.glob("별표_*.pdf")):
        no = pdf.stem.split("_", 1)[1]
        if no in SKIP:
            continue
        doc = pymupdf.open(pdf)
        if no in TABLE_BYL:
            items = table_items(doc)
            (DIR / f"별표_{no}.txt").write_text("\n\n".join(items), encoding="utf-8")
            rows.extend(table_chunks(no, items))
            continue
        lines = [ln for p in doc for ln in page_lines(p)]
        if lines and lines[0].startswith("[별표"):
            lines = lines[1:]                       # "[별표 4의3]" 머리
        paras = join_wrapped(lines)
        (DIR / f"별표_{no}.txt").write_text("\n".join(paras), encoding="utf-8")
        rows.extend(chunk(no, paras))
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


if __name__ == "__main__":
    rows = build()
    from collections import Counter
    print("chunks", len(rows), dict(Counter(r["byl"] for r in rows)))
