"""평가매뉴얼 OCR(쪽 단위 txt) → 평가항목 단위 구조화 (manual_items.jsonl / manual_items.md)

질문 시점에는 OCR을 하지 않는다. 이 스크립트를 한 번 돌려 결과 파일을 만들고,
검색은 결과 파일을 읽는다.

항목 경계 규칙 (OCR 원문 관찰 기반)
- 평가항목은 배점 표시 "(0~N점)"로 끝나는 진술문이다.
  (4부 HACCP관리 기준은 "(0~5)"처럼 '점'이 빠져 있다)
- 진술문 시작은 배점 표시 앞 10줄 안에서 가장 가까운 "번호" 줄이다.
  번호 형태: "6 작업장의…", "37. 영업자는…", "1"(혼자 한 줄)
- 항목 본문은 다음 항목 시작 직전까지(쪽이 넘어가도 이어 붙인다).
- 업종 구간은 표지 쪽의 "3-1", "4-2" 같은 표시로 정한다(목차 OCR 쪽수는 일부 틀림).
- 쪽 머리말("PART 3. …", "HACCP 평가(심사) 매뉴얼")과 쪽 끝 쪽번호 줄은 본문에서 뺀다.
  (혼자 있는 "1" 줄이 항목 번호인 경우가 있어 숫자 줄은 쪽의 마지막 줄일 때만 쪽번호로 본다)
- 번호 OCR 오인식("11"→"71" 등)은 앞뒤 번호가 연속이면 가운데 번호로 바로잡는다.
"""
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import manual_index as M  # noqa: E402

OUT_DIR = M.OCR_DIR
OUT_JSONL = OUT_DIR / "manual_items.jsonl"
OUT_MD = OUT_DIR / "manual_items.md"

SCORE = re.compile(r"\(\s*0\s*[~～\-]\s*(\d+)\s*(?:점)?\s*\)")
NUM_LINE = re.compile(r"^\s*(\d{1,2})(?:\.|\s|$)")
SECTION_MARK = re.compile(r"^\s*([3-6])\s*-\s*(\d)\s*$")
HEADER = re.compile(r"^\s*(PAR[TI]\b.*|HACCP\s*평가\s*\(심사\)\s*매뉴[얼업월].*)\s*$")
PAGE_NO = re.compile(r"^\s*\d{1,3}\s*$")
LOOKBACK = 10

PART_NAME = {
    "3": "선행요건관리 기준 해설",
    "4": "HACCP관리 기준 해설",
    "5": "소규모업소 등 선행요건관리 기준 해설",
    "6": "소규모업소 등 HACCP관리 기준 해설",
}


def section_title(text: str) -> str:
    """표지 쪽에서 업종 이름 줄을 모은다(머리말·표시 줄 제외)."""
    keep = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s or SECTION_MARK.match(s) or s.startswith("PAR") or "기준 해설" in s:
            continue
        if s in ("소규모업소", "등", "소규모업소 등") or s.startswith("("):
            continue
        keep.append(s)
    return " · ".join(keep)[:160]


def build():
    pages = M.load_pages()
    # 1) 줄 단위로 펼치되 머리말·쪽번호는 버리고, 쪽 번호와 구간을 줄마다 붙인다
    lines = []          # (page, section, text)
    sections = {}       # section -> title
    cur = None
    for p in pages:
        raw = p["text"].split("\n")
        mark = next((SECTION_MARK.match(l) for l in raw[:6] if SECTION_MARK.match(l)), None)
        if mark:
            cur = f"{mark.group(1)}-{mark.group(2)}"
            sections[cur] = section_title(p["text"])
            continue            # 표지 쪽은 본문이 없다
        while raw and not raw[-1].strip():
            raw.pop()
        if raw and PAGE_NO.match(raw[-1]):
            raw.pop()                       # 쪽 끝 쪽번호
        for pos, ln in enumerate(raw):
            if HEADER.match(ln):
                continue
            lines.append((p["page"], cur, ln.rstrip(), pos))

    # 2) 배점 표시 위치 → 항목 시작 줄 찾기
    starts = []
    for i, (_, sec, ln, _pos) in enumerate(lines):
        m = SCORE.search(ln)
        if not m or sec is None:
            continue
        for j in range(i, max(-1, i - LOOKBACK), -1):
            if NUM_LINE.match(lines[j][2]) and lines[j][1] == sec:
                if not starts or starts[-1][0] < j:
                    starts.append((j, i, int(m.group(1))))
                break

    # 3) 시작~다음 시작 직전까지를 한 항목으로 묶는다
    items = []
    for k, (j, i, score_max) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        sec = lines[j][1]
        # 다음 구간으로 넘어가면 거기서 끊는다
        for t in range(j, end):
            if lines[t][1] != sec:
                end = t
                break
        no = NUM_LINE.match(lines[j][2]).group(1)
        statement = " ".join(l[2].strip() for l in lines[j:i + 1])
        statement = re.sub(r"^\s*\d{1,2}\.?\s*", "", statement)
        statement = SCORE.sub("", statement).strip()
        # 진술문 바로 앞의 짧은 줄은 소제목(예: "배수 및 배관")
        topic = ""
        # 소제목은 쪽 맨 위(머리말 바로 아래)에만 온다. 그 밖의 짧은 줄은 앞 항목의 꼬리다
        if j > 0 and lines[j - 1][1] == sec and lines[j - 1][0] == lines[j][0] and lines[j - 1][3] <= 3:
            prev = lines[j - 1][2].strip()
            if 0 < len(prev) <= 20 and not NUM_LINE.match(prev) and "점" not in prev:
                topic = prev
        body = "\n".join(l[2] for l in lines[i + 1:end]).strip()
        # 본문 끝의 다음 항목 소제목 한 줄은 다음 항목 몫이다
        if k + 1 < len(starts):
            nj = starts[k + 1][0]
            if nj - 1 >= 0 and lines[nj - 1][3] <= 3 and nj - 1 < end + 1:
                # 다음 항목의 쪽 맨 위 소제목 줄은 다음 항목 몫이다
                tail = body.split("\n")[-1].strip() if body else ""
                if 0 < len(tail) <= 20 and "점" not in tail:
                    body = "\n".join(body.split("\n")[:-1]).strip()
        pg = [l[0] for l in lines[j:end]]
        items.append({
            "id": f"M{sec}-{int(no):02d}",
            "part": sec.split("-")[0],
            "part_name": PART_NAME.get(sec.split("-")[0], ""),
            "section": sec,
            "section_title": sections.get(sec, ""),
            "item_no": int(no),
            "topic": topic,
            "title": statement,
            "score_max": score_max,
            "page_start": min(pg),
            "page_end": max(pg),
            "text": body,
        })

    # 번호 OCR 오인식 보정: 앞뒤가 연속(n-1, ?, n+1)인데 가운데만 어긋나면 n으로 바꾼다
    for k in range(1, len(items) - 1):
        a, b, c = items[k - 1], items[k], items[k + 1]
        if a["section"] == b["section"] == c["section"] and c["item_no"] == a["item_no"] + 2 \
                and b["item_no"] != a["item_no"] + 1:
            b["ocr_item_no"] = b["item_no"]
            b["item_no"] = a["item_no"] + 1

    # id는 구간 안 순번으로 매긴다. 4부는 소제목마다 번호가 1부터 다시 시작해 인쇄 번호가 겹친다.
    seq = {}
    for it in items:
        seq[it["section"]] = seq.get(it["section"], 0) + 1
        it["id"] = f'M{it["section"]}-{seq[it["section"]]:03d}'
    return items, sections


def write(items, sections):
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    md = ["# HACCP 평가(심사) 매뉴얼 — 평가항목 구조화본", "",
          "원본: 2025 HACCP 평가(심사) 매뉴얼(식품의약품안전처, 공개자료) OCR. "
          "OCR 오인식이 남아 있을 수 있으니 정확한 문구는 원문 PDF로 확인한다.", ""]
    cur = None
    for it in items:
        if it["section"] != cur:
            cur = it["section"]
            md += ["", f'## {cur} {it["part_name"]} — {sections.get(cur, "")}', ""]
        head = f'### {it["item_no"]}. {it["title"]} (0~{it["score_max"]}점)'
        md += [head, f'- 쪽: {it["page_start"]}–{it["page_end"]}' + (f' · 소제목: {it["topic"]}' if it["topic"] else ""), "",
               it["text"], ""]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    items, sections = build()
    write(items, sections)
    print("sections", len(sections), "items", len(items))
    print("written", OUT_JSONL, OUT_MD)
