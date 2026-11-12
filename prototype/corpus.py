"""고시 텍스트를 조문 단위로 파싱한다."""
import re
import pathlib

ART_RE = re.compile(r'제\s*(\d+)\s*조(?:\s*의\s*(\d+))?\s*(?:\(([^)]{1,40})\))?')

# 이 고시의 실제 조문 순서 (본문 기준). 교차 인용(「식품위생법」제48조 등)을 걸러내기 위해 사용.
EXPECTED = ['1', '2', '3', '4', '5', '6', '6의2', '7', '8', '9', '10', '11', '11의2', '11의3',
            '12', '13', '14', '15', '16', '17', '18', '19', '20', '20의2', '21', '22', '22의2',
            '22의3', '23', '23의2', '23의3', '24', '25', '26', '27', '28']


def load_articles(path: str) -> list[dict]:
    raw = pathlib.Path(path).read_text(encoding="utf-8").replace("\u200b", "")
    start = raw.find("제1장 총 칙")
    if start > 0:
        raw = raw[start:]

    marks = []
    for m in ART_RE.finditer(raw):
        num = m.group(1) + (f"의{m.group(2)}" if m.group(2) else "")
        marks.append((m.start(), num, m.group(3)))

    arts, ptr = [], 0
    for i, (pos, num, title) in enumerate(marks):
        if ptr >= len(EXPECTED) or num != EXPECTED[ptr]:
            continue
        end = len(raw)
        if ptr + 1 < len(EXPECTED):
            nxt = EXPECTED[ptr + 1]
            for p2, n2, _ in marks[i + 1:]:
                if n2 == nxt:
                    end = p2
                    break
        body = re.sub(r"\s+", " ", raw[pos:end]).strip()
        arts.append({"num": num, "title": (title or '').strip(), "text": body})
        ptr += 1
    return arts


if __name__ == "__main__":
    p = r"C:\Users\gbsong\Documents\HACCP_RAG\corpus\01_공개자료\식품및축산물안전관리인증기준_고시_2026-60호.txt"
    a = load_articles(p)
    print("조문 수:", len(a), "/ 기대", len(EXPECTED))
    for x in a:
        print(f"제{x['num']}조({x['title']}) - {len(x['text'])}자")
