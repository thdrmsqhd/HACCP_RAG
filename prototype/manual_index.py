"""OCR 텍스트 보정 + 페이지 단위 인덱스.

easyocr 결과에서 반복적으로 나타나는 오인식만 보수적으로 치환한다.
(과보정 위험이 있으므로 관찰된 패턴만 넣는다.)
"""
import pathlib
import re

OCR_DIR = pathlib.Path(r"C:\Users\gbsong\Documents\HACCP_RAG\corpus\01_공개자료\manual_ocr")

FIXES = [
    ("직업장", "작업장"), ("직업자", "작업자"), ("암도록", "않도록"),
    ("사랑", "사항"), ("교치", "교차"), ("수치올", "수칙을"), ("기준올", "기준을"),
    ("등올", "등을"), ("것올", "것을"), ("수 있도록", "수 있도록"),
    ("점가물", "첨가물"), ("인종", "인증"), ("률", "를"), ("틀", "를"),
    ("잇는", "있는"), ("논", "는"), ("세적", "세척"),
]


def clean(text: str) -> str:
    for a, b in FIXES:
        text = text.replace(a, b)
    text = text.replace("수칙올", "수칙을").replace("확인올", "확인을")
    # 평가 등급 범례(5점 모두 만족 / 4점 만족 / 3점 일부 미흡 / 2점 미흡 / 1점 이하 재검토)
    text = re.sub(r"(점\s*모두\s*)민족", r"\1만족", text)
    text = re.sub(r"(\d점\s*(?:일부\s*)?)미[혼음롭움습급]", r"\1미흡", text)
    text = text.replace("구점 이하 재검토", "1점 이하 재검토")
    text = re.sub(r"올(?=\s)", "을", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pages():
    pages = []
    for p in sorted(OCR_DIR.glob("page_*.txt")):
        pno = int(re.search(r"page_(\d+)", p.name).group(1))
        pages.append({"page": pno, "text": clean(p.read_text(encoding="utf-8"))})
    return pages


if __name__ == "__main__":
    ps = load_pages()
    print("pages", len(ps))
    print(ps[40]["text"][:300])
