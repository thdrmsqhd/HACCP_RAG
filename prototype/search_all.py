"""조문 + 매뉴얼(OCR) 통합 검색.

- 조문은 **항(①②③…) 단위**로 쪼갠다. 한 조문에 숫자 규정이 여러 개 들어 있어
  (예: 제20조 교육시간) 조문 전체를 주면 모델이 첫 사실만 답하기 때문이다.
- 두 소스 모두 BM25. 소스별로 IDF를 따로 계산해 점수를 섞지 않는다.
- 조문은 **질문 용어 포함률 0.34 이상**일 때만 근거로 인정한다(무관 조문 차단).
"""
import json
import math
import os
import re
import sys
from collections import Counter

from corpus import load_articles
from manual_index import OCR_DIR, load_pages

# 매뉴얼 검색 단위. "items" = 미리 구조화한 평가항목(manual_items.py 결과) + 항목에 안 걸린 쪽,
# "pages" = 쪽 단위(이전 방식). 환경변수 HACCP_MANUAL_MODE로 바꿔 비교할 수 있다.
MANUAL_MODE = os.environ.get("HACCP_MANUAL_MODE", "items")  # 홀드아웃 정답 동일(9/15), 무관 근거 혼입 감소 → 기본값 items
MANUAL_ITEMS = OCR_DIR / "manual_items.jsonl"
# A1: 고시 별표 본문(byl_text.py 결과). 고시와 같은 법령 근거로 검색한다.
BYL_CHUNKS = OCR_DIR.parent / "별표" / "byl_chunks.jsonl"
USE_BYL = os.environ.get("HACCP_USE_BYL", "1") == "1"
BYL_REF = re.compile(r"별표\s*(\d+(?:\s*의\s*\d+)?)")


FORM_CHUNKS = OCR_DIR.parent / "서식" / "form_chunks.jsonl"
FORM_REF = re.compile(r"별지\s*제(\d+)호서식")
# 서식은 질문이 그 서식 자체를 물을 때만 붙인다(인증서 기재사항 등). 조문마다 붙이면 잡음이 된다.
FORM_CUE = {"53": re.compile(r"인증서(?:에|의)\s*(?:는\s*)?(?:어떤|무엇|무슨)|기재|적히|적는"), "54": re.compile(r"변경\s*신청서|서식")}


def form_docs():
    if not USE_BYL or not FORM_CHUNKS.exists():
        return []
    out = []
    for l in FORM_CHUNKS.open(encoding="utf-8"):
        r = json.loads(l)
        out.append({"source": "고시", "label": f"시행규칙 별지 제{r['form']}호서식({r['title']})",
                    "article": f"서식{r['form']}", "art_title": r["title"], "text": r["text"],
                    "toks_src": f"별지 제{r['form']}호서식 {r['title']} " + r["text"]})
    return out


def byl_docs():
    if not USE_BYL or not BYL_CHUNKS.exists():
        return []
    out = []
    for l in BYL_CHUNKS.open(encoding="utf-8"):
        r = json.loads(l)
        art = f"별표{r['byl']}"
        out.append({"source": "고시", "label": f"별표 {r['byl']}({r['title']}) {r['part']}",
                    "article": art, "art_title": r["title"], "text": r["text"],
                    "toks_src": f"별표 {r['byl']} {r['title']} " + r["text"]})
    return out


def manual_docs():
    pages = load_pages()
    if MANUAL_MODE != "items" or not MANUAL_ITEMS.exists():
        return [{"source": "매뉴얼", "label": f"평가매뉴얼 p.{p['page']}", "article": "",
                 "text": p["text"], "toks_src": p["text"]} for p in pages]
    items = [json.loads(l) for l in MANUAL_ITEMS.open(encoding="utf-8")]
    covered = {n for it in items for n in range(it["page_start"], it["page_end"] + 1)}
    out = []
    for it in items:
        head = f'{it["topic"]} {it["title"]}'.strip()
        pg = it["page_start"] if it["page_start"] == it["page_end"] else f'{it["page_start"]}-{it["page_end"]}'
        out.append({"source": "매뉴얼", "label": f"평가매뉴얼 p.{pg}", "article": it["id"],
                    "text": head + "\n" + it["text"],
                    # 항목 제목은 검색에서 두 번 센다(본문보다 질문과 직접 맞닿는 문장)
                    "toks_src": head + " " + head + " " + it["text"]})
    for p in pages:                      # 개요·절차 등 항목이 없는 쪽은 쪽 단위로 남긴다
        if p["page"] not in covered:
            out.append({"source": "매뉴얼", "label": f"평가매뉴얼 p.{p['page']}", "article": "",
                        "text": p["text"], "toks_src": p["text"]})
    return out
from retrieve import GOSI, tokens

K1, B = 1.5, 0.75
GOSI_FULL = 12.0    # 조문 BM25 정규화 상한(법령 근거를 우선 노출)
BM25_FULL = 15.0    # 매뉴얼 BM25 정규화 상한
TITLE_BOOST = 2.0   # 질문어가 조문 제목에 걸리면 가중(예: "검증" → 제18조 감독기관의 검증기준)


# 제목에 흔히 나오는 일반어는 판별력이 없어 제외한다.
# 우선순위: 질문의 ‘행위’를 가장 잘 나타내는 말부터
ACTION_TERMS = ("신청", "변경", "반납", "취소", "등록", "평가", "지정")
STRONG_ACTIONS = ("취소", "반납")

TITLE_STOP = {"안전관리인증기준", "교육훈련기관", "haccp", "기준", "등", "관리"}


def title_ratio(qtoks, title_toks) -> float:
    """질문어가 제목 단어를 얼마나 덮는지(0~1).

    짧은 제목이 질문과 정확히 맞으면 가중이 커진다(제24조 ‘교육훈련기관 평가’ vs
    제23의2조 ‘교육훈련기관의 강사전문성 평가 등’).
    """
    cand = [t for t in title_toks if t not in TITLE_STOP]
    if not cand:
        return 0.0
    hit = set()
    for q in qtoks:
        if len(q) < 2 or q in TITLE_STOP:
            continue
        for t in cand:
            if q in t or t in q:
                hit.add(t)
    return len(hit) / len(cand)

MARKS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮"
NUM_UNIT = re.compile(r"\d+\s*(년|개월|일|시간|회|점|%|퍼센트|명|개)")
FAMILY_RE = {
    "기간": re.compile(r"\d+\s*(년|개월|일)"),
    "시간": re.compile(r"\d+\s*시간"),
    "횟수": re.compile(r"\d+\s*회"),
    "점수": re.compile(r"\d+\s*점"),
    "비율": re.compile(r"\d+\s*(?:%|퍼센트)"),
}


# 질문 어휘 → 문서 어휘 (BM25는 표면형이 달라지면 못 찾는다)
SYN_Q = {
    "위생설비": ["세척", "소독", "구비"],
    "설비": ["세척", "소독", "구비", "설치"],
    "갖춰": ["구비", "설치"],
    "갖추": ["구비", "설치"],
    "구비": ["설치"],
    "수칙": ["기준"],
    "복장": ["위생복"],
    "출입구": ["출입"],
    "수립": ["설정", "운영"],
}


def query_tokens(q: str) -> list[str]:
    """질문 토큰 + 동의어 확장(중복 제거)."""
    base = tokens(q)
    out = list(dict.fromkeys(base))
    for t in base:
        for k, vs in SYN_Q.items():
            if k in t:
                out.extend(v for v in vs if v not in out)
    return out


# 질문 유형 — 같은 조문 안에서 어느 항이 답인지가 유형에 따라 달라진다.
ACTOR_RE = re.compile(r"식품의약품안전처장|한국식품안전관리인증원장|지방식품의약품안전청장|농림축산식품부장관|영업자|농업인")
PROC_RE = re.compile(r"신청|제출|심사|조사|평가|명할 수 있|요구할 수 있|통보|지정")
REQ_RE = re.compile(r"요건|기준|갖추|구비|설치|수립")


ACTION_RE = re.compile(r"명할 수 있|시정명령|인증을 취소|개선하도록|보완하도록")


def question_type(q: str) -> str:
    if re.search(r"몇|얼마|며칠|언제까지|기한|최대|최소", q):
        return "기한"
    if re.search(r"어떻게|절차|방법|신청", q):
        return "절차"
    if re.search(r"조치|받나요|받게 되나요", q):
        return "조치"
    if re.search(r"누가|누구|어느 기관|담당|주체", q):
        return "주체"
    if re.search(r"요건|기준|갖추|필요한", q):
        return "요건"
    if re.search(r"무엇인가요|란\?|정의", q):
        return "정의"
    return "일반"


def unit_family(q: str):
    if re.search(r"언제까지|며칠|몇\s*(?:년|개월|일)|기한|기간|얼마나\s*보관", q):
        return "기간"
    if re.search(r"몇\s*시간|시간", q):
        return "시간"
    if re.search(r"몇\s*회|횟수", q):
        return "횟수"
    if re.search(r"몇\s*점|점수", q):
        return "점수"
    if "%" in q or "퍼센트" in q:
        return "비율"
    return None


def split_paragraphs(text: str) -> list[str]:
    """조문 본문을 항(①②③…) 경계로 분리한다. 항 표기가 없으면 원문 그대로."""
    idx = [i for i, ch in enumerate(text) if ch in MARKS]
    if len(idx) < 2:
        return [text]
    parts = []
    for j, start in enumerate(idx):
        end = idx[j + 1] if j + 1 < len(idx) else len(text)
        parts.append(text[start:end].strip())
    head = text[: idx[0]].strip()
    if head:
        if re.fullmatch(r"제\d+조(?:의\d+)?\s*\([^)]*\)", head):
            parts[0] = head + " " + parts[0]      # 제목만 있는 머리는 ①항에 붙인다
        else:
            parts.insert(0, head)
    return parts


def para_mark(text: str) -> str:
    return text[0] if text and text[0] in MARKS else ""


class _BM25:
    """한 소스 안에서만 IDF를 계산하는 BM25."""

    def __init__(self, docs, idx):
        self.idx = idx
        self.tfs = {i: Counter(docs[i]["toks"]) for i in idx}
        self.lens = {i: sum(self.tfs[i].values()) for i in idx}
        self.avg = sum(self.lens.values()) / max(1, len(self.lens))
        df = Counter()
        for i in idx:
            df.update(set(self.tfs[i]))
        n = len(idx)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, qtoks, i):
        s = 0.0
        for w in qtoks:
            f = self.tfs[i].get(w, 0)
            if not f:
                continue
            s += self.idf.get(w, 0) * f * (K1 + 1) / (
                f + K1 * (1 - B + B * self.lens[i] / self.avg))
        return s


class Unified:
    def __init__(self):
        self.docs = []
        for a in load_articles(GOSI):
            parts = split_paragraphs(a["text"])
            for k, body in enumerate(parts):
                if not body.strip():
                    continue
                label = f"제{a['num']}조({a['title']})"
                if len(parts) > 1:
                    label += f" {para_mark(body) or (str(k + 1) + '항')}"
                self.docs.append({
                    "source": "고시",
                    "label": label,
                    "article": f"제{a['num']}조",
                    "text": body,
                    "toks": tokens(f"제{a['num']}조 {a['title']} " + body),
                })
        for d in byl_docs() + form_docs():
            d["toks"] = tokens(d["toks_src"])
            self.docs.append(d)
        for d in manual_docs():
            d["toks"] = tokens(d["toks_src"])
            self.docs.append(d)

        self.gosi_idx = [i for i, d in enumerate(self.docs) if d["source"] == "고시"]
        self.page_idx = [i for i, d in enumerate(self.docs) if d["source"] == "매뉴얼"]
        self.bm_gosi = _BM25(self.docs, self.gosi_idx)
        self.bm_page = _BM25(self.docs, self.page_idx)

        # 조문 단위 인덱스 — 항 단위만 쓰면 탐색형 질문에서 조문 후보가 좁아진다.
        self.art_text = {}
        for i in self.gosi_idx:
            a = self.docs[i]["article"]
            self.art_text.setdefault(a, []).append(self.docs[i]["text"])
        titles = {f"제{a['num']}조": a["title"] for a in load_articles(GOSI)}
        titles.update({self.docs[i]["article"]: self.docs[i]["art_title"]
                       for i in self.gosi_idx if "art_title" in self.docs[i]})
        # 별표는 조문 후보 경쟁에 넣지 않는다(넣으면 선행요건 별표 1이 매뉴얼 질문까지 끌어온다).
        # 별표는 조문이 가리킬 때(A3)와 취소·반납 질문(A4)에서만 붙인다 → answer.retrieve
        self.art_docs = [{"article": a,
                          "toks": tokens(" ".join(t)),
                          "title_toks": tokens(titles.get(a, ""))} for a, t in self.art_text.items()
                         if not a.startswith(("별표", "서식"))]
        self.bm_art = _BM25(self.art_docs, list(range(len(self.art_docs))))

    def rank_articles(self, q, k=2):
        """조문 단위 BM25로 후보 조문을 고른다."""
        qtoks = query_tokens(q)
        qtype = question_type(q)
        # 제목 가중은 절차·조치·주체·요건형에서만. 수치·기한형에 걸면
        # 엉뚱한 조문(예: 실시상황평가표 5점 → 제11조)이 올라온다.
        use_title = qtype in ("절차", "조치", "주체", "요건")
        fam = unit_family(q)
        fam_re = FAMILY_RE.get(fam) if fam else None
        scored = []
        for i, d in enumerate(self.art_docs):
            s = self.bm_art.score(qtoks, i)
            if s > 0 and use_title:
                s *= 1 + TITLE_BOOST * title_ratio(qtoks, d["title_toks"])
                # 질문의 행위어(신청·변경·취소·지정·등록)가 제목에 있으면 우선
                # (예: “어떻게 신청” → 제21조 지정신청, “변경하려면” → 제12조 변경)
                act = next((t for t in ACTION_TERMS if t in q), None)
                if act and any(act in t for t in d["title_toks"]):
                    s *= 1.4
                # A4: 취소·반납은 조문 제목에 잘 안 나온다. 질문의 대상어와 행위어가 본문에 함께 있으면 우선.
                if act in STRONG_ACTIONS:
                    body = " ".join(self.art_text[d["article"]])
                    if act in body and any(t in body for t in tokens(q) if len(t) >= 4):
                        s *= 1.6
            if s > 0 and fam_re is not None and fam_re.search(" ".join(self.art_text[d["article"]])):
                s *= 1.5          # 질문이 묻는 단위의 숫자가 있는 조문 우선
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: -x[0])
        return [(s, self.art_docs[i]["article"]) for s, i in scored[:k]]

    def action_annex(self, q, k=1):
        """A4: 취소·반납 질문 → 행위어와 대상어(4글자 이상 질문어)가 함께 든 별표 조각."""
        act = next((t for t in STRONG_ACTIONS if t in q), None)
        if not act:
            return []
        targets = [t for t in tokens(q) if len(t) >= 4]
        qtoks = tokens(q)
        scored = []
        for i in self.gosi_idx:
            d = self.docs[i]
            if not d["article"].startswith("별표") or act not in d["text"]:
                continue
            if not any(t in d["text"] for t in targets):
                continue
            scored.append((self.bm_gosi.score(qtoks, i), i))
        scored.sort(key=lambda x: -x[0])
        return [(min(1.0, s / GOSI_FULL), self.docs[i]) for s, i in scored[:k] if s > 0]

    def best_annex_chunk(self, art, q):
        """별표 안에서 조각 1개. BM25 점수가 비슷하면 질문의 드문 단어(예: 조도·육안)를
        더 많이 담은 조각을 고른다. (요건형 ‘기준’ 가중이 엉뚱한 ‘검사기준’ 조각을 올리던 문제)"""
        qtoks = list(dict.fromkeys(tokens(q)))
        idf = self.bm_gosi.idf
        total = sum(idf.get(t, 0) for t in qtoks) or 1.0
        best = None
        for i in self.gosi_idx:
            d = self.docs[i]
            if d["article"] != art:
                continue
            bm = self.bm_gosi.score(qtoks, i)
            if bm <= 0:
                continue
            cover = sum(idf.get(t, 0) for t in qtoks if t in d["toks"]) / total
            s = bm * (0.5 + cover)
            is_def = bool(re.search(r"용어의 정의|(?:이)?란\s.{0,200}말한다", d["text"]))
            if question_type(q) == "정의":
                s *= 2.0 if is_def else 1.0   # “○○이란” → 목적·용어 정의 조각
            elif is_def:
                s *= 0.5                      # 요건·절차 질문에 용어 정의 조각이 끼지 않게
            if best is None or s > best[0]:
                best = (s, i)
        return [(min(1.0, best[0] / GOSI_FULL), self.docs[best[1]])] if best else []

    def chunks_of(self, art, q, k=2):
        """한 조문 안에서 질의에 맞는 항을 고른다.

        수량을 묻는 질문(“몇 년/몇 시간/며칠”)은 숫자+단위가 들어간 항을 우선한다.
        숫자는 질문에 없어서 BM25로는 찾히지 않는다(예: “기록 보관” → “2년”).
        """
        qtoks = tokens(q)
        idx = [i for i in self.gosi_idx if self.docs[i]["article"] == art]
        fam = unit_family(q)
        fam_re = FAMILY_RE.get(fam) if fam else None
        qtype = question_type(q)
        # 수량 질문이면 그 단위의 숫자가 든 항만 후보로 좁힌다.
        # (예: “언제까지” → 6개월이 적힌 ③, “몇 시간” → 16시간이 적힌 ④)
        if fam_re is not None:
            with_num = [i for i in idx if fam_re.search(self.docs[i]["text"])]
            if with_num:
                idx = with_num
        pref = {"주체": ACTOR_RE, "절차": PROC_RE, "요건": REQ_RE}.get(qtype)
        scored = []
        for i in idx:
            s = self.bm_gosi.score(qtoks, i)
            text = self.docs[i]["text"]
            if pref is not None and pref.search(text):
                s *= 1.4                     # 유형에 맞는 항 우선
            if qtype == "요건" and re.search(r"각 호|요건|기준|갖추", text):
                s *= 1.5
            if qtype == "절차":
                if re.search(r"신청서|제출", text):
                    s *= 1.6                  # “어떻게 신청” → 신청서 제출 항
                elif re.search(r"심사|평가", text):
                    s *= 1.1
            if qtype == "정의" and re.search(r"(?:이)?란\s|말한다|목적", text):
                s *= 2.0                      # “○○이란 무엇” → 용어 정의·목적 조각
            if qtype == "조치":
                # “어떤 조치를 받나요” → 업소에 대한 조치(시정명령·인증취소) 항.
                # 원장의 ‘통보 의무’ 항은 주체가 달라 오답을 만든다.
                if ACTION_RE.search(text):
                    s *= 1.8
                elif "통보" in text and not ACTION_RE.search(text):
                    s *= 0.7
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: -x[0])
        return [(min(1.0, s / GOSI_FULL), self.docs[i]) for s, i in scored[:k]]

    def rank_pages(self, q, k=2):
        qtoks = query_tokens(q)
        scored = [(self.bm_page.score(qtoks, i), i) for i in self.page_idx]
        scored = [(min(1.0, s / BM25_FULL), i) for s, i in scored if s > 0]
        scored.sort(key=lambda x: -x[0])
        return [(s, self.docs[i]) for s, i in scored[:k]]

    def search(self, q, topk=3):
        """한 목록으로 합쳐서 보고 싶을 때(디버그·데모용)."""
        arts = self.rank_articles(q, 2)
        out = []
        for _, art in arts:
            out.extend(self.chunks_of(art, q, 2))
        out.extend(self.rank_pages(q, 2))
        return out[:topk] if topk else out


if __name__ == "__main__":
    u = Unified()
    qs = sys.argv[1:] or [
        "청결구역 출입 복장 기준",
        "선행요건 평가 점수 기준",
        "인증 신청 서류",
        "교육훈련기관 지정 요건",
    ]
    for q in qs:
        print("Q:", q)
        for s, d in u.search(q, 3):
            snippet = " ".join(d["text"].split())[:110]
            print(f"  {s:.3f} [{d['source']}] {d['label']} — {snippet}")
        print()
