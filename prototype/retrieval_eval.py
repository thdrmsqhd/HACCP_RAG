"""검색만 따로 잰다 — 모델 호출 없이 몇 초.

왜: 답이 틀렸을 때 '근거를 못 가져온 것'인지 '근거는 왔는데 모델이 못 쓴 것'인지 구분해야 한다.
지표:
- corpus  : 정답 문장이 코퍼스에 있는가(없으면 그 문항의 gold 패턴이 잘못된 것)
- delivered: answer.retrieve()가 모델에 넘기는 근거 안에 정답 문장이 들어왔는가
- rank    : 후보를 넓혔을 때(조문 상위 20 + 매뉴얼 상위 20) 정답이 몇 번째에 처음 나오는가.
            delivered=X인데 rank가 있으면 → 재정렬로 살릴 수 있음, rank도 없으면 → 질문-문서 단어 불일치(질문 재작성·의미 검색 필요)

데이터 분리 (규칙):
- DEV  : 튜닝용. 오답을 보고 고쳐도 된다. (FAQ 질문 중 TEST와 근거 조문이 겹치지 않는 것)
- TEST : answer_holdout3.py 35문항. **튜닝 중에는 보지 않는다.** 개선 단계를 끝낼 때만 `--test`로 한 번 잰다.
gold = 정답 문장에 들어 있는 정규식(공백 제거 후 label+text에 매칭). 하나라도 맞으면 정답 문서.

사용: python retrieval_eval.py          # DEV
      python retrieval_eval.py --test   # TEST(단계 마감 때만)
"""
import json, pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import answer as A

# (질문, gold 정규식) — 질문은 인증원 FAQ 2020·2021 민원 질문 문구
DEV = [
    ("안전관리인증기준(HACCP)을 적용할 수 있는 영업은 어떠한 것이 있나요?", r"^제4조\(적용품목"),
    ("운영하고 있는 중요관리점(CCP)을 스마트 HACCP으로 등록할 때 우선 일부만 등록할 수 있나요?", r"60%이상"),
    ("HACCP 인증업체는 연장신청을 언제 해야 하나요?", r"연장신청"),
    ("축산물 조사평가 95% 이상 시 평가 면제기간은 어떻게 되나요?", r"95%이상인경우2년간"),
    ("제품설명서 작성은 어떻게 하나요?", r"제품설명서"),
    ("제조공정도 작성은 어떻게 하나요?", r"제조공정도"),
    ("원료 위해요소분석은 어떻게 시작해야 하나요?", r"과학적근거"),
    ("가열공정 한계기준 설정은 어떻게 하나요?", r"한계기준설정을위해|유.성평가자료"),
    ("심사 후 보완 판정을 받았습니다. 지적사항에 대한 보완은 어떠한 방법으로 수행해야 하나요?", r"3개월이내에보완"),
    ("HACCP인증 받은 유형의 생산 중단 등의 사유로 HACCP 인증서를 반납하고자 하는데 어떻게 해야 하나요?", r"인증서의반납"),
    ("식품 HACCP인증 이후 HACCP팀장이 변경될 경우 새로운 HACCP 팀장은 신규 HACCP팀장 교육(16시간)을 이수해야 하나요?", r"팀장교체"),
    ("위해요소분석 시 생물학적 위해요소에 대한 발생가능성 평가는 어떻게 하나요?", r"발생가능성"),
    ("수입원료 입고 시 수입신고필증으로 검사성적서를 갈음할 수 있나요?", r"수입필증|시험성적서"),
    ("고시에 따르면 HACCP기록은 최소 2년간 보관하여야 한다는데, 반드시 2년 이상 보관하여야하는 기록서류에는 어떤 것이 있나요?", r"2년간보관"),
    ("HACCP관리기준서를 업종별, 유형별로 각각 구비하여야 하나요?", r"업종별에따라각각"),
    ("작업장 내 창문을 만들 수 있나요? HACCP에서 어떠한 관리가 필요 할까요?", r"채광"),
    ("작업장의 구역설정(청결구역, 준청결구역, 일반구역)을 할 때 어떠한 사항을 고려하여 설정해야 하나요?", r"청결구역"),
]
DEV_NEG = [   # 코퍼스에 근거 없음(인증원 민원·상담 업무) — 거절 관문 튜닝용
    "영문인증서 발급은 어떻게 신청하여 발급 받을 수 있나요?",
    "HACCP 인증 전·후 상담 및 신청 방법을 알고 싶어요.",
    "작업장을 신축하려 합니다. HACCP 평가기준에 맞추어 작업장을 설계 하려고 하는데, 도면을 검토해주실 수 있나요?",
]

# TEST: answer_holdout3.CASES의 pos 31문항 순서대로 gold
TEST_GOLD = [
    r"6개월이내", r"화장실", r"내수성", r"장신구", r"포충", r"내부식성", r"압축공기", r"저수조",
    r"선입선출|이격", r"검.?교정", r"인수인계", r"위탁.{0,40}공정|공정.{0,40}위탁", r"인증대상의추가",
    r"적용업소인증사항변경", r"적용업소인증사항변경", r"통합관리프로그램", r"기본심벌", r"5억",
    r"외부전문가", r"생물학적.{0,3}화학적.{0,3}물리적", r"재검토", r"개선조치", r"\"검증\(Verification\)\"이란",
    r"유.성평가", r"검.?교정", r"동일함", r"위생모", r"수질검사", r"검사기관", r"교육훈련계획", r"회수관리|회수프로그램",
]


def n(s):
    return re.sub(r"\s+", "", s)


def doc_text(d):
    return n(d.get("label", "") + " " + d.get("text", ""))


def run(items):
    u = A.index()
    rows = []
    for q, gold in items:
        g = re.compile(gold)
        corpus = any(g.search(doc_text(d)) for d in u.docs)
        hits = A.retrieve(q)
        delivered = any(g.search(doc_text(d)) for _, d in hits)
        cands = [(a, " ".join(map(str, t)) if isinstance(t := u.art_text.get(a, ""), list) else str(t)) for _, a in u.rank_articles(q, 20)]
        cands += [(d.get("label", ""), d.get("text", "")) for _, d in u.rank_pages(q, 20)]
        rank = next((r for r, (lab, t) in enumerate(cands, 1) if g.search(n(lab + " " + t))), None)
        top = [d.get("label", "")[:24] for _, d in hits][:4]
        rows.append({"q": q, "gold": gold, "corpus": corpus, "delivered": delivered, "rank": rank, "top": top})
    return rows


def main():
    test = "--test" in sys.argv
    if test:
        import answer_holdout3 as H
        pos = [c[1] for c in H.CASES if c[4] == "pos"]
        assert len(pos) == len(TEST_GOLD), (len(pos), len(TEST_GOLD))
        items = list(zip(pos, TEST_GOLD))
    else:
        items = DEV
    rows = run(items)
    for i, r in enumerate(rows, 1):
        mark = "O" if r["delivered"] else ("X" if r["corpus"] else "?")
        rk = r["rank"] if r["rank"] else "-"
        print(f"{i:2d} {mark} r={rk!s:>2} {r['q'][:40]} | {' / '.join(r['top'])}")
    ok = sum(r["delivered"] for r in rows)
    wide = sum(bool(r["rank"]) for r in rows)
    bad_gold = [i for i, r in enumerate(rows, 1) if not r["corpus"]]
    name = "TEST" if test else "DEV"
    print(f"\n{name} 근거 전달 {ok}/{len(rows)} | 후보 40개 안 {wide}/{len(rows)}  (gold 패턴이 코퍼스에 없음: {bad_gold})")
    out = pathlib.Path(__file__).with_name(f"retrieval_eval_{name.lower()}.jsonl")
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
