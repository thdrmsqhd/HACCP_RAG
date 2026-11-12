"""HACCP 도우미 CLI — 3단 답변(설명/근거/다음 할 일)."""
import re
import sys

import ccp
from retrieve import get_index

STAGE = {
    "3": "1단계 · 적용 대상 판단", "4": "1단계 · 적용 대상 판단",
    "5": "2단계 · 선행요건 정비",
    "6": "3단계 · HACCP 관리계획 수립", "9": "3단계 · HACCP 관리계획 수립",
    "10": "4단계 · 인증 신청", "11": "4단계 · 인증 신청",
    "15": "5단계 · 인증 후 유지", "20": "5단계 · 인증 후 유지", "8": "5단계 · 인증 후 유지",
}

NEXT = {
    "3": "우리 업종이 적용 대상인지 확인하고, 의무/자율 여부를 정리하세요.",
    "4": "연매출액과 상시근로자 수를 확인해 의무 적용 시기를 판단하세요.",
    "5": "우리 업종의 선행요건 항목을 뽑고 담당자·점검 주기를 정하세요.",
    "6": "위해요소 분석 → CCP → 한계기준 → 모니터링 → 개선조치 → 검증 → 문서화 순으로 계획을 작성하세요.",
    "9": "HACCP팀(팀장·팀원)을 구성하고 팀장 교육·훈련 계획을 세우세요.",
    "10": "별지 제52서식 신청서와 식품안전관리인증계획서를 준비하세요.",
    "11": "현장조사 대비 시설·설비·기록의 실제 운영 상태를 정리하세요.",
    "15": "정기 조사·평가(연 1회 이상) 대비 기록을 점검하세요.",
    "20": "인증일로부터 6개월 이내 신규교육을 이수하세요.",
    "8": "모니터링·검증 기록이 끊기지 않게 관리하세요.",
}

USE_EMB = False

PROCESS_KW = ["살균", "금속검출", "세척", "소독", "냉각", "건조", "포장", "원료입고", "가열"]
INDUSTRY_KW = ["육가공", "식육", "유가공", "알가공", "식품제조", "건강기능식품", "급식"]


def pick_sentence(text: str, query: str, n: int = 2) -> str:
    sents = [s.strip() for s in re.split(r"(?<=[.다])\s", text) if len(s.strip()) > 15]
    q = set(re.findall(r"[가-힣]{2,}", query))
    scored = sorted(sents, key=lambda s: -sum(1 for w in q if w in s))
    return " ".join(scored[:n])[:400]


def answer(q: str) -> str:
    # 정형 질의 라우팅
    if "CCP" in q.upper() or any(k in q for k in PROCESS_KW):
        process = next((k for k in PROCESS_KW if k in q), None)
        industry = next((k for k in INDUSTRY_KW if k in q), None)
        r = ccp.query(업종=industry, 공정=process)
        if r["count"]:
            head = f"[설명] '{process or industry}' 관련 인증업체 {r['count']}건에서 확인된 패턴입니다. " \
                   f"가장 많이 설정된 CCP는 {r['top_ccp'][0][0]}({r['top_ccp'][0][1]}건)입니다."
            return "\n".join([
                head,
                f"[근거] 공공데이터 'HACCP 중요관리점 정보'(한국식품안전관리인증원, 2025-07-21, {r['count']}건 조회)",
                f"[참고] 주요 CCP: {', '.join(f'{c}({n})' for c, n in r['top_ccp'][:3])}",
                f"[참고] 주요 위해요소: {', '.join(f'{h}({n})' for h, n in r['top_hazard'][:3])}",
                "[다음 할 일] 자사 공정흐름도에 이 CCP 유형을 대입해 후보를 표시하고, 한계기준은 자사 데이터로 설정하세요.",
            ])

    # 서술형 질의
    if USE_EMB:
        from hybrid import get_hybrid
        best = get_hybrid(alpha=0.6).search(q, 3)
        idx = get_index()
        by_num = {a["num"]: a for a in idx.arts}
        hits = [(sc, by_num[num]) for sc, num, *_ in best if num in by_num]
    else:
        hits = get_index().search(q, 3)
    if not hits:
        return "[설명] 근거를 찾지 못했습니다.\n[근거] 없음\n[다음 할 일] 질문을 업무 단계(대상 판단/선행요건/계획/신청/유지)로 나눠 다시 물어보세요."
    score, art = hits[0]
    stage = STAGE.get(art["num"], "")
    return "\n".join([
        f"[설명] {pick_sentence(art['text'], q)}",
        f"[근거] 「식품 및 축산물 안전관리인증기준」 제{art['num']}조({art['title']})",
        f"[단계] {stage}" if stage else "[단계] -",
        f"[다음 할 일] {NEXT.get(art['num'], '해당 조항을 확인하고 담당자와 범위를 확정하세요.')}",
    ])


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--emb" in args:
        USE_EMB = True
        args = [a for a in args if a != "--emb"]
    qs = args or [
        "우리 공장이 의무 적용인지 어떻게 판단하나요?",
        "인증 신청 서류는 무엇인가요?",
        "인증 후 교육은 언제까지 받아야 하나요?",
        "살균공정 CCP는 보통 무엇인가요?",
    ]
    for q in qs:
        print("=" * 60)
        print("Q:", q)
        print(answer(q))
    print("=" * 60)
