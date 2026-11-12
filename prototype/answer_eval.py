"""답변 정확성 평가 — 30문항.

채점 기준(자동):
- 정답: 필수 사실 토큰을 모두 포함
- 부분: 일부만 포함
- 오답: 하나도 없음
- 출처 일치: [근거] 줄에 기대 출처(조문번호/페이지)가 표기됨

한계: 토큰 포함 여부는 의미 일치를 완벽히 대표하지 않는다. 실패 사례는 사람이 따로 검토한다.
"""
import json
import pathlib
import re
import sys
import time

import answer

OUT = pathlib.Path(__file__).with_name("answer_eval_results.jsonl")

# (질문, 필수 토큰, 기대 출처)
CASES = [
    # 고시
    ("HACCP 기록은 몇 년간 보관해야 하나요?", ["2년"], "제8조"),
    ("인증 신청 서류가 미흡하면 며칠 안에 보완해야 하나요?", ["15일"], "제10조"),
    ("현장조사 후 보완 요구 기한은 얼마인가요?", ["3개월"], "제11조"),
    ("정기 조사·평가 점수가 95% 이상이면 어떻게 되나요?", ["2년"], "제15조"),
    ("정기 조사·평가 점수가 90% 이상 95% 미만이면?", ["1년"], "제15조"),
    ("점수가 85% 미만 70% 이상이면 조사와 기술지원을 몇 회 받나요?", ["1회", "기술지원"], "제15조"),
    ("점수가 70% 미만이면 기술지원을 몇 회 받나요?", ["2회"], "제15조"),
    ("영업자 교육훈련 시간은 몇 시간인가요?", ["2시간"], "제20조"),
    ("HACCP 팀장 교육훈련은 몇 시간인가요?", ["16시간"], "제20조"),
    ("HACCP 팀원과 기타 종업원 교육훈련 시간은?", ["4시간"], "제20조"),
    ("교육훈련기관 강사 평가는 몇 점 이상이면 적합인가요?", ["80점"], "제23의3조"),
    ("자동 기록관리 시스템으로 심벌 표시·광고를 하려면 CCP를 몇 % 이상 적용해야 하나요?", ["60%"], "제27조"),
    ("검증 결과 보완이 필요하면 며칠 안에 보완해야 하나요?", ["1개월"], "제18조"),
    ("의무적용 시기는 최대 얼마까지 유예할 수 있나요?", ["6개월"], "제4조"),
    # 매뉴얼
    ("실시상황평가표에서 5점은 어떤 의미인가요?", ["만족"], "p.21"),
    ("평가항목이 필수항목인데 0점이면 어떻게 되나요?", ["부적합"], "p.22"),
    ("청결구역과 일반구역은 어떻게 구분해야 하나요?", ["분리"], "p.29"),
    ("작업장 출입구에 갖춰야 하는 위생설비는?", ["세척", "소독"], "p.32"),
    ("작업장 내부 창·유리에 대해 무엇을 수립해야 하나요?", ["이물"], "p.34"),
    ("작업장과 육안 검사 구간의 조도 기준은?", ["220", "540"], "p.35"),
    ("위생복은 어떻게 보관해야 하나요?", ["분리"], "p.38"),
    ("물류와 종업원 이동에 대해 무엇을 수립해야 하나요?", ["동선"], "p.39"),
    ("원료 입고부터 운송까지 이물은 어떻게 관리하나요?", ["이물", "기준"], "p.40"),
    ("구역별 출입 복장과 세척·소독은 무엇을 수립해야 하나요?", ["위생 수칙"], "p.41"),
    ("공정별 온도와 습도는 어떻게 관리하나요?", ["온도", "습도"], "p.42"),
    ("외부 흡기구와 배기구에는 각각 무엇을 설치하나요?", ["여과망", "방충망"], "p.44"),
    ("세척·소독 게시물은 어디에 게시해야 하나요?", ["출입구", "게시"], "p.50"),
    ("온도를 높이거나 낮추는 시설에는 무엇을 설치해야 하나요?", ["온도"], "p.57"),
    ("공정흐름에 따라 이동경로를 어떻게 표시해야 하나요?", ["이동경로"], "p.33"),
    ("해충 유입을 막기 위한 관리 기준에는 무엇을 정해야 하나요?", ["점검 주기"], "p.45"),
]


def norm(s: str) -> str:
    """띄어쓰기·전각기호 차이로 채점이 흔들리지 않게 정규화."""
    return re.sub(r"\s+", "", s).replace("％", "%")


def score_one(q, keys, src):
    t0 = time.time()
    r = answer.ask(q)
    ans = r["answer"]
    flat = norm(ans)
    found = [k for k in keys if norm(k) in flat]
    if len(found) == len(keys):
        verdict = "정답"
    elif found:
        verdict = "부분"
    else:
        verdict = "오답"
    cite_ok = norm(src) in flat
    return {
        "q": q, "keys": keys, "found": found, "verdict": verdict,
        "cite_expected": src, "cite_ok": cite_ok,
        "sec": round(time.time() - t0, 1),
        "answer": ans,
        "hits": r["hits"], "unsupported": r.get("unsupported_numbers", []),
    }


def main():
    rows = []
    with OUT.open("w", encoding="utf-8") as f:
        for i, (q, keys, src) in enumerate(CASES, 1):
            try:
                row = score_one(q, keys, src)
            except Exception as e:
                row = {"q": q, "keys": keys, "found": [], "verdict": "오류",
                       "cite_expected": src, "cite_ok": False, "sec": 0,
                       "answer": f"ERROR {e}", "hits": []}
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{i:2d}/{len(CASES)} {row['verdict']:<4} cite={'O' if row['cite_ok'] else 'X'} {row['sec']:5.1f}s | {q[:40]}", flush=True)

    n = len(rows)
    cnt = {v: sum(1 for r in rows if r["verdict"] == v) for v in ["정답", "부분", "오답", "오류"]}
    cites = sum(1 for r in rows if r["cite_ok"])
    print("\n=== 요약 ===")
    print(f"정답 {cnt['정답']}/{n} ({cnt['정답']/n*100:.0f}%)  부분 {cnt['부분']}  오답 {cnt['오답']}  오류 {cnt['오류']}")
    print(f"출처 일치 {cites}/{n} ({cites/n*100:.0f}%)")
    print(f"평균 {sum(r['sec'] for r in rows)/n:.1f}초/문항")
    print("결과 파일:", OUT)


if __name__ == "__main__":
    main()
