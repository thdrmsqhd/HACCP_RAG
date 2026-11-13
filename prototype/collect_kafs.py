"""한국식품안전관리인증원 공개 파일데이터 수집 (data.go.kr).

사용:
    python collect_kafs.py 15145158 15163536 15146780

- data.go.kr 파일데이터는 두 단계로 받는다.
  1) 페이지의 fn_fileDataDown(pk, uddi, '', '1','1') 인자 확인
  2) POST /tcs/dss/selectFileDataDownload.do -> atchFileId
  3) GET  /cmm/cmm/fileDownload.do?atchFileId=..&fileDetailSn=1 -> 실제 파일
- 로그인 없이 받을 수 있는 파일데이터만 대상. OpenAPI는 인증키가 필요해 제외.
"""
from __future__ import annotations

import io
import json
import pathlib
import re
import sys
import urllib.parse
import urllib.request
import zipfile

OUT = pathlib.Path(__file__).resolve().parent.parent / "corpus" / "01_공개자료" / "인증원"
UA = {"User-Agent": "Mozilla/5.0"}
BASE = "https://www.data.go.kr"


def get(url: str, data: bytes | None = None) -> bytes:
    req = urllib.request.Request(url, data=data, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def detail_args(pk: str) -> tuple[str, str]:
    """페이지에서 (pk, uddi) 추출."""
    html = get(f"{BASE}/data/{pk}/fileData.do").decode("utf-8", "replace")
    m = re.search(r"fn_fileDataDown\(\s*'(\d+)'\s*,\s*'([^']+)'", html)
    if not m:
        raise RuntimeError(f"{pk}: fn_fileDataDown 인자 없음")
    return m.group(1), m.group(2)


def attch_id(pk: str, uddi: str) -> str:
    body = urllib.parse.urlencode(
        {"publicDataPk": pk, "publicDataDetailPk": uddi}
    ).encode()
    js = json.loads(get(f"{BASE}/tcs/dss/selectFileDataDownload.do", body))
    if not js.get("status") or not js.get("atchFileId"):
        raise RuntimeError(f"{pk}: 다운로드 거부 {js.get('status')}")
    return js["atchFileId"]


def title_of(pk: str) -> str:
    html = get(f"{BASE}/data/{pk}/fileData.do").decode("utf-8", "replace")
    m = re.search(r"<title>\s*(.*?)\s*</title>", html, re.S)
    t = re.sub(r"\s*\|\s*공공데이터포털.*$", "", m.group(1)) if m else f"dataset_{pk}"
    return re.sub(r'[\\/:*?"<>|]', "_", t).strip()[:80]


def save(pk: str, blob: bytes) -> list[pathlib.Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{pk}_{title_of(pk)}"
    if zipfile.is_zipfile(io.BytesIO(blob)):
        z = zipfile.ZipFile(io.BytesIO(blob))
        saved = []
        for info in z.infolist():
            if info.is_dir():
                continue
            # zip 안 파일명이 cp949로 깨져 오는 경우가 있어 순번으로 대체
            ext = pathlib.Path(info.filename).suffix or ".bin"
            p = OUT / f"{name}_{len(saved)+1}{ext}"
            p.write_bytes(z.read(info))
            saved.append(p)
        return saved
    p = OUT / f"{name}.csv"
    p.write_bytes(blob)
    return [p]


def main(ids: list[str]) -> int:
    manifest = OUT.parent.parent / "03_메타" / "인증원_수집목록.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    new = not manifest.exists()
    rows = []
    for pk in ids:
        try:
            d_pk, uddi = detail_args(pk)
            fid = attch_id(d_pk, uddi)
            blob = get(f"{BASE}/cmm/cmm/fileDownload.do?atchFileId={fid}&fileDetailSn=1")
            files = save(pk, blob)
            for p in files:
                print(f"OK {pk} {p.name} {p.stat().st_size:,}B")
                rows.append((pk, p.name, p.stat().st_size))
        except Exception as e:  # 한 건 실패가 전체를 막지 않게
            print(f"FAIL {pk}: {e}")
    if rows:
        with manifest.open("a", encoding="utf-8-sig", newline="") as f:
            if new:
                f.write("dataset_id,file,bytes\n")
            for pk, name, size in rows:
                f.write(f"{pk},{name},{size}\n")
    print(f"저장 위치: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["15145158", "15163536", "15146780", "15001781"]))
