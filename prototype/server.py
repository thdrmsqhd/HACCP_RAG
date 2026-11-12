"""HACCP 도우미 로컬 웹 UI (표준 라이브러리만).

실행: python server.py  → http://127.0.0.1:8765
"""
import json
import pathlib
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import answer

ROOT = pathlib.Path(__file__).parent
PORT = 8765


def warmup():
    """모델을 미리 올려 둔다. 첫 질문에서 35~40초 로딩을 기다리지 않게."""
    try:
        req = urllib.request.Request(
            answer.OLLAMA,
            data=json.dumps({"model": answer.MODEL, "prompt": "hi", "stream": False,
                             "think": False, "keep_alive": answer.KEEP_ALIVE,
                             "options": {"num_predict": 1}}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=300).read()
        print("모델 준비 완료", flush=True)
    except Exception as e:
        print("워밍업 실패:", e, flush=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _query(self):
        return (parse_qs(urlparse(self.path).query).get("q") or [""])[0].strip()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/ask":
            q = self._query()
            if not q:
                self._send(400, b'{"error":"empty query"}')
                return
            try:
                self._send(200, json.dumps({"q": q, **answer.ask(q)}, ensure_ascii=False).encode())
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}, ensure_ascii=False).encode())
            return
        if path == "/api/stream":
            q = self._query()
            if not q:
                self._send(400, b'{"error":"empty query"}')
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                meta, gen = answer.ask_stream(q)
                self._chunk((json.dumps({"hits": meta["hits"]}, ensure_ascii=False) + "\n").encode())
                if "answer" in meta:
                    self._chunk(meta["answer"].encode())
                else:
                    for piece in gen:
                        self._chunk(piece.encode())
            except Exception as e:
                self._chunk(("\n[오류] " + str(e)).encode())
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
        self._send(404, b"not found")

    def _chunk(self, b: bytes):
        if not b:
            return
        self.wfile.write(f"{len(b):X}\r\n".encode() + b + b"\r\n")
        self.wfile.flush()


if __name__ == "__main__":
    threading.Thread(target=warmup, daemon=True).start()
    print(f"http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
