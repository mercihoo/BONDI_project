#!/usr/bin/env python3
"""정적 서버 + 캡처 저장 — viewer.html 이 렌더한 캔버스를 POST /save?name=... 로 받아 renders/ 에 PNG 로 쓴다.
브라우저가 그린 그림(원본은 질감, 결과는 음영 메시)을 그대로 파일로 남기기 위한 것. 이미지가 대화 컨텍스트를 거치지 않는다.

    cd ~/work/LaS-Comp && python3 tools/capture_server.py 8791
"""
import sys, os, re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8791
OUT = os.path.join(os.getcwd(), "renders")
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")


class H(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *a):  # 조용히
        pass

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/save":
            self.send_error(404); return
        name = parse_qs(u.query).get("name", [""])[0]
        if not NAME_RE.match(name) or not name.endswith(".png"):
            self.send_error(400, "bad name"); return
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0 or n > 50_000_000:
            self.send_error(400, "bad length"); return
        data = self.rfile.read(n)
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, name), "wb") as f:
            f.write(data)
        body = ("saved %s %d bytes" % (name, n)).encode()
        self.send_response(200); self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        print("[capture]", name, n, "bytes", flush=True)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
