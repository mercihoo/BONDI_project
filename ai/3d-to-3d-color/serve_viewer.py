"""
serve_viewer.py — 비교 뷰어용 정적 서버 + 마스크 붓 저장 API

`python -m http.server`와 같은 정적 서버인데, 뷰어의 붓 도구가 칠한 마스크를 저장소 안에 바로 쓸 수 있도록 두 개의 API를
더한다. 저장 위치는 `ai/3d-to-3d-color/out_restore/mask_paint/<유물키>_paint.png`(흑백, 흰색 = 마스크).

  python ai/3d-to-3d-color/serve_viewer.py [--port 8765] [--root <저장소 루트>]
  http://localhost:8765/ai/3d-to-3d-color/out_restore/viewer.html

API
  POST /api/mask/save   {"key": "ssu022891", "png": "data:image/png;base64,..."}  -> {"path": ...,"white_frac": ...}
  GET  /api/mask/load?key=ssu022891                                               -> PNG (없으면 404)
  POST /api/restore     {"key": "ssu022891"}     칠한 마스크로 복원 + 뷰어 복사본 재생성 (백그라운드, 10분 안팎)
  GET  /api/restore/status                                                        -> {"running":, "log":, "done":}

칠한 마스크는 복원 스크립트에 그대로 넘긴다:
  python restore_original_colour.py ... --remove-cracks --crack-mask out_restore/mask_paint/<키>_paint.png
"""
import argparse
import base64
import io
import json
import os
import re
import subprocess
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

MASK_DIR = os.path.join("ai", "3d-to-3d-color", "out_restore", "mask_paint")
KEY_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
AI_DIR = os.path.join("ai", "3d-to-3d-color")
POC_IN = r"C:\ai\poc_inputs"                       # 8K OBJ 원본이 있는 곳 (--poc-inputs 로 변경)
PY = os.environ.get("RESTORE_PYTHON", r"C:\ai\venv-3dtools\Scripts\python.exe")
JOB = {"proc": None, "log": [], "key": None, "done": True, "rc": None, "running": False, "t0": 0.0}
JOB_LOCK = threading.Lock()


def find_model(root, key):
    """<poc_inputs>/<유물 폴더>/<stem>.obj 를 찾는다. 유물 키가 폴더 이름과 다를 수 있어(don000498_001 등) 앞부분으로도 맞춰 본다."""
    base = POC_IN if os.path.isabs(POC_IN) else os.path.join(root, POC_IN)
    for cand in (key, key.split("_")[0]):
        d = os.path.join(base, cand)
        if os.path.isdir(d):
            objs = [f for f in sorted(os.listdir(d)) if f.lower().endswith(".obj")]
            if objs:
                return os.path.join(d, objs[0])
    return None


def run_job(*a):
    try:
        _run_job(*a)
    except Exception as e:                                            # a crash here must not leave the job "running" for ever
        with JOB_LOCK:
            JOB["log"] = (JOB["log"] + [f"내부 오류: {e!r}"])[-60:]
            JOB["rc"] = -1
    finally:
        with JOB_LOCK:
            JOB["done"], JOB["running"] = True, False


def _run_job(root, key, model, mask, material, args_extra):
    out = f"out_restore/poc_{key}_brush"
    cmd = [PY, "restore_original_colour.py", "--model", model.replace("\\", "/"), "--out", out,
           "--material", material, "--mode", "conserved", "--no-smooth", "--remove-cracks",
           "--crack-mask", mask, "--tag", f"{material}_brush"] + args_extra
    cwd = os.path.join(root, AI_DIR)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")   # piped stdout is block-buffered: without
                                                                            # this the log stays empty for ten minutes
    for step in (cmd, [PY, "make_viewer_copies.py", "--only", key]):
        with JOB_LOCK:
            JOB["log"].append("$ " + " ".join(step[1:]))
        pr = subprocess.Popen(step, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1)
        with JOB_LOCK:
            JOB["proc"] = pr
        for line in pr.stdout:
            line = line.rstrip()
            if line:
                with JOB_LOCK:
                    JOB["log"] = (JOB["log"] + [line])[-60:]
        pr.wait()
        if pr.returncode:
            with JOB_LOCK:
                JOB["rc"] = pr.returncode
            return
    with JOB_LOCK:
        JOB["rc"] = 0


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # The viewer is the read-out of a pipeline that rewrites these files. A cached manifest.json or viewer.html means
        # a rerun (or a brush restore) silently shows the previous run's result, which is worse than showing nothing.
        if self.path.split("?")[0].endswith((".json", ".html", ".js")):
            self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _mask_path(self, key):
        return os.path.join(self.directory, MASK_DIR, f"{key}_paint.png")

    def do_GET(self):
        if self.path.startswith("/api/restore/status"):
            with JOB_LOCK:
                return self._json(200, {"running": JOB["running"], "key": JOB["key"], "done": JOB["done"], "rc": JOB["rc"],
                                        "elapsed_s": round(time.time() - JOB["t0"]) if JOB["t0"] else 0, "log": JOB["log"][-12:]})
        if self.path.startswith("/api/mask/load"):
            key = ""
            if "?" in self.path:
                for part in self.path.split("?", 1)[1].split("&"):
                    if part.startswith("key="):
                        key = part[4:]
            if not KEY_RE.match(key):
                return self._json(400, {"error": "bad key"})
            p = self._mask_path(key)
            if not os.path.exists(p):
                return self._json(404, {"error": "no saved mask", "path": os.path.relpath(p, self.directory)})
            data = open(p, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/restore":
            return self._restore()
        if self.path != "/api/mask/save":
            return self._json(404, {"error": "unknown endpoint"})
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0 or n > 64 * 1024 * 1024:
            return self._json(413, {"error": "bad size"})
        try:
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            key = req["key"]
            if not KEY_RE.match(key):
                return self._json(400, {"error": "bad key"})
            raw = base64.b64decode(req["png"].split(",", 1)[-1])
        except Exception as e:                                        # noqa: BLE001 - report to the browser
            return self._json(400, {"error": f"bad request: {e}"})
        p = self._mask_path(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(raw)
        info = {"path": os.path.relpath(p, self.directory).replace("\\", "/"), "bytes": len(raw)}
        try:                                                          # report the painted fraction so the browser can show it
            from PIL import Image
            import numpy as np
            a = np.asarray(Image.open(io.BytesIO(raw)).convert("L")) > 127
            info["white_frac"] = float(a.mean())
            info["size"] = list(a.shape[::-1])
        except Exception:                                             # noqa: BLE001 - Pillow is optional here
            pass
        print(f"saved {info['path']}  {info.get('white_frac', float('nan')):.4%} white")
        return self._json(200, info)

    def _restore(self):
        """Run restore_original_colour.py with the painted mask, then rebuild the viewer copies. One job at a time."""
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
            key = req.get("key", "")
        except Exception as e:                                        # noqa: BLE001
            return self._json(400, {"error": str(e)})
        if not KEY_RE.match(key):
            return self._json(400, {"error": "bad key"})
        mask = self._mask_path(key)
        if not os.path.exists(mask):
            return self._json(404, {"error": "먼저 마스크를 저장하세요"})
        model = find_model(self.directory, key)
        if not model:
            return self._json(404, {"error": f"{POC_IN} 에서 {key} 의 OBJ를 찾지 못했습니다"})
        material = req.get("material") or "earthenware"
        base = req.get("base_mask")                                   # the mask the variant on screen was made with
        if base and re.match(r"^[A-Za-z0-9_./-]{1,200}$", base) and ".." not in base:
            merged = self._merge_masks(mask, os.path.join(self.directory, AI_DIR, base), key)
            if merged:
                mask = merged
        extra = [a for a in (req.get("extra") or []) if re.match(r"^[-A-Za-z0-9_.=/ ]+$", str(a))]
        rel_mask = os.path.relpath(mask, os.path.join(self.directory, AI_DIR)).replace("\\", "/")
        with JOB_LOCK:                                                # claim the slot under the lock: two quick clicks
            if JOB["running"]:                                        # used to start two jobs, because the worker thread
                return self._json(409, {"error": "이미 실행 중", "key": JOB["key"]})   # had not filled JOB["proc"] yet
            JOB.update({"proc": None, "log": [], "key": key, "done": False, "rc": None, "running": True, "t0": time.time()})
        threading.Thread(target=run_job, args=(self.directory, key, model, rel_mask, material, extra), daemon=True).start()
        return self._json(200, {"started": True, "key": key, "model": model, "mask": rel_mask, "material": material})

    @staticmethod
    def _merge_masks(paint_png, base_png, key):
        """Union of the freshly painted mask and the one the displayed variant already used. Without it, running from
        the viewer would DROP the earlier SAM / hand mask and the result would be worse than what is on screen."""
        if not os.path.exists(base_png) or os.path.abspath(base_png) == os.path.abspath(paint_png):
            return None
        try:
            from PIL import Image
            import numpy as np
            a = np.asarray(Image.open(paint_png).convert("L")) > 127
            b = Image.open(base_png).convert("L")
            if b.size != a.shape[::-1]:
                b = b.resize(a.shape[::-1], Image.NEAREST)
            out = a | (np.asarray(b) > 127)
            p = os.path.join(os.path.dirname(paint_png), f"{key}_combined.png")
            Image.fromarray((out * 255).astype("uint8")).save(p)
            print(f"merged mask -> {os.path.basename(p)}  ({out.mean():.4%})")
            return p
        except Exception as e:                                        # noqa: BLE001 - fall back to the painted mask alone
            print("mask merge failed:", e)
            return None

    def log_message(self, fmt, *args):                                # keep the console readable: only errors
        if not str(args[0]).startswith(("GET /viewer", "GET /ai")):
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--poc-inputs", default=POC_IN, help="8K OBJ 원본 폴더 (뷰어의 '이 마스크로 복원' 버튼이 모델을 찾는 곳)")
    ap.add_argument("--python", default=PY, help="복원 스크립트를 실행할 파이썬 (기본 C:/ai/venv-3dtools)")
    a = ap.parse_args()
    globals()["POC_IN"], globals()["PY"] = a.poc_inputs, a.python
    srv = ThreadingHTTPServer((a.bind, a.port), partial(Handler, directory=a.root))
    print(f"serving {a.root} on http://{a.bind}:{a.port}")
    print(f"viewer:  http://{a.bind}:{a.port}/ai/3d-to-3d-color/out_restore/viewer.html")
    srv.serve_forever()


if __name__ == "__main__":
    main()
