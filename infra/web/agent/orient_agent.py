#!/usr/bin/env python3
"""자료서버 회전 점검 에이전트 — 로컬 Qwen2.5-VL(무료)이 미리보기를 보고 유물이 바로 섰는지 판정한다.

고치지 않는다 — **보고만 한다** (2026-09-23 결정). 대신 사람이 그대로 실행할 rebuild 명령을 제안한다.
모든 프롬프트·판정·호출은 JSONL 로 남긴다 (D-19 의 "프롬프트 전문 + 작업 로그" 규칙 그대로).

판정 방식 — 절대 방향을 묻지 않는다 (2026-09-28 실측: 7B 는 "바로 섰나?"에 누운 향로도 0.95 로
"바로 섬"이라 답했다). 대신 **같은 그림을 0/90/180/270° 돌린 네 장을 주고 "바른 것은 몇 번?"** 을
묻는다 — 비교 선택으로 바꾸면 같은 모델이 정답지 3/3 을 맞혔고, 고치는 회전까지 골라냈다.

    지각    썸네일을 네 방향으로 돌려 보여준다
    판단    바른 그림 번호 선택 → 1번(원본)이 아니면 문제
    행동    모델이 고른 회전을 적용한 그림을 만들어 "이건 바로 섰나?" 재확인
    재지각  아니라고 하면 미리보기 GLB 를 받아 정면 렌더로 한 번 더
    보고    콘솔 + JSONL + rebuild 제안 (실행은 안 한다)

  python orient_agent.py check "new20260918_02/3d/source" [...]
  python orient_agent.py check --all                        # 서버 3D 카드 전부
  python orient_agent.py upload new20260918_02 restored.glb --label "LaS-Comp v5" \
      --method "LaS-Comp 제로샷 완성 · tau 0.6" --owner 기여자     # 에이전트 등록(일반 등록과 병존)

필요한 것: Windows Ollama + `ollama pull qwen2.5vl:7b`. 파이썬은 표준 라이브러리만.
이미지 회전·추가 렌더는 WSL lascomp 환경(PIL·pyrender)을 부른다 — 없으면 그만큼 라운드가 준다.
"""
import argparse
import base64
import json
import mimetypes
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

for _s in (sys.stdout, sys.stderr):          # Windows 콘솔(cp949)에서 한글이 깨지지 않게
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                       # infra/web/agent → 저장소 루트
LAS_COMP = REPO / "ai" / "las-comp-restoration"


def load_env(path):
    """infra/web/.env 의 KEY=VALUE 를 환경변수로 — 이미 잡혀 있는 값이 이긴다. 서버 주소는 저장소에 두지 않는다."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and k and not k.startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


load_env(HERE.parent / ".env")
BASE = f"http://{os.environ['SERVER_HOST']}" if os.environ.get("SERVER_HOST") else None
OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen2.5vl:7b"

# best 번호 → (판정, 원본을 세우는 회전각 CCW)
PICK = {1: ("upright", 0), 2: ("lying_down", 90), 3: ("upside_down", 180), 4: ("lying_down", 270)}
VERDICT_KO = {"upright": "바로 섬", "upside_down": "거꾸로", "lying_down": "누움", "unsure": "판단 불가"}

PICK_SCHEMA = {"type": "object", "properties": {
    "best": {"type": "integer", "minimum": 1, "maximum": 4},
    "reason": {"type": "string"}}, "required": ["best", "reason"]}
PICK_PROMPT = ("같은 박물관 유물 그림을 네 방향으로 돌린 것이다 (1번, 2번, 3번, 4번 순서).\n"
               "유물이 전시 자세로 바르게 서 있는 그림은 몇 번인가?\n"
               "(그릇·향로·병은 아가리/뚜껑이 위, 굽·다리가 아래 · 관·불상은 세워진 자세)\n"
               "best 에 번호, reason 에 한국어 한 문장.")
YES_SCHEMA = {"type": "object", "properties": {
    "upright": {"type": "boolean"},
    "reason": {"type": "string"}}, "required": ["upright", "reason"]}
YES_PROMPT = ("박물관 유물 그림이다. 유물이 전시 자세로 바르게 서 있으면 upright=true,\n"
              "뒤집혔거나 누웠으면 false. reason 은 한국어 한 문장.")


# ── HTTP (표준 라이브러리만) ────────────────────────────────────────────────────
def _req(url, data=None, headers=None, timeout=120):
    r = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return f.read()


def get_json(url, timeout=60):
    return json.loads(_req(url, timeout=timeout).decode("utf-8"))


def post_multipart(url, fields, files, timeout=600):
    """files: [(필드명, 파일경로)] — 표준 라이브러리로 multipart/form-data."""
    b = "----agent" + uuid.uuid4().hex
    out = []
    for k, v in fields.items():
        if v is None:
            continue
        out += [f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode("utf-8")]
    for k, p in files:
        p = Path(p)
        ct = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        out += [(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"; "
                 f"filename=\"{p.name}\"\r\nContent-Type: {ct}\r\n\r\n").encode("utf-8"),
                p.read_bytes(), b"\r\n"]
    out += [f"--{b}--\r\n".encode("utf-8")]
    return json.loads(_req(url, data=b"".join(out), timeout=timeout,
                           headers={"Content-Type": f"multipart/form-data; boundary={b}"}).decode("utf-8"))


# ── 지각: 모델이 본다 ───────────────────────────────────────────────────────────
def look(images, prompt, schema, a, log):
    """이미지(경로 목록)를 Qwen 에게 보여주고 구조화 판정을 받는다."""
    imgs = [base64.b64encode(Path(p).read_bytes()).decode("ascii") for p in images]
    payload = {"model": a.model, "stream": False, "format": schema,
               # 기본 컨텍스트(4096)는 이미지 토큰이 넘는다 — 3000×2000 사진이 약 4.2k 토큰 (실측)
               "options": {"temperature": 0, "num_ctx": 16384},
               "messages": [{"role": "user", "content": prompt, "images": imgs}]}
    t0 = time.time()
    body, hdr = json.dumps(payload).encode("utf-8"), {"Content-Type": "application/json"}
    try:
        raw = _req(a.ollama + "/api/chat", data=body, headers=hdr, timeout=900)
    except urllib.error.HTTPError as e:            # Ollama 가 간헐적으로 500 을 낸다 — 한 번만 재시도
        if e.code < 500:
            raise
        log("look_retry", code=e.code)
        time.sleep(5)
        raw = _req(a.ollama + "/api/chat", data=body, headers=hdr, timeout=900)
    msg = json.loads(raw.decode("utf-8"))["message"]["content"]
    try:
        v = json.loads(msg)
    except json.JSONDecodeError:
        v = None
    log("look", images=[Path(p).name for p in images], prompt=prompt.splitlines()[0],
        raw=msg[:300], sec=round(time.time() - t0, 1))
    return v


# ── 행동: 증거를 만든다 (WSL 의 PIL·pyrender 사용, 없으면 건너뜀) ────────────────
def _wsl_path(p):
    p = str(Path(p).resolve())
    return "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")


def _wsl(py_code, timeout=600):
    cmd = "source ~/miniforge3/etc/profile.d/conda.sh && conda activate lascomp && python - <<'EOF'\n" + py_code + "\nEOF"
    return subprocess.run(["wsl.exe", "bash", "-lc", cmd], capture_output=True, timeout=timeout,
                          text=True, encoding="utf-8", errors="replace")  # WSL 출력은 UTF-8 — 기본(cp949)이면 reader 스레드가 죽는다


def make_rotations(image, workdir, log):
    """0/90/180/270° 회전본 4장. 실패하면 [] (그때는 원본 한 장 예/아니오 판정으로만)."""
    r = _wsl(f"""
from PIL import Image
im = Image.open("{_wsl_path(image)}").convert("RGB")
im.thumbnail((640, 640))
for a in (0, 90, 180, 270):
    im.rotate(a, expand=True).save("{_wsl_path(workdir)}/rot_%d.png" % a)
print("ok")""")
    outs = [workdir / f"rot_{d}.png" for d in (0, 90, 180, 270)]
    ok = r.returncode == 0 and all(o.exists() for o in outs)
    log("make_rotations", ok=ok, err=(r.stderr or "")[-300:] if not ok else "")
    return outs if ok else []


def render_front(glb_url, workdir, base, log):
    """미리보기 GLB 를 받아 정면 렌더 1장. 실패하면 None."""
    glb = workdir / "preview.glb"
    try:
        glb.write_bytes(_req(base + glb_url if glb_url.startswith("/") else glb_url, timeout=300))
    except Exception as e:
        log("render_front", ok=False, err=str(e)[:200])
        return None
    cmd = ("source ~/miniforge3/etc/profile.d/conda.sh && conda activate lascomp && "
           f"cd {_wsl_path(LAS_COMP)} && python tools/render_mesh_views.py "
           f"--pane '미리보기={_wsl_path(glb)}' --out {_wsl_path(workdir / 'srv')} --size 640 --views front")
    r = subprocess.run(["wsl.exe", "bash", "-lc", cmd], capture_output=True, timeout=600,
                       text=True, encoding="utf-8", errors="replace")
    out = workdir / "srv_front.png"
    ok = r.returncode == 0 and out.exists()
    log("render_front", ok=ok, err=(r.stderr or "")[-300:] if not ok else "")
    return out if ok else None


# ── 서버 질의 ──────────────────────────────────────────────────────────────────
def wait_preview(asset_id, base, timeout=240):
    q = urllib.parse.quote(asset_id, safe="")
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = get_json(f"{base}/api/preview-status?asset={q}")
        if j.get("state") == "error" or (j.get("state") in ("done", "none") and j.get("preview_url")):
            return j
        time.sleep(5)
    return {"state": "timeout"}


def suggest(verdict, status, asset_id, base):
    """고치지 않는 대신, 사람이 그대로 실행할 수 있는 rebuild 명령을 제안으로 만든다."""
    q = urllib.parse.quote(asset_id, safe="")
    cur_up, cur_flip = status.get("up"), status.get("flip")
    if verdict == "upside_down":
        flip = 0 if str(cur_flip) == "1" else 1
        return [f"curl -X POST \"{base}/api/preview/rebuild?asset={q}&flip={flip}\""]
    if verdict == "lying_down":
        cands = [u for u in ("Z", "X", "Y") if u != (cur_up or "").upper()]
        return [f"curl -X POST \"{base}/api/preview/rebuild?asset={q}&up={u}\"  # 후보 {i + 1}"
                for i, u in enumerate(cands[:2])]
    return []


# ── 에이전트 본체 ──────────────────────────────────────────────────────────────
def judge_image(image, workdir, a, log):
    """그림 한 장을 4회전 비교로 판정. (verdict, 근거, 라운드수) 를 돌려준다."""
    workdir.mkdir(parents=True, exist_ok=True)
    rots = make_rotations(image, workdir, log)
    if not rots:                                       # WSL 이 없으면 예/아니오 한 번으로만
        v = look([image], YES_PROMPT, YES_SCHEMA, a, log) or {}
        up = v.get("upright")
        return ("upright" if up else "unsure" if up is None else "lying_down"), v.get("reason", ""), 1

    v = look(rots, PICK_PROMPT, PICK_SCHEMA, a, log) or {}
    verdict, fix_ccw = PICK.get(v.get("best"), ("unsure", None))
    reason = v.get("reason", "")
    if verdict == "upright":
        return verdict, reason, 1

    # [행동→재지각] 모델이 고른 회전을 적용한 그림으로 "이제 바로 섰나?" 재확인
    v2 = look([workdir / f"rot_{fix_ccw}.png"], YES_PROMPT, YES_SCHEMA, a, log) or {}
    if v2.get("upright") is True:
        return verdict, f"{reason} / 교정본 재확인 통과: {v2.get('reason', '')}", 2
    return "unsure", f"{reason} / 교정본 재확인 실패: {v2.get('reason', '')}", 2


def check_asset(asset_id, a, log):
    q = urllib.parse.quote(asset_id, safe="")
    card = get_json(f"{a.base}/api/cards/{q}")
    title = card.get("title_ko") or asset_id
    if card.get("media_type") != "3d":
        print(f"  [건너뜀] {title} — 3D 카드가 아님")
        return None
    status = wait_preview(asset_id, a.base)
    if not card.get("thumb_url"):
        card = get_json(f"{a.base}/api/cards/{q}")     # 미리보기 직후라면 이제 생겼다
    if not card.get("thumb_url"):
        print(f"  [보류] {title} — 썸네일이 아직 없음 (state={status.get('state')})")
        log("skip", asset=asset_id, why="no-thumb", state=status.get("state"))
        return None

    work = Path(a.workdir) / re.sub(r"[^A-Za-z0-9_.-]", "__", asset_id)
    work.mkdir(parents=True, exist_ok=True)
    thumb = work / "thumb.png"
    thumb.write_bytes(_req(a.base + card["thumb_url"], timeout=120))

    verdict, reason, rounds = judge_image(thumb, work, a, log)

    # 애매하면 스스로 서버 미리보기 GLB 를 받아 정면 렌더로 한 번 더 본다
    if verdict == "unsure" and a.max_rounds > rounds and status.get("preview_url"):
        front = render_front(status["preview_url"], work, a.base, log)
        if front:
            verdict, reason2, r2 = judge_image(front, work / "r2", a, log)
            reason, rounds = f"{reason} → 렌더 재판정: {reason2}", rounds + r2

    tips = suggest(verdict, status, asset_id, a.base)
    mark = {"upright": "○", "unsure": "?"}.get(verdict, "★")
    print(f"  [{mark}] {title}")
    print(f"      판정 {VERDICT_KO[verdict]} · {rounds}라운드 — {reason}")
    for t in tips:
        print(f"      제안: {t}")
    log("verdict", asset=asset_id, title=title, verdict=verdict, reason=reason,
        rounds=rounds, up=status.get("up"), flip=status.get("flip"), tips=tips)
    return verdict


def cmd_check(a, log):
    assets = list(a.asset)
    if a.all:
        cards = get_json(f"{a.base}/api/cards?limit=500")["items"]
        assets = [c["asset_id"] for c in cards if c.get("media_type") == "3d"]
        print(f"3D 카드 {len(assets)}개를 점검한다 (개당 수십 초)")
    bad = err = 0
    for aid in assets:
        try:
            v = check_asset(aid, a, log)
        except Exception as e:                      # 한 건의 실패가 배치를 죽이지 않게
            err += 1
            print(f"  [오류] {aid} — {type(e).__name__}: {str(e)[:120]}")
            log("error", asset=aid, error=f"{type(e).__name__}: {e}")
            continue
        if v and v != "upright":
            bad += 1
    print(f"\n끝 — {len(assets)}개 중 확인 필요 {bad}개 · 오류 {err}개. 기록: {a.logfile}")


def cmd_upload(a, log):
    fields = {"method": a.method, "owner": a.owner, "note": a.note,
              "label": a.label, "overwrite": "true" if a.overwrite else "false"}
    files = [("files", p) for p in a.glb]
    url = f"{a.base}/api/artifacts/{urllib.parse.quote(a.artifact)}/restored"
    print(f"[에이전트 등록] {a.artifact} ← {', '.join(Path(p).name for p in a.glb)}")
    try:
        r = post_multipart(url, fields, files)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        print(f"  업로드 실패 HTTP {e.code}: {detail}")
        log("upload_error", artifact=a.artifact, code=e.code, detail=detail)
        sys.exit(1)
    log("upload", artifact=a.artifact,
        resp={k: r.get(k) for k in ("ok", "asset_id", "variant", "slug", "files", "warning")})
    print(f"  올림 → asset_id={r['asset_id']}" + (f"  경고: {r['warning']}" if r.get("warning") else ""))
    print("  미리보기를 기다렸다가 회전을 점검한다…")
    check_asset(r["asset_id"], a, log)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--ollama", default=OLLAMA)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-rounds", type=int, default=4, help="유물 하나에 모델을 부르는 상한")
    ap.add_argument("--workdir", default=str(HERE / "work"))
    ap.add_argument("--logfile", default=str(HERE / "logs" / time.strftime("orient-%Y%m%d.jsonl")))
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="회전 점검만 (보고 전용)")
    c.add_argument("asset", nargs="*", help="카드 asset_id (예: new20260918_02/3d/source)")
    c.add_argument("--all", action="store_true", help="서버의 3D 카드 전부")

    u = sub.add_parser("upload", help="복원본을 올리고 곧바로 점검 (에이전트 등록)")
    u.add_argument("artifact", help="원본 유물 ID (originals/<ID>)")
    u.add_argument("glb", nargs="+", help="올릴 파일들 (GLB + 같이 갈 텍스처)")
    u.add_argument("--label", required=True, help="복원본 이름 — 카드 제목에 그대로 붙는다")
    u.add_argument("--method", default=None)
    u.add_argument("--owner", default=None)
    u.add_argument("--note", default="orient_agent 로 등록")
    u.add_argument("--overwrite", action="store_true")

    a = ap.parse_args()
    if not a.base:
        ap.error("서버 주소가 없습니다 — infra/web/.env 에 SERVER_HOST 를 넣거나 (.env.example) --base 를 주세요")
    if a.cmd == "check" and not (a.asset or a.all):
        ap.error("asset_id 를 주거나 --all 을 쓰세요")

    Path(a.logfile).parent.mkdir(parents=True, exist_ok=True)

    def log(event, **kw):
        with open(a.logfile, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **kw},
                               ensure_ascii=False) + "\n")

    log("start", cmd=a.cmd, model=a.model, argv=sys.argv[1:])
    (cmd_check if a.cmd == "check" else cmd_upload)(a, log)


if __name__ == "__main__":
    main()
