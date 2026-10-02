"""문화유산 3D 자료 카탈로그 API. FastAPI + SQLite.

로컬 개발 환경을 위한 도구이며 외부 공개 시 별도의 인증 계층이 필요하다.

유저플로우 → 엔드포인트
  ① 2D/3D 선택                GET /api/stats                       (카드 수·용량)
  ② 자료 목록 (한글명)          GET /api/cards?media=3d
  ③ 검색                      GET /api/cards?media=3d&q=반가
  ④ 파일 → 미리보기·다운로드     GET /api/cards/{asset_id}            (files[].url / preview_path)
  유물 단위 (원본↔복원 비교)     GET /api/artifacts/{artifact_id}
  메타 수정                    PATCH /api/artifacts/{id} · PATCH /api/cards/{asset_id}
  파일 넣은 뒤                  POST /api/rescan

정적 파일(/files, /preview)은 nginx 가 직접 낸다. 이 앱은 JSON 만 만든다.
실행:  uvicorn app.main:app --host 127.0.0.1 --port 8412
"""
import os, sys, subprocess, threading, time, json, re, shutil, hashlib
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query, Body, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from . import db

ROOT = os.environ.get("CATALOG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILES_DIR = os.environ.get("CATALOG_FILES", os.path.join(ROOT, "files"))
PREVIEW_DIR = os.environ.get("CATALOG_PREVIEW", os.path.join(ROOT, "preview"))

app = FastAPI(title="본디 자료 서버", version="0.1", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_rescan_lock = threading.Lock()
TRASH_DIR_NAME = "_trash"        # 삭제한 자료를 옮겨 두는 곳 (files/_trash/<날짜시각>_<유물ID>/)


def _vurl(path):
    """내용이 바뀌면 URL 도 바뀌게 수정시각(?v=)을 붙인다.
    nginx 는 /preview/ 에 7일, /files/ 에 1시간 캐시를 준다(용량이 크니 그게 맞다).
    그런데 유물을 지우고 같은 ID 로 다시 등록하면 파일 경로가 똑같아져서,
    브라우저가 **지운 자료의 옛 사진·옛 GLB** 를 그대로 보여준다. 실제로 그 일이 있었다."""
    if path.startswith("preview/"):
        base, real = "/" + path, os.path.join(PREVIEW_DIR, path[len("preview/"):])
    else:
        base, real = "/files/" + path, os.path.join(FILES_DIR, path)
    try:
        return "%s?v=%d" % (base, os.path.getmtime(real))
    except OSError:
        return base


def _url(path):
    return _vurl(path)


def _card_out(c):
    c = dict(c)
    c["thumb_url"] = _vurl(c["thumb_path"]) if c.get("thumb_path") else None
    c["preview_url"] = _vurl(c["preview_path"]) if c.get("preview_path") else None
    # 복원 종류 — 코드와 한글 이름을 같이 준다 (화면이 배지로 찍는다)
    codes = [t for t in (c.get("restore_types") or "").split(",") if t.strip()]
    c["restore_types"] = codes
    c["restore_types_ko"] = [db.RESTORE_TYPES.get(t, t) for t in codes]
    c.pop("search_text", None)
    return c


def _scan_out(s):
    if not s:
        return s
    s = dict(s)
    for k in ("started_at", "finished_at"):
        s[k] = _utc(s.get(k))
    return s


def _file_out(f):
    f = dict(f)
    f["url"] = _url(f["path"])
    p = f.get("preview_path")
    f["preview_url"] = None if not p else _vurl(p)
    return f


WEB_DIR = os.path.join(ROOT, "web")


def _asset_version():
    """화면 파일들의 수정시각 합. 배포로 화면이 바뀌면 값이 달라진다 —
    열어 둔 탭이 옛 화면을 들고 있는 것을 스스로 알아채게 하기 위한 것."""
    t = 0
    for f in ("index.html", "app.js", "app.css", "viewer.js"):
        try:
            t += int(os.path.getmtime(os.path.join(WEB_DIR, f)))
        except OSError:
            pass
    return t


@app.get("/api/health")
def health():
    return {"ok": True, "db": db.DB, "files_root": FILES_DIR, "preview_root": PREVIEW_DIR,
            "asset_version": _asset_version()}


@app.get("/api/stats")
def stats():
    by_media = {r["media_type"]: r for r in db.rows(
        "SELECT media_type, COUNT(*) AS cards, SUM(file_count) AS files, SUM(total_bytes) AS bytes, "
        "SUM(CASE WHEN preview_kind!='none' THEN 1 ELSE 0 END) AS with_preview FROM assets GROUP BY media_type")}
    return {
        "artifacts": db.one("SELECT COUNT(*) AS n FROM artifacts")["n"],
        "by_media": by_media,
        "by_source": {r["source_org"]: r["n"] for r in db.rows("SELECT source_org, COUNT(*) AS n FROM artifacts GROUP BY 1")},
        "by_license": {r["license_status"]: r["n"] for r in db.rows("SELECT license_status, COUNT(*) AS n FROM artifacts GROUP BY 1")},
        # 실재하는 카드만 센다 — 유물을 지워도 favorites 행은 남아서(FK 없음, 재스캔 CASCADE 방지)
        # 상단 뱃지는 3 인데 목록은 2건인 일이 있었다
        "favorites": db.one("SELECT COUNT(*) AS n FROM favorites f JOIN assets a USING(asset_id)")["n"],
        "vr": db.one("SELECT COUNT(*) AS n FROM v_cards WHERE vr=1")["n"],
        "vr_starred": db.one("SELECT COUNT(*) AS n FROM v_cards WHERE vr=1 AND starred=1")["n"],
        "restore_types": {t: db.one(
            "SELECT COUNT(*) AS n FROM v_cards WHERE (',' || COALESCE(restore_types,'') || ',') LIKE ?",
            "%," + t + ",%")["n"] for t in db.RESTORE_TYPES},
        "feedback": db.one("SELECT COUNT(*) AS n FROM feedback")["n"],
        "feedback_open": db.one("SELECT COUNT(*) AS n FROM feedback WHERE status IN ('new','seen','hold','doing')")["n"],
        "last_scan": _scan_out(db.one("SELECT * FROM scans ORDER BY id DESC LIMIT 1")),
    }


# ───────────────────────────── 피드백 게시판 (팀 공용) ─────────────────────────
# 자유롭게 적고 채팅처럼 쌓인다. 항목마다 처리 상태를 바꾼다.
FB_STATUS = {"new": "미확인", "seen": "확인", "hold": "대기", "doing": "진행", "done": "완료"}
FB_OPEN = ("new", "seen", "hold", "doing")  # 아직 안 끝난 것 — 버튼 배지에 쓴다


def _utc(s):
    """SQLite CURRENT_TIMESTAMP 는 UTC 다. 화면이 문자열을 그대로 찍는 바람에
    한국 시각보다 9시간 이른 시각이 떴다. UTC 임을 표시해 내보내고 시각 변환은 브라우저에 맡긴다."""
    if not s:
        return s
    s = str(s).replace(" ", "T")
    return s if s.endswith("Z") or "+" in s else s + "Z"


def _fb_out(r):
    r = dict(r)
    r["status_ko"] = FB_STATUS.get(r["status"], r["status"])
    for k in ("created_at", "updated_at"):
        r[k] = _utc(r.get(k))
    return r


@app.get("/api/feedback")
def feedback_list(status: Optional[str] = Query(None, pattern="^(new|seen|hold|doing|done|open)$"),
                  limit: int = Query(300, ge=1, le=1000)):
    """오래된 것부터 (채팅처럼 위에서 아래로 읽는다)."""
    w, args = "", []
    if status == "open":
        w, args = "WHERE status IN (%s)" % ",".join("?" * len(FB_OPEN)), list(FB_OPEN)
    elif status:
        w, args = "WHERE status=?", [status]
    rows = db.rows("SELECT * FROM feedback %s ORDER BY id DESC LIMIT ?" % w, *args, limit)
    marks = ",".join("?" * len(FB_OPEN))
    return {"total": db.one("SELECT COUNT(*) AS n FROM feedback")["n"],
            "open": db.one("SELECT COUNT(*) AS n FROM feedback WHERE status IN (%s)" % marks, *FB_OPEN)["n"],
            "items": [_fb_out(r) for r in reversed(rows)]}


@app.post("/api/feedback")
def feedback_add(body: dict = Body(...)):
    text = (body.get("body") or "").strip()
    if not text:
        raise HTTPException(400, "내용을 적어 주세요")
    if len(text) > 4000:
        raise HTTPException(400, "4000자까지 적을 수 있습니다")
    st = body.get("status") or "new"
    if st not in FB_STATUS:
        raise HTTPException(400, "status: %s" % list(FB_STATUS))
    # 시각을 여기서 못박는다 — 이미 만들어진 DB 의 열 기본값은 'localtime' 이라, 그대로 두면
    # 서버(UTC)와 개발용 PC(KST)가 서로 다른 시각을 적는다. 저장은 늘 UTC, 변환은 브라우저가.
    db.write("INSERT INTO feedback(body, author, status, created_at) VALUES(?,?,?,datetime('now'))",
             text, (body.get("author") or "").strip() or None, st)
    return _fb_out(db.one("SELECT * FROM feedback ORDER BY id DESC LIMIT 1"))


@app.patch("/api/feedback/{fb_id}")
def feedback_patch(fb_id: int, body: dict = Body(...)):
    """상태 변경, 또는 내용·작성자 수정."""
    sets, args = [], []
    if "status" in body:
        if body["status"] not in FB_STATUS:
            raise HTTPException(400, "status: %s" % list(FB_STATUS))
        sets.append("status=?"); args.append(body["status"])
    for k in ("body", "author"):
        if k in body:
            v = (body[k] or "").strip()
            if k == "body" and not v:
                raise HTTPException(400, "내용은 비울 수 없습니다")
            sets.append("%s=?" % k); args.append(v or None)
    if not sets:
        raise HTTPException(400, "status / body / author 중 하나를 주세요")
    n = db.write("UPDATE feedback SET %s, updated_at=datetime('now') WHERE id=?" % ", ".join(sets),
                 *args, fb_id)
    if not n:
        raise HTTPException(404, "그런 피드백이 없습니다")
    return _fb_out(db.one("SELECT * FROM feedback WHERE id=?", fb_id))


@app.delete("/api/feedback/{fb_id}")
def feedback_delete(fb_id: int):
    if not db.write("DELETE FROM feedback WHERE id=?", fb_id):
        raise HTTPException(404, "그런 피드백이 없습니다")
    return {"ok": True, "id": fb_id}


# ───────────────────────────── 즐겨찾기 (팀 공용) ─────────────────────────────
@app.get("/api/favorites")
def favorites_list():
    """즐겨찾기한 카드 전부 (2D·3D 섞여서). 최근 찍은 것부터."""
    return {"items": [_card_out(c) for c in db.rows("SELECT * FROM v_cards WHERE starred=1 ORDER BY registered_at DESC, starred_at DESC")]}


@app.put("/api/favorites/{asset_id:path}")
def favorite_add(asset_id: str, body: Optional[dict] = Body(None)):
    if not db.one("SELECT 1 FROM assets WHERE asset_id=?", asset_id):
        raise HTTPException(404, "card not found")
    db.write("INSERT INTO favorites(asset_id, note) VALUES(?,?) ON CONFLICT(asset_id) DO UPDATE SET note=COALESCE(excluded.note, favorites.note)",
             asset_id, (body or {}).get("note"))
    return {"asset_id": asset_id, "starred": 1, "total": db.one("SELECT COUNT(*) AS n FROM favorites")["n"]}


@app.delete("/api/favorites/{asset_id:path}")
def favorite_remove(asset_id: str):
    db.write("DELETE FROM favorites WHERE asset_id=?", asset_id)
    return {"asset_id": asset_id, "starred": 0, "total": db.one("SELECT COUNT(*) AS n FROM favorites")["n"]}


@app.get("/api/cards")
def cards(media: Optional[str] = Query(None, pattern="^(2d|3d)$"), q: Optional[str] = None,
          source_org: Optional[str] = None, license: Optional[str] = None, variant: Optional[str] = None,
          restore_type: Optional[str] = Query(None, description="복원 종류: shape|color|img2mesh|symmetry|pointr"),
          vr: Optional[bool] = None,
          has_preview: Optional[bool] = None, starred: Optional[bool] = None,
          sort: str = Query("recent", pattern="^(title|accession|size|recent|starred|vr)$"),   # 기본은 최신 등록순
          limit: int = Query(60, ge=1, le=500), offset: int = Query(0, ge=0)):
    where, args = [], []
    if starred is not None:
        where.append("starred=?"); args.append(1 if starred else 0)      # 즐겨찾기만 / 제외
    if vr is not None:
        where.append("vr=?"); args.append(1 if vr else 0)                # VR 에서 볼 수 있는 것만
    if restore_type:
        if restore_type not in db.RESTORE_TYPES:
            raise HTTPException(400, "restore_type: %s" % list(db.RESTORE_TYPES))
        # 쉼표 목록에서 한 칸을 정확히 찾는다 ('color' 가 'colorize' 에 걸리지 않게 양끝에 쉼표를 붙여 비교)
        where.append("(',' || COALESCE(restore_types,'') || ',') LIKE ?"); args.append("%," + restore_type + ",%")
    if media:
        where.append("media_type=?"); args.append(media)
    if q:
        for term in q.lower().split():
            where.append("search_text LIKE ?"); args.append("%" + term + "%")
    if source_org:
        where.append("source_org=?"); args.append(source_org)
    if license:
        where.append("license_status=?"); args.append(license)
    if variant:
        where.append("variant=?"); args.append(variant)
    if has_preview is not None:
        where.append("preview_kind" + ("!=" if has_preview else "=") + "'none'")
    w = ("WHERE " + " AND ".join(where)) if where else ""
    # recent: 등록 시각 내림차순. 3D·2D·즐겨찾기 모두 이 순서로 보여 달라는 요청.
    # registered_at 이 비어 있으면(재스캔 전 옛 DB) asset_id 로 떨어진다.
    order = {"title": "name_ko, variant", "accession": "accession, variant", "size": "total_bytes DESC",
             "recent": "registered_at DESC, asset_id DESC", "starred": "registered_at DESC, starred_at DESC",
             # 시연 흐름: 즐겨찾기 안에서도 VR 로 볼 수 있는 것을 맨 앞에 (피드백 2026-09-17)
             "vr": "vr DESC, registered_at DESC, asset_id DESC"}[sort]
    total = db.one("SELECT COUNT(*) AS n FROM v_cards " + w, *args)["n"]
    items = db.rows("SELECT * FROM v_cards %s ORDER BY %s LIMIT ? OFFSET ?" % (w, order), *args, limit, offset)
    return {"total": total, "limit": limit, "offset": offset, "items": [_card_out(c) for c in items]}


@app.get("/api/cards/{asset_id:path}")
def card(asset_id: str):
    c = db.one("SELECT * FROM v_cards WHERE asset_id=?", asset_id)
    if not c:
        raise HTTPException(404, "card not found")
    out = _card_out(c)
    out["artifact"] = db.one("SELECT * FROM artifacts WHERE artifact_id=?", c["artifact_id"])
    out["artifact"].pop("search_text", None)
    out["files"] = [_file_out(f) for f in db.rows("SELECT * FROM files WHERE asset_id=? ORDER BY sort_order", asset_id)]
    out["siblings"] = [_card_out(s) for s in db.rows(
        "SELECT * FROM v_cards WHERE artifact_id=? AND asset_id!=? ORDER BY media_type, variant", c["artifact_id"], asset_id)]
    return out


@app.get("/api/artifacts/{artifact_id}")
def artifact(artifact_id: str):
    a = db.one("SELECT * FROM artifacts WHERE artifact_id=?", artifact_id)
    if not a:
        raise HTTPException(404, "artifact not found")
    a.pop("search_text", None)
    a["cards"] = [_card_out(c) for c in db.rows("SELECT * FROM v_cards WHERE artifact_id=? ORDER BY media_type, variant", artifact_id)]
    for c in a["cards"]:
        c["files"] = [_file_out(f) for f in db.rows("SELECT * FROM files WHERE asset_id=? ORDER BY sort_order", c["asset_id"])]
    return a


def _patch(table, key_col, key, body, editable):
    bad = [k for k in body if k not in editable]
    if bad:
        raise HTTPException(400, "editable: %s (got %s)" % (list(editable), bad))
    if not body:
        raise HTTPException(400, "empty body")
    sets = ", ".join("%s=?" % k for k in body) + ", updated_at=datetime('now')"
    n = db.write("UPDATE %s SET %s WHERE %s=?" % (table, sets, key_col), *body.values(), key)
    if n == 0:
        raise HTTPException(404, "not found")


@app.patch("/api/artifacts/{artifact_id}")
def patch_artifact(artifact_id: str, body: dict = Body(...)):
    _patch("artifacts", "artifact_id", artifact_id, body, db.ARTIFACT_EDITABLE)
    a = db.one("SELECT * FROM artifacts WHERE artifact_id=?", artifact_id); a.pop("search_text", None)
    return a


@app.patch("/api/cards/{asset_id:path}")
def patch_card(asset_id: str, body: dict = Body(...)):
    _patch("assets", "asset_id", asset_id, body, db.ASSET_EDITABLE)
    return _card_out(db.one("SELECT * FROM v_cards WHERE asset_id=?", asset_id))


@app.get("/api/files")
def files(kind: Optional[str] = None, sha256: Optional[str] = None, limit: int = Query(200, ge=1, le=2000)):
    where, args = [], []
    if kind:
        where.append("kind=?"); args.append(kind)
    if sha256:
        where.append("sha256=?"); args.append(sha256)
    w = ("WHERE " + " AND ".join(where)) if where else ""
    return {"items": [_file_out(f) for f in db.rows("SELECT * FROM files %s ORDER BY path LIMIT ?" % w, *args, limit)]}


ZIPS_DIR = os.environ.get("CATALOG_ZIPS", os.path.join(ROOT, "zips"))
_zip_locks = {}
_zip_locks_guard = threading.Lock()
GROUP_KO = {"digital_obj": "모델 세트 (OBJ+MTL+텍스처)", "scan_ply": "점군·메시 (PLY)", "print_stl": "프린트용 (STL)",
            "photo_2d": "사진 전체", "records": "복원 리포트", "pointcloud_las": "점군 (LAS)", "all": "이 카드의 모든 파일"}


def _group_of(path, artifact_folder):
    """files.path 에서 유물 폴더 바로 아래 디렉터리명. originals/<유물>/digital_obj/x.obj → digital_obj"""
    parts = path.split("/")
    try:
        i = parts.index(artifact_folder)
        return parts[i + 1] if len(parts) > i + 2 else "_"
    except ValueError:
        return parts[-2] if len(parts) > 1 else "_"


@app.get("/api/zip")
def zip_group(asset: str, group: str = "all"):
    """카드의 한 폴더(또는 전체)를 ZIP 으로 묶어 /zips/ 로 보낸다. OBJ 는 MTL·텍스처와 함께 있어야 열리므로 폴더 단위가 기본 단위다.

    ZIP 은 /srv/catalog/zips/ 에 만들어 두고(원본이 더 새로우면 다시 만듦) nginx 가 직접 내준다 — 큰 파일이 앱을 통과하지 않는다.
    ZIP 안에는 `<유물폴더>/<그룹>/파일` 구조를 유지해 풀었을 때 상대경로가 그대로 맞는다.
    """
    import zipfile
    from fastapi.responses import RedirectResponse
    c = db.one("SELECT asset_id, artifact_id, variant FROM assets WHERE asset_id=?", asset)
    if not c:
        raise HTTPException(404, "card not found")
    files = db.rows("SELECT path, size FROM files WHERE asset_id=? ORDER BY path", asset)
    folder = files[0]["path"].split("/")[1] if files else c["artifact_id"]     # originals/<유물폴더>/...
    if group != "all":
        files = [f for f in files if _group_of(f["path"], folder) == group]
    if not files:
        raise HTTPException(404, "no files in group %r" % group)
    acc = db.one("SELECT accession FROM artifacts WHERE artifact_id=?", c["artifact_id"])["accession"] or c["artifact_id"]
    name = "%s_%s_%s.zip" % (acc, c["variant"], group)
    os.makedirs(ZIPS_DIR, exist_ok=True)
    out = os.path.join(ZIPS_DIR, name)
    newest = max(os.path.getmtime(os.path.join(FILES_DIR, f["path"])) for f in files)
    with _zip_locks_guard:
        lock = _zip_locks.setdefault(name, threading.Lock())
    with lock:
        if not os.path.exists(out) or os.path.getmtime(out) < newest:
            tmp = out + ".part"
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:      # 이미 압축된 jpg 가 대부분 → 저장만
                for f in files:
                    src = os.path.join(FILES_DIR, f["path"])
                    arc = "/".join(f["path"].split("/")[1:])                              # <유물폴더>/<그룹>/파일
                    z.write(src, arc)
            os.replace(tmp, out)
    return RedirectResponse("/zips/" + name, status_code=302)


@app.get("/api/groups")
def card_groups(asset: str):
    """카드 안의 폴더(그룹)별 파일 수·용량 — 화면의 'ZIP 받기' 버튼용.
    (/api/cards/{asset_id:path} 가 경로를 전부 먹으므로 하위 경로 대신 쿼리 파라미터를 쓴다.)"""
    asset_id = asset
    files = db.rows("SELECT path, size, kind FROM files WHERE asset_id=? ORDER BY path", asset_id)
    if not files:
        raise HTTPException(404, "card not found")
    folder = files[0]["path"].split("/")[1]
    g = {}
    for f in files:
        k = _group_of(f["path"], folder)
        e = g.setdefault(k, {"group": k, "label_ko": GROUP_KO.get(k, k), "count": 0, "bytes": 0, "kinds": set()})
        e["count"] += 1; e["bytes"] += f["size"]; e["kinds"].add(f["kind"])
    out = [dict(v, kinds=sorted(v["kinds"]), zip_url="/api/zip?asset=%s&group=%s" % (asset_id, k)) for k, v in g.items()]
    out.append({"group": "all", "label_ko": GROUP_KO["all"], "count": len(files), "bytes": sum(f["size"] for f in files),
                "kinds": sorted({f["kind"] for f in files}), "zip_url": "/api/zip?asset=%s&group=all" % asset_id})
    return {"folder": folder, "groups": out}


# ─────────────────── 복원본 등록 (원본 상세에서 파일만 올리면 된다) ───────────────────
# 사용자가 올리는 것: 복원 결과 파일들. 나머지(유물명·시대·재질·출처·라이선스)는 원본 _meta.json 에서 상속한다.
# 파일명·폴더는 규칙대로 서버가 정하고, OBJ 의 mtllib 과 MTL 의 텍스처 경로를 새 이름으로 다시 쓴다.
# 복원본 이름(라벨) → 폴더 슬러그. 영문·숫자·-_ 만 남긴다 (폴더명·asset_id·URL 세 군데에 들어가므로).
# 한글 라벨은 여기서 전부 걸러져 빈 문자열이 된다 — 그대로 두면 라벨을 안 준 것과 구분이 안 돼
# 기본 칸(restored)으로 떨어지고, 적은 이름이 화면에서 사라진다 (2026-09-21 "색 보정"에서 드러남).
# 그래서 걸러져 비면 라벨 해시로 대체 슬러그를 만든다 — ASCII 는 유지하면서 자기 칸을 갖는다.
# 같은 라벨은 늘 같은 슬러그라 다시 올리면 같은 칸으로 간다 (영문 라벨과 동작이 같다).
RESTORED_BAD = re.compile(r"[^a-z0-9._-]+")


def restored_slug(label):
    """라벨 → 슬러그. 빈 라벨이면 "" (기본 칸), 걸러져 비면 ko-<해시6>."""
    lab = (label or "").strip()
    slug = RESTORED_BAD.sub("", lab.lower().replace(" ", "-")).strip("-_")[:40]
    if not slug and lab:
        slug = "ko-" + hashlib.sha1(lab.encode("utf-8")).hexdigest()[:6]
    return slug
UPLOAD_EXT = {".obj", ".mtl", ".ply", ".stl", ".glb", ".jpg", ".jpeg", ".png", ".json", ".txt", ".md", ".npy", ".las"}
UPLOAD_MAX_TOTAL = 2 * 1024 ** 3          # 한 번에 2GB 까지
MODEL_EXT = {".obj", ".ply", ".stl", ".glb", ".las"}   # 이게 하나라도 있으면 같이 올린 이미지는 텍스처로 본다
TEXT_EXT = {".obj", ".mtl"}

# 업로드 직후 3D 미리보기(GLB·썸네일)를 서버에서 만든다.
# GLB 생성은 GPU 가 필요 없는 CPU 작업이라 이 서버(4코어)로 충분하다 — Blender 5.2.1 을 /opt 에 설치해 뒀다.
BLENDER = os.environ.get("BLENDER", "/usr/local/bin/blender")
PREVIEW_TARGET_TRIS = 200_000
# 진행 상태는 메모리가 아니라 파일에 적는다.
# uvicorn 이 워커 2개로 돌아서, 올린 워커와 상태를 묻는 워커가 다를 수 있다 —
# 메모리에 두면 절반은 "생성 중"도 "실패"도 보지 못한다. 재시작해도 남는 이점도 있다.
JOBS_DIR = os.environ.get("CATALOG_JOBS", os.path.join(ROOT, "jobs"))
# 세워놓는 방향을 사람이 고정해 두는 곳. blender_preview.py 의 자동 추정은 "납작한 쪽을
# 바닥으로" 놓는데, 빗살무늬토기는 납작한 쪽이 아가리고 굽다리바리는 위아래가 다 납작해서
# 거꾸로 서는 것이 나온다. 틀린 카드만 적어 둔다.
#   ORIENT_BASE : 코드와 함께 커밋해 둔 목록. 육안 확인한 것을 리뷰를 거쳐 넣는다.
#                 local/make_preview.py 가 쓰는 파일과 같은 형식이다.
#   ORIENT_FILE : 서버에서 API 로 고친 것. 자료 폴더 옆에 있어 배포·재스캔에도 남고
#                 같은 카드가 양쪽에 있으면 이쪽이 이긴다 (배포 없이 바로잡을 수 있게).
ORIENT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orientation_overrides.json")
ORIENT_FILE = os.environ.get("CATALOG_ORIENT", os.path.join(ROOT, "orientation.json"))
PREVIEW_TIMEOUT = 1800                    # Blender 한 번에 30분까지
_preview_lock = threading.Lock()          # 같은 워커 안에서 파일 쓰기 직렬화


def _orient_read(path):
    """{asset_id: {up, flip}} 만 남긴다. '_note' 같은 설명 키는 버린다."""
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            return {}
        return {k: v for k, v in d.items() if not k.startswith("_") and isinstance(v, dict)}
    except Exception:      # 없거나 깨졌으면 없는 것으로 — 자동 추정으로 돌아간다
        return {}


def _orient_all():
    d = _orient_read(ORIENT_BASE)
    for k, v in _orient_read(ORIENT_FILE).items():
        d[k] = dict(d.get(k, {}), **v)
    return d


def _orient_get(aid):
    """그 카드에 고정해 둔 방향. 없으면 (None, None, None) — 자동 추정에 맡긴다.
    yaw 는 세로축 둘레 회전(도) — up/flip 으로는 못 고치는 "옆을 보고 선" 것을 맞춘다."""
    v = _orient_all().get(aid)
    if not isinstance(v, dict):
        return None, None, None
    up = v.get("up")
    up = up.upper() if isinstance(up, str) and up.upper() in ("X", "Y", "Z") else None
    fl = v.get("flip")
    fl = "1" if fl in (1, "1", True) else ("0" if fl in (0, "0", False) else None)
    try:
        yaw = float(v["yaw"]) if v.get("yaw") not in (None, "") else None
    except (TypeError, ValueError):
        yaw = None
    return up, fl, yaw


def _orient_set(aid, changes):
    """ORIENT_FILE 에만 쓴다. changes 에 담긴 항목만 바꾼다 — 값이 있으면 고정,
    None 이면 그 항목의 덮어쓰기를 지운다(커밋된 목록에 있으면 그 값으로 돌아간다).
    없는 키는 건드리지 않는다 (flip 만 고쳐도 up 고정이 살아 있어야 한다).
    임시파일로 갈아치워서 워커가 2개여도 반쪽만 쓰인 파일이 남지 않는다."""
    d = _orient_read(ORIENT_FILE)
    cur = dict(d.get(aid)) if isinstance(d.get(aid), dict) else {}
    for k, val in changes.items():
        if val is None:
            cur.pop(k, None)
        else:
            cur[k] = val
    if cur:
        d[aid] = cur
    else:
        d.pop(aid, None)
    tmp = ORIENT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(d, f, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, ORIENT_FILE)
    return d.get(aid)


def _job_file(aid):
    # asset_id 는 쿼리 파라미터로 들어오므로 경로 구분자를 모두 없애 폴더를 벗어나지 못하게 한다
    safe = re.sub(r"[^0-9A-Za-z가-힣._-]", "_", aid.replace("/", "__"))
    return os.path.join(JOBS_DIR, safe + ".json")


def _job_set(aid, data):
    """상태 기록. 읽는 쪽이 반쪽짜리 파일을 보지 않도록 tmp 후 교체."""
    os.makedirs(JOBS_DIR, exist_ok=True)
    p = _job_file(aid)
    with _preview_lock:
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, p)


def _job_get(aid):
    try:
        with open(_job_file(aid), encoding="utf-8") as f:
            j = json.load(f)
    except Exception:
        return {"state": "none"}
    # 워커가 죽어 running 인 채 남은 것은 실패로 본다 (화면이 영원히 기다리지 않게)
    if j.get("state") == "running" and time.time() - j.get("started", 0) > PREVIEW_TIMEOUT + 120:
        return {"state": "failed", "error": "생성이 끝나지 않았습니다 (시간 초과)"}
    return j


def _claim_preview(aid):
    """이미 만드는 중이면 False. 워커가 여러 개라 잠금 파일(O_EXCL)로 중복 실행을 막는다."""
    os.makedirs(JOBS_DIR, exist_ok=True)
    lock = _job_file(aid) + ".lock"
    for _ in range(2):
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return True
        except FileExistsError:
            try:
                stale = time.time() - os.path.getmtime(lock) > PREVIEW_TIMEOUT + 120
            except OSError:
                stale = True
            if not stale:
                return False
            try:                      # 워커가 죽어 남은 잠금은 치우고 다시 시도
                os.remove(lock)
            except OSError:
                pass
    return False


def _release_preview(aid):
    try:
        os.remove(_job_file(aid) + ".lock")
    except OSError:
        pass


def _start_preview(artifact_id, variant="restored"):
    """백그라운드로 미리보기 생성 시작. 이미 돌고 있으면 False."""
    aid = "%s/3d/%s" % (artifact_id, variant)
    if not _claim_preview(aid):
        return False
    _job_set(aid, {"state": "running", "started": time.time()})
    threading.Thread(target=_make_preview, args=(artifact_id, variant), daemon=True).start()
    return True


PREVIEW_SRC_ORDER = [("digital_obj", ".obj"), ("digital_obj", ".glb"),
                     ("scan_ply", ".ply"), ("print_stl", ".stl"), ("pointcloud_las", ".las")]
GLB_DIRECT_MAX = 30 * 1024 ** 2      # 이보다 큰 GLB 는 브라우저가 버거우니 줄인 뒤에 보여준다


def _pick(d, ext):
    """폴더에서 대표 파일 하나. 'model' 이 든 이름을 먼저, 그다음 짧은 이름을
    먼저 본다 — 그래야 model.obj 가 중복번호 붙은 model-2.obj 보다 앞선다."""
    cands = [f for f in sorted(os.listdir(d)) if f.lower().endswith(ext)]
    cands.sort(key=lambda f: (0 if "model" in f.lower() else 1, len(f), f))
    return os.path.join(d, cands[0]) if cands else None


def _pick_preview_source(folder):
    """프리뷰로 쓸 대표 파일 하나. OBJ > GLB > PLY > STL > LAS 순."""
    for sub, ext in PREVIEW_SRC_ORDER:
        d = os.path.join(folder, sub)
        if os.path.isdir(d):
            p = _pick(d, ext)
            if p:
                return p
    return None


def _artifact_folder(artifact_id, variant):
    """자료가 실제로 있는 폴더. 네 가지 배치를 다 본다:
      originals/<유물>/            박물관·팀 등록 원본
      restored/<유물>/             복원본 (한 칸)
      restored/<유물>__<슬러그>/    복원본이 여러 개일 때 (variant = restored__<슬러그>)
      aihub/<유물>/<변형>/         AIHub 은 변형이 한 칸 더 들어간다(source·pointr_restored…)

    **정확 비교하면 안 된다** — restored__<슬러그> 가 originals 로 떨어져 원본으로 미리보기가
    만들어지는 사고가 있었다 (2026-09-18).
    """
    cands = []
    if variant.startswith("restored"):
        if "__" in variant:
            cands.append(os.path.join(FILES_DIR, "restored",
                                      artifact_id + "__" + variant.split("__", 1)[1]))
        cands.append(os.path.join(FILES_DIR, "restored", artifact_id))
    else:
        cands.append(os.path.join(FILES_DIR, "originals", artifact_id))
    cands += [os.path.join(FILES_DIR, "aihub", artifact_id, variant),
              os.path.join(FILES_DIR, "originals", artifact_id),
              os.path.join(FILES_DIR, "restored", artifact_id)]
    for d in cands:
        if os.path.isdir(d):
            return d
    return None


def _set_preview(artifact_id, variant, has_png):
    db.write("UPDATE assets SET preview_path=?, preview_kind='glb'" +
             (", thumb_path=COALESCE(thumb_path,?)" if has_png else "") + " WHERE asset_id=?",
             *(["preview/%s/%s.glb" % (artifact_id, variant)] +
               (["preview/%s/%s.png" % (artifact_id, variant)] if has_png else []) +
               ["%s/3d/%s" % (artifact_id, variant)]))


def _publish_glb_direct(artifact_id, variant, folder):
    """올린 GLB 를 그대로 미리보기로 쓴다. GLB 는 브라우저가 바로 읽는 형식이라 변환이 필요 없다 —
    올리는 즉시 돌려 볼 수 있다. 뒤이어 Blender 가 줄인 판본과 썸네일을 만들어 덮어쓴다."""
    d = os.path.join(folder, "digital_obj")
    src = _pick(d, ".glb") if os.path.isdir(d) else None
    if not src or os.path.getsize(src) > GLB_DIRECT_MAX:
        return None
    out_dir = os.path.join(PREVIEW_DIR, artifact_id)
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, variant + ".glb")
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    _set_preview(artifact_id, variant, has_png=False)
    return "preview/%s/%s.glb" % (artifact_id, variant)


def _make_preview(artifact_id, variant="restored"):
    """경량 GLB + 512px 썸네일 생성. 백그라운드 스레드에서 돈다.
    메시(OBJ/GLB/PLY/STL)는 Blender, 점군(LAS)은 순수 파이썬 — Blender 가 LAS 를 못 읽는다."""
    aid = "%s/3d/%s" % (artifact_id, variant)
    folder = _artifact_folder(artifact_id, variant)
    out_dir = os.path.join(PREVIEW_DIR, artifact_id)
    started = _job_get(aid).get("started") or time.time()
    try:
        if not folder:
            raise RuntimeError("자료 폴더를 찾지 못했습니다: %s / %s" % (artifact_id, variant))
        src = _pick_preview_source(folder)
        if not src:
            raise RuntimeError("프리뷰로 쓸 3D 파일이 없습니다 (OBJ·GLB·PLY·STL·LAS 중 하나가 있어야 합니다)")
        os.makedirs(out_dir, exist_ok=True)
        glb = os.path.join(out_dir, variant + ".glb")
        png = os.path.join(out_dir, variant + ".png")
        rep = {}
        if src.lower().endswith(".las"):
            from . import las_preview
            n, xyz, rgb = las_preview.read_las(src)
            if not xyz:
                raise RuntimeError("LAS 에서 점을 읽지 못했습니다")
            las_preview.write_glb_points(glb, xyz, rgb)
            las_preview.thumb_points(png, xyz, rgb)
            rep = {"pt_count": n, "preview_pt_count": len(xyz)}
        else:
            if not os.path.exists(BLENDER):
                raise RuntimeError("Blender 가 없습니다: %s" % BLENDER)
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blender_preview.py")
            cmd = [BLENDER, "-b", "--python-exit-code", "1", "-P", script, "--",
                   src, glb, png, str(PREVIEW_TARGET_TRIS)]
            o_up, o_flip, o_yaw = _orient_get(aid)   # 사람이 고정해 둔 방향이 있으면 자동 추정을 덮어쓴다
            if o_up:
                cmd += ["--up", o_up]
            if o_flip is not None:
                cmd += ["--flip", o_flip]
            if o_yaw:
                cmd += ["--yaw", "%g" % o_yaw]
            p = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=1800)
            if p.returncode != 0 or not os.path.exists(glb):
                tail = (p.stderr or p.stdout).strip().splitlines()[-5:]
                raise RuntimeError("blender rc=%d: %s" % (p.returncode, " | ".join(tail))[:400])
            m = re.search(r"PREVIEW_REPORT (\{.*\})", p.stdout)
            rep = json.loads(m.group(1)) if m else {}
        _set_preview(artifact_id, variant, has_png=os.path.exists(png))
        _job_set(aid, dict(rep, state="done", glb_mb=round(os.path.getsize(glb) / 1048576, 2),
                           took_s=round(time.time() - started, 1),
                           thumb=os.path.exists(png)))
    except Exception as e:
        _job_set(aid, {"state": "failed", "error": str(e)[:400]})
    finally:
        _release_preview(aid)


@app.get("/api/preview-status")
def preview_status(asset: str):
    """업로드 직후 화면이 몇 초마다 물어본다. done 이 되면 뷰어를 켠다."""
    j = _job_get(asset)
    row = db.one("SELECT preview_path FROM assets WHERE asset_id=?", asset)
    # _vurl 로 ?v=수정시각 을 붙인다 — 여기만 빠져 있었다. nginx 가 /preview/ 에 7일 캐시를 주므로
    # 버전 없는 URL 을 돌려주면 다시 만든 직후에도 브라우저가 **옛 GLB** 를 그대로 쓴다 (2026-09-21).
    j["preview_url"] = _vurl(row["preview_path"]) if row and row.get("preview_path") else None
    if j["state"] == "none" and j["preview_url"]:
        j["state"] = "done"
    return j


def _dest_dir(ext, name_low):
    """확장자로 들어갈 폴더를 정한다 (scan.py 의 분류와 같은 규칙)."""
    if ext in (".obj", ".mtl", ".glb"):
        return "digital_obj"
    if ext == ".ply":
        return "scan_ply"
    if ext == ".stl":
        return "print_stl"
    if ext == ".las":
        return "pointcloud_las"
    if ext in (".json", ".txt", ".md", ".npy"):
        return "records"
    if ext in (".jpg", ".jpeg", ".png"):
        # 리포트성 이미지(비교·손상맵)는 records, 그 밖의 이미지는 텍스처로 본다
        return "records" if any(k in name_low for k in ("compare", "damage", "overlay", "report", "diag")) else "digital_obj"
    return "records"


def _role_of(ext, name_low, seen):
    """새 파일명에 쓸 역할. 올린 파일 이름의 힌트를 쓰되 같은 역할이 겹치면 번호를 붙인다."""
    if ext in (".obj", ".glb"):
        r = "model"
    elif ext == ".mtl":
        r = "model_conserved" if "conserv" in name_low else "model"
    elif ext == ".ply":
        r = "scan"
    elif ext == ".stl":
        r = "print"
    elif ext == ".las":
        r = "points"
    elif ext in (".jpg", ".jpeg", ".png"):
        if "normal" in name_low or "_nor" in name_low:
            r = "normal"
        elif "pristine" in name_low:
            r = "diffuse-pristine"
        elif any(k in name_low for k in ("compare", "damage", "overlay", "diag")):
            r = os.path.splitext(os.path.basename(name_low))[0][:40]   # 리포트 이미지는 원래 이름을 살린다
        else:
            r = "diffuse"
    else:
        r = os.path.splitext(os.path.basename(name_low))[0][:40]
    # 중복 번호는 같은 확장자 안에서만 센다.
    # model.obj 와 model.mtl 은 역할 이름이 같아도 파일명이 겹치지 않으므로 둘 다 "model" 이어야 한다.
    key = (r, ext)
    n = seen.get(key, 0) + 1
    seen[key] = n
    return r if n == 1 else "%s-%d" % (r, n)


# ── 새 유물(원본) 등록 ──────────────────────────────────────────────────────────
# 복원본과 달리 상속할 원본이 없다. 이름만 필수로 받고 나머지는 비워도 되게 한다.
NEW_ID_BAD = re.compile(r"[^0-9A-Za-z_-]+")
# 3D 칸과 2D 칸을 따로 받는다. 어느 칸에 넣었는지가 곧 답이라 추측할 것이 없다
# — 예전처럼 이름으로 짐작하면 모델의 텍스처가 '유물 사진'으로 등록되는 사고가 난다.
REPORT_HINT = ("compare", "damage", "overlay", "report", "diag")   # 비교·손상맵류는 어느 칸에 넣든 기록으로


def _dest_2d(ext, name_low):
    """2D 칸: 사진은 photo_2d, 리포트성 이미지만 records."""
    if ext in (".jpg", ".jpeg", ".png"):
        return "records" if any(k in name_low for k in REPORT_HINT) else "photo_2d"
    return _dest_dir(ext, name_low)


@app.get("/api/upload-plan")
def upload_plan(names: str = Query(..., description="올릴 파일 이름들, 줄바꿈으로 구분"),
                bucket: str = Query("3d", pattern="^(2d|3d)$")):
    """올리기 전에 각 파일이 어디로 갈지 미리 알려준다 (화면이 그대로 보여준다)."""
    ns = [n.strip() for n in names.split("\n") if n.strip()]
    out = []
    for n in ns:
        ext = os.path.splitext(n)[1].lower()
        ok = ext in UPLOAD_EXT and not (bucket == "2d" and ext in MODEL_EXT)
        why = None if ok else ("사진 칸에는 3D 파일을 넣을 수 없습니다" if ext in MODEL_EXT else "지원하지 않는 형식")
        dest = (_dest_2d if bucket == "2d" else _dest_dir)(ext, n.lower()) if ok else None
        out.append({"name": n, "ok": ok, "dest": dest, "why": why})
    return {"bucket": bucket, "files": out}


@app.post("/api/artifacts")
async def create_artifact(files_3d: List[UploadFile] = File(default=[]),
                          files_2d: List[UploadFile] = File(default=[]),
                          name_ko: str = Form(...),
                          description_ko: Optional[str] = Form(None),
                          accession: Optional[str] = Form(None),
                          period: Optional[str] = Form(None),
                          material: Optional[str] = Form(None),
                          museum: Optional[str] = Form(None),
                          designation: Optional[str] = Form(None),
                          size: Optional[str] = Form(None),
                          owner: Optional[str] = Form(None),
                          note: Optional[str] = Form(None)):
    name_ko = (name_ko or "").strip()
    if not name_ko:
        raise HTTPException(400, "유물명을 적어 주세요")
    files_3d = [f for f in (files_3d or []) if f.filename]
    files_2d = [f for f in (files_2d or []) if f.filename]
    if not (files_3d or files_2d):
        raise HTTPException(400, "파일을 한 개 이상 올려 주세요")
    bad = [f.filename for f in files_3d + files_2d if os.path.splitext(f.filename)[1].lower() not in UPLOAD_EXT]
    if bad:
        raise HTTPException(400, "허용되지 않는 확장자: %s (허용 %s)" % (bad[:5], sorted(UPLOAD_EXT)))
    models_in_2d = [f.filename for f in files_2d if os.path.splitext(f.filename)[1].lower() in MODEL_EXT]
    if models_in_2d:
        raise HTTPException(400, "사진 칸에 3D 파일이 들어 있습니다. 3D 칸에 넣어 주세요: %s" % models_in_2d[:3])

    # 폴더 이름 = 소장품번호를 정리한 것. 없으면 날짜로 만든다
    acc = (accession or "").strip()
    if acc:
        artifact_id = NEW_ID_BAD.sub("", acc).lower()
        if not artifact_id:
            raise HTTPException(400, "소장품번호에 쓸 수 있는 글자가 없습니다 (영문·숫자·_-)")
    else:
        # 지운 번호를 다시 쓰지 않는다. 같은 ID 를 재사용하면 파일 경로가 똑같아져서
        # 브라우저가 지운 자료의 옛 사진을 캐시에서 꺼내 보여준다(실제로 그 일이 있었다).
        base = "new" + time.strftime("%Y%m%d")
        used = set()
        for g in ("originals", "restored", TRASH_DIR_NAME):
            d = os.path.join(FILES_DIR, g)
            if os.path.isdir(d):
                used |= {(x.split("_", 1)[1] if g == TRASH_DIR_NAME and "_" in x else x) for x in os.listdir(d)}
        used |= {r["artifact_id"] for r in db.rows("SELECT artifact_id FROM artifacts")}
        n = 1
        while "%s_%02d" % (base, n) in used:
            n += 1
        artifact_id = "%s_%02d" % (base, n)
        acc = artifact_id
    dst_dir = os.path.join(FILES_DIR, "originals", artifact_id)
    if os.path.isdir(dst_dir) and os.listdir(dst_dir):
        raise HTTPException(409, "이미 있는 유물입니다: originals/%s" % artifact_id)

    has_model = any(os.path.splitext(f.filename)[1].lower() in MODEL_EXT for f in files_3d)
    tmp_root = dst_dir + ".uploading"
    shutil.rmtree(tmp_root, ignore_errors=True)
    os.makedirs(tmp_root, exist_ok=True)
    try:
        # 칸마다 따로 저장한다. 3D 칸의 이미지는 텍스처, 2D 칸의 이미지는 유물 사진이다.
        w3, fixed, t3, missing = await _store_upload(files_3d, tmp_root, acc, "", _dest_dir)
        w2, _, t2, _ = await _store_upload(files_2d, tmp_root, acc, "", _dest_2d)
        written, total = w3 + w2, t3 + t2

        desc = (description_ko or "").strip()
        auto = not desc
        if auto:                                  # 비우면 적어 준 값으로 한 줄 초안 — 나중에 메모 탭에서 고친다
            bits = [x for x in (period, material, designation) if x and x.strip()]
            desc = "%s. %s" % (name_ko, " · ".join(b.strip() for b in bits)) if bits else name_ko
            desc += " (팀이 등록한 자료. 설명은 아직 채우지 않았다.)"
        meta = {
            "artifact_id": artifact_id, "accession": acc, "accession_ko": accession or None,
            "kind": "original", "kind_ko": "원본 (팀 등록)",
            "name_ko": name_ko, "name_ko_source": "derived",
            "description_ko": desc, "description_source": "derived_team" if auto else "derived",
            "designation": (designation or None), "period": (period or None), "material": (material or None),
            "size": (size or None), "source_org": (museum or "팀 등록"),
            "license_note": "팀이 등록한 자료. 출처·이용조건은 등록자가 확인할 것.",
            "license_status": "unknown", "counterpart": None,
            "registered_by": (owner or None), "note": (note or None),
            "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),   # UTC — 등록순 정렬의 기준
            "source_files": [{"uploaded": o, "stored": "%s/%s" % (s, n), "bytes": b} for s, n, o, b in written],
        }
        json.dump(meta, open(os.path.join(tmp_root, "_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        safe_name = re.sub(r'[\\/:*?"<>|]', "", name_ko)[:60] or artifact_id
        with open(os.path.join(tmp_root, "_%s.md" % safe_name), "w", encoding="utf-8") as f:
            f.write("# %s\n\n- 등록: %s%s\n- 파일 %d개 (%.1f MB)\n\n%s\n"
                    % (name_ko, meta["uploaded_at"], (" · " + owner) if owner else "", len(written), total / 1e6, desc))

        if os.path.isdir(dst_dir):          # 409 를 통과했으니 비어 있는 폴더뿐이다
            shutil.rmtree(dst_dir)
        os.replace(tmp_root, dst_dir)       # 통째로 원자적 교체 — 반쯤 만들어진 폴더가 스캔되지 않게
    except HTTPException:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise HTTPException(500, "등록 실패: %s" % str(e)[:300])

    scan = rescan()
    made = sorted({s for s, n, o, b in written})
    direct = None
    if has_model:
        # GLB 를 올렸으면 변환을 기다리지 않고 바로 돌려 볼 수 있게 한다
        direct = _publish_glb_direct(artifact_id, "source", dst_dir)
        _start_preview(artifact_id, "source")
    return {"ok": True, "artifact_id": artifact_id, "name_ko": name_ko,
            "files_3d": len(w3), "files_2d": len(w2), "files": len(written),
            "bytes": total, "folders": made, "reference_fixed": fixed, "missing_refs": missing,
            "asset_3d": ("%s/3d/source" % artifact_id) if has_model else None,
            "asset_2d": ("%s/2d/source" % artifact_id) if "photo_2d" in made else None,
            "description_auto": auto, "scan": scan.get("scan") if isinstance(scan, dict) else None,
            "warning": ("같이 올렸어야 할 파일이 빠졌습니다: %s — 모델이 회색으로 보입니다"
                        % ", ".join(sorted({m["ref"] for m in missing}))) if missing else None,
            "preview_url": _vurl(direct) if direct else None,
            "preview": (("올린 GLB 로 바로 볼 수 있습니다. 경량 판본과 썸네일은 곧 만들어집니다"
                         if direct else "3D 미리보기를 만드는 중입니다 (보통 10~30초)") if has_model else None)}


# ── 삭제 ─────────────────────────────────────────────────────────────────────
# 지우지 않고 files/_trash/ 로 옮긴다. 인증이 없는 서버(AS-04)라 진짜로 지워 버리면
# 실수 한 번에 자료가 사라진다. scan.py 는 originals·restored·aihub 만 훑으므로
# 휴지통으로 옮기는 것만으로 목록에서는 사라진다. 되살리려면 폴더를 도로 옮기고 재스캔.


@app.delete("/api/artifacts/{artifact_id}")
def delete_artifact(artifact_id: str,
                    confirm: str = Query(..., description="실수 방지 — artifact_id 를 그대로 다시 적는다"),
                    what: str = Query("all", pattern="^(all|restored)$"),
                    variant: Optional[str] = Query(None, description="복원본이 여러 개일 때 그중 하나만 (예: restored__lascomp)")):
    if "/" in artifact_id or "\\" in artifact_id or artifact_id.startswith("."):
        raise HTTPException(400, "invalid artifact_id")
    if confirm != artifact_id:
        raise HTTPException(400, "confirm 이 유물 ID 와 다릅니다")
    if not db.one("SELECT 1 FROM artifacts WHERE artifact_id=?", artifact_id):
        raise HTTPException(404, "그런 유물이 없습니다: %s" % artifact_id)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    trash = os.path.join(FILES_DIR, TRASH_DIR_NAME, "%s_%s" % (stamp, artifact_id))
    os.makedirs(trash, exist_ok=True)
    moved = []
    groups = ["restored"] if what == "restored" else ["originals", "restored"]
    for g in groups:
        if g == "restored":
            # 복원본은 한 원본에 여러 개일 수 있다 — <유물ID> 와 <유물ID>__* 를 모두 본다.
            # variant 를 주면 그 하나만 지운다.
            want = None
            if variant:
                want = artifact_id + ("__" + variant.split("__", 1)[1] if "__" in variant else "")
            gdir = os.path.join(FILES_DIR, g)
            if not os.path.isdir(gdir):
                continue
            for nm in sorted(os.listdir(gdir)):
                if nm != artifact_id and not nm.startswith(artifact_id + "__"):
                    continue
                if want and nm != want:
                    continue
                src = os.path.join(gdir, nm)
                if os.path.isdir(src):
                    shutil.move(src, os.path.join(trash, g + "_" + nm))
                    moved.append("%s/%s" % (g, nm))
            continue
        src = os.path.join(FILES_DIR, g, artifact_id)
        if os.path.isdir(src):
            shutil.move(src, os.path.join(trash, g))
            moved.append("%s/%s" % (g, artifact_id))
    if what == "all":                       # 미리보기도 같이 (다시 등록할 때 옛 썸네일이 붙지 않게)
        pv = os.path.join(PREVIEW_DIR, artifact_id)
        if os.path.isdir(pv):
            shutil.move(pv, os.path.join(trash, "preview"))
            moved.append("preview/%s" % artifact_id)
    else:                                   # 복원본만 지울 때는 그 변형(들)의 프리뷰만
        pvd = os.path.join(PREVIEW_DIR, artifact_id)
        if os.path.isdir(pvd):
            for nm in sorted(os.listdir(pvd)):
                stem, ext = os.path.splitext(nm)
                if ext not in (".glb", ".png") or not stem.startswith("restored"):
                    continue
                if variant and stem != variant:
                    continue
                os.makedirs(os.path.join(trash, "preview"), exist_ok=True)
                shutil.move(os.path.join(pvd, nm), os.path.join(trash, "preview", nm))
    if not moved:
        shutil.rmtree(trash, ignore_errors=True)
        raise HTTPException(404, "옮길 폴더가 없습니다")

    # 원본만 남기고 복원본을 지웠으면 원본의 counterpart 도 지운다
    if what == "restored":
        mp = os.path.join(FILES_DIR, "originals", artifact_id, "_meta.json")
        if os.path.exists(mp):
            try:
                om = json.load(open(mp, encoding="utf-8"))
                if om.pop("counterpart", None) is not None:
                    json.dump(om, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            except Exception:
                pass
    scan = rescan()
    # 지운 자료의 즐겨찾기 행은 남는다 (favorites 에 FK 가 없다 — 재스캔이 CASCADE 로 쓸어가지 않게).
    # 재스캔이 끝난 뒤, 카드가 사라진 즐겨찾기만 정리한다.
    db.write("DELETE FROM favorites WHERE asset_id NOT IN (SELECT asset_id FROM assets)")
    return {"ok": True, "artifact_id": artifact_id, "what": what, "moved": moved,
            "trash": "files/%s/%s_%s" % (TRASH_DIR_NAME, stamp, artifact_id),
            "note": "지운 것이 아니라 휴지통으로 옮겼습니다. 서버에서 폴더를 도로 옮기고 재스캔하면 되살아납니다.",
            "scan": scan.get("scan") if isinstance(scan, dict) else None}


async def _store_upload(files, tmp_root, acc, tag, dest_of):
    """올린 파일을 규칙대로 배치·개명하고, OBJ 의 mtllib·MTL 의 텍스처 경로를 새 이름으로 고친다.
    tag 가 있으면 <acc>_<tag>_<역할>.<ext>, 없으면 <acc>_<역할>.<ext> (기존 원본 자료와 같은 꼴)."""
    rename = {}          # 올린 원래 파일명 → 새 파일명 (MTL·OBJ 참조 재작성용)
    written, total, seen_roles = [], 0, {}
    for up in files:
        orig = os.path.basename(up.filename or "file")
        ext = os.path.splitext(orig)[1].lower()
        low = orig.lower()
        sub = dest_of(ext, low)
        if sub == "photo_2d":
            # 사진은 텍스처가 아니다. 박물관 자료와 같은 꼴(<acc>_photo_01)로 번호를 매긴다
            seen_roles["photo"] = n_photo = seen_roles.get("photo", 0) + 1
            role = "photo_%02d" % n_photo
        else:
            role = _role_of(ext, low, seen_roles)
        newname = "%s_%s%s%s" % (acc, (tag + "_") if tag else "", role, ext)
        os.makedirs(os.path.join(tmp_root, sub), exist_ok=True)
        path = os.path.join(tmp_root, sub, newname)
        with open(path, "wb") as out:
            while True:
                chunk = await up.read(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
                if total > UPLOAD_MAX_TOTAL:
                    raise HTTPException(413, "전체 용량이 %d GB 를 넘습니다" % (UPLOAD_MAX_TOTAL // 1024 ** 3))
                out.write(chunk)
        rename[orig] = newname
        written.append((sub, newname, orig, os.path.getsize(path)))

    # 이름이 바뀐 뒤에도 모델이 재질·텍스처를 찾도록 참조를 다시 쓴다
    fixed, missing = [], []          # missing: 올리지 않아 못 찾은 참조 (회색 모델의 원인)
    for sub, newname, orig, _ in written:
        if os.path.splitext(newname)[1].lower() not in TEXT_EXT:
            continue
        p = os.path.join(tmp_root, sub, newname)
        txt = open(p, encoding="utf-8", errors="replace").read()
        out_lines, changed = [], False
        for line in txt.splitlines():
            m = re.match(r"^(\s*)(mtllib|map_\w+|bump|disp|decal|norm|refl)(\s+)(.*)$", line, re.I)
            if m:
                ref = os.path.basename(m.group(4).strip().replace("\\", "/"))
                tgt = rename.get(ref) or rename.get(os.path.basename(ref))
                if not tgt:      # 대소문자 무시하고 한 번 더
                    for k, v in rename.items():
                        if k.lower() == ref.lower():
                            tgt = v
                            break
                if tgt:
                    line = "%s%s%s%s" % (m.group(1), m.group(2), m.group(3), tgt)
                    changed = True
                elif ref and not os.path.exists(os.path.join(tmp_root, sub, ref)):
                    missing.append({"in": newname, "ref": ref})   # 같이 올렸어야 할 파일이 빠졌다
            out_lines.append(line)
        if changed:
            open(p, "w", encoding="utf-8").write("\n".join(out_lines) + "\n")
            fixed.append(newname)
    return written, fixed, total, missing


@app.post("/api/artifacts/{artifact_id}/restored")
async def upload_restored(artifact_id: str,
                          files: List[UploadFile] = File(...),
                          method: Optional[str] = Form(None),
                          owner: Optional[str] = Form(None),
                          note: Optional[str] = Form(None),
                          label: Optional[str] = Form(None),
                          overwrite: bool = Form(False)):
    """한 원본에 복원본을 **여러 개** 올릴 수 있다 (2026-09-18).
    label 을 주면 `restored/<유물ID>__<슬러그>` 로 따로 쌓이고 variant 가 `restored__<슬러그>` 가 된다.
    label 이 없으면 기존과 같은 한 칸(`restored`)이라 덮어쓰기 규칙도 그대로다."""
    # 경로 조작 차단 — artifact_id 는 폴더 한 칸이어야 한다
    if "/" in artifact_id or "\\" in artifact_id or artifact_id.startswith("."):
        raise HTTPException(400, "invalid artifact_id")
    src_dir = os.path.join(FILES_DIR, "originals", artifact_id)
    if not os.path.isdir(src_dir):
        raise HTTPException(404, "원본 유물 폴더가 없습니다: originals/%s" % artifact_id)
    slug = restored_slug(label)
    folder = artifact_id + ("__" + slug if slug else "")
    variant = "restored" + ("__" + slug if slug else "")
    dst_dir = os.path.join(FILES_DIR, "restored", folder)
    if os.path.isdir(dst_dir) and os.listdir(dst_dir) and not overwrite:
        raise HTTPException(409, "이미 같은 이름의 복원본이 있습니다(%s). 다른 이름을 적거나 overwrite=true" % (slug or "이름 없음"))

    # 확장자·개수 검사 (읽기 전)
    bad = [f.filename for f in files if os.path.splitext(f.filename or "")[1].lower() not in UPLOAD_EXT]
    if bad:
        raise HTTPException(400, "허용되지 않는 확장자: %s (허용 %s)" % (bad[:5], sorted(UPLOAD_EXT)))
    if not files:
        raise HTTPException(400, "파일이 없습니다")

    meta = {}
    mp = os.path.join(src_dir, "_meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp, encoding="utf-8"))
    acc = meta.get("accession") or artifact_id.split("_", 1)[0]

    tmp_root = dst_dir + ".uploading"
    shutil.rmtree(tmp_root, ignore_errors=True)
    os.makedirs(tmp_root, exist_ok=True)
    try:
        written, fixed, total, missing = await _store_upload(files, tmp_root, acc, "restored", _dest_dir)

        # _meta.json — 원본에서 상속하고 복원본 표시만 바꾼다
        rmeta = dict(meta)
        rmeta.update({
            "artifact_id": artifact_id, "accession": acc,
            "kind": "restored", "kind_ko": "복원본 (팀 AI 복원 산출물)",
            "restoration_label": (label or "").strip() or None, "restoration_slug": slug or None,
            "counterpart": "originals/" + artifact_id,
            "restoration_note": note or meta.get("restoration_note"),
            "restoration_method": method, "restoration_owner": owner,
            "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),   # UTC — 등록순 정렬의 기준
            "source_files": [{"uploaded": o, "stored": "%s/%s" % (s, n), "bytes": b} for s, n, o, b in written],
        })
        rmeta.pop("photo_2d_count", None); rmeta.pop("asset_kinds", None); rmeta.pop("file_count", None)
        if meta.get("license_note"):
            rmeta["license_note"] = meta["license_note"] + " / 복원 결과는 팀 산출물이며 AI 추정이 포함된 복원 가설이다."
        json.dump(rmeta, open(os.path.join(tmp_root, "_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

        # 탐색기에서 바로 보이는 한글명 표지 (원본과 같은 방식)
        nk = re.sub(r"\s+", " ", (meta.get("name_ko") or artifact_id)).strip()
        safe = re.sub(r'[\\/:*?"<>|\r\n\t]', " ", nk)
        safe = re.sub(r"\s+", " ", safe).strip()[:80] or artifact_id
        open(os.path.join(tmp_root, "_%s.md" % safe), "w", encoding="utf-8").write(
            "# %s — 복원본\n\n원본: `originals/%s`\n\n올린 시각: %s\n" % (nk, artifact_id, rmeta["uploaded_at"]))

        # 전부 성공했을 때만 제자리로 (반쯤 올라간 폴더가 보이지 않게)
        if os.path.isdir(dst_dir):
            shutil.rmtree(dst_dir + ".old", ignore_errors=True)
            os.replace(dst_dir, dst_dir + ".old")
        os.replace(tmp_root, dst_dir)
        shutil.rmtree(dst_dir + ".old", ignore_errors=True)
    except HTTPException:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise HTTPException(500, "업로드 실패: %s" % str(e)[:300])

    # 원본 쪽에도 짝을 기록해 둔다
    if os.path.exists(mp):
        try:
            om = json.load(open(mp, encoding="utf-8"))
            om["counterpart"] = "restored/" + artifact_id          # 기존 키 — 호환 유지
            cps = om.get("counterparts") or []
            cp = "restored/" + folder
            if cp not in cps:
                cps.append(cp)
            om["counterparts"] = cps
            json.dump(om, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        except Exception:
            pass

    r = rescan()          # DB 에 바로 반영 — 화면에서 새로고침하면 복원본 카드가 보인다
    asset_id = "%s/3d/%s" % (artifact_id, variant)
    # 3D 미리보기는 오래 걸리니 백그라운드로. 화면은 /api/preview-status 로 완료를 기다린다
    direct = _publish_glb_direct(artifact_id, variant, dst_dir)
    _start_preview(artifact_id, variant)
    return {"ok": True, "artifact_id": artifact_id, "files": len(written),
            "bytes": total, "reference_fixed": fixed,
            "stored": ["%s/%s" % (s, n) for s, n, _, _ in written],
            "asset_id": asset_id,
            # 적은 이름이 어느 칸으로 갔는지 알려 준다 — 한글 라벨은 슬러그가 해시가 되므로
            # 이게 없으면 "내가 적은 이름이 어디 갔지" 가 된다. 화면 이름은 label 그대로다.
            "label": (label or "").strip() or None, "slug": slug or None, "variant": variant,
            "scan": r.get("scan") if isinstance(r, dict) else None,
            "missing_refs": missing,
            "warning": ("같이 올렸어야 할 파일이 빠졌습니다: %s — 모델이 회색으로 보입니다"
                        % ", ".join(sorted({m["ref"] for m in missing}))) if missing else None,
            "preview_url": _vurl(direct) if direct else None,
            "preview": ("올린 GLB 로 바로 볼 수 있습니다. 경량 판본과 썸네일은 곧 만들어집니다"
                        if direct else "3D 미리보기를 만드는 중입니다 (보통 10~30초)")}


@app.post("/api/preview/rebuild")
def preview_rebuild(asset: str, up: str = None, flip: str = None, yaw: float = None):
    """미리보기를 다시 만든다 (실패했거나 파일을 바꿨을 때).

    up=X|Y|Z, flip=0|1 을 주면 세우는 방향을 고정한다 (거꾸로 선 것 바로잡기).
    yaw=<도> 는 세로축 둘레로 돌린다 — 바로 섰지만 옆을 보고 있을 때 (yaw=0 이면 지운다).
    up=auto / flip=auto 는 그 항목만 자동 추정으로 되돌린다.
    무엇을 고정했는지는 /api/preview-status 의 up·flip·override 로 확인한다."""
    c = db.one("SELECT artifact_id, variant, media_type FROM assets WHERE asset_id=?", asset)
    if not c or c["media_type"] != "3d":
        raise HTTPException(404, "3D 카드가 아닙니다")
    changes = {}
    if up is not None:
        u = up.strip().upper()
        if u not in ("X", "Y", "Z", "AUTO"):
            raise HTTPException(400, "up 은 X·Y·Z·auto 중 하나여야 합니다")
        changes["up"] = None if u == "AUTO" else u
    if flip is not None:
        f = flip.strip().lower()
        if f not in ("0", "1", "auto", "true", "false"):
            raise HTTPException(400, "flip 은 0·1·auto 중 하나여야 합니다")
        changes["flip"] = None if f == "auto" else ("1" if f in ("1", "true") else "0")
    if yaw is not None:
        if not -360.0 <= yaw <= 360.0:
            raise HTTPException(400, "yaw 는 -360~360 도 사이여야 합니다")
        changes["yaw"] = None if yaw == 0 else round(yaw % 360.0, 3)
    if changes:
        _orient_set(asset, changes)
    if not _start_preview(c["artifact_id"], c["variant"]):
        raise HTTPException(409, "이미 생성 중입니다")
    kept = _orient_all().get(asset)
    return {"ok": True, "asset_id": asset, "state": "running", "orientation": kept}


@app.post("/api/rescan")
def rescan():
    """파일 투입 후 호출. scan.py 를 서브프로세스로 돌린다. 동시 실행 방지 락 1개."""
    if not _rescan_lock.acquire(blocking=False):
        raise HTTPException(409, "rescan already running")
    try:
        t0 = time.time()
        env = dict(os.environ, CATALOG_FILES=FILES_DIR, CATALOG_DB=db.DB, CATALOG_PREVIEW=PREVIEW_DIR)
        p = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "scan.py")],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=1800)
        db.reopen()
        if p.returncode != 0:
            return JSONResponse(status_code=500, content={"ok": False, "error": (p.stderr or p.stdout)[-2000:]})
        last = db.one("SELECT * FROM scans ORDER BY id DESC LIMIT 1")
        return {"ok": True, "took_ms": int((time.time() - t0) * 1000), "scan": last, "log": p.stdout.strip()[-500:]}
    finally:
        _rescan_lock.release()
