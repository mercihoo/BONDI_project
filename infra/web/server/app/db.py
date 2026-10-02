"""SQLite 연결과 질의 도우미. 인증 없음(AS-04). 쓰기는 화이트리스트 열만."""
import os, sqlite3, threading

DB = os.environ.get("CATALOG_DB", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "catalog.db"))
_lock = threading.Lock()
_cx = None


def connect():
    global _cx
    if _cx is None:
        _cx = sqlite3.connect(DB, check_same_thread=False)
        _cx.row_factory = sqlite3.Row
        _cx.execute("PRAGMA journal_mode=WAL")
        _cx.execute("PRAGMA foreign_keys=ON")
        # 스키마를 앱 시작 때도 한 번 적용한다 — CREATE IF NOT EXISTS / DROP+CREATE VIEW 라 몇 번 돌려도 같다.
        # 새 표(favorites)나 뷰의 새 열이 배포만으로 반영되게 하기 위해 (scan.py 를 다시 안 돌려도).
        try:
            migrate(_cx)
            _cx.executescript(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql"), encoding="utf-8").read())
            _cx.commit()
        except Exception:
            pass
    return _cx


def migrate(cx):
    """CREATE TABLE IF NOT EXISTS 는 이미 있는 표에 새 열을 붙여 주지 않는다.
    새 열을 참조하는 뷰(v_cards)가 schema.sql 에 있으니, 그보다 먼저 여기서 붙인다."""
    have = {r[1] for r in cx.execute("PRAGMA table_info(assets)")}
    for col in ("registered_at", "restore_types"):       # 순서대로 붙는다 — 뒤에 열이 늘면 여기에 추가
        if have and col not in have:
            cx.execute("ALTER TABLE assets ADD COLUMN %s TEXT" % col)
            cx.commit()
            if col == "restore_types":
                _backfill_restore_types(cx)


def _backfill_restore_types(cx):
    """열을 새로 붙이면 기존 행은 비어 있다 — 배포 직후 재스캔하기 전까지 배지가 하나도 안 보인다.
    variant 와 method 글만 보고 한 번 채워 둔다. 재스캔이 _meta.json 까지 보고 더 정확히 덮어쓴다.
    (scan.py 의 restore_types_of 와 같은 규칙. 여기서 import 하지 않는 이유는 scan.py 가 무거운 모듈이라서.)"""
    rows = cx.execute("SELECT asset_id, variant, COALESCE(method,'') FROM assets "
                      "WHERE media_type='3d' AND variant!='source' AND restore_types IS NULL").fetchall()
    hints = [("color", ("색", "컬러", "color", "cielab", "채색", "질감", "texture")),
             ("img2mesh", ("2d", "이미지", "사진", "image", "trellis", "tripo", "lascomp", "las-comp")),
             ("symmetry", ("회전대칭", "대칭", "symmetry")),
             ("pointr", ("pointr", "포인터"))]
    by_variant = {"symmetry_restored": ["shape", "symmetry"], "pointr_restored": ["shape", "pointr"],
                  "pointr_raw_raw": ["shape", "pointr"], "pointr_raw_yup": ["shape", "pointr"],
                  "pointr_raw_output": ["shape", "pointr"]}
    for aid, variant, method in rows:
        out = list(by_variant.get(variant, []))
        low = (method or "").lower()
        for code, words in hints:
            if any(w in low for w in words):
                out.append(code)
        out = list(dict.fromkeys(out)) or ["shape"]
        cx.execute("UPDATE assets SET restore_types=? WHERE asset_id=?", (",".join(out), aid))
    if rows:
        cx.commit()


def rows(sql, *args):
    with _lock:
        return [dict(r) for r in connect().execute(sql, args).fetchall()]


def one(sql, *args):
    r = rows(sql, *args)
    return r[0] if r else None


def write(sql, *args):
    with _lock:
        cx = connect()
        cur = cx.execute(sql, args)
        cx.commit()
        return cur.rowcount


def reopen():
    """scan.py 가 파일을 다시 쓴 뒤 연결을 새로 잡는다."""
    global _cx
    with _lock:
        if _cx is not None:
            _cx.close()
        _cx = None


# 사람이 웹에서 고칠 수 있는 열. 이 밖의 키가 오면 400.
ARTIFACT_EDITABLE = ("tags", "note", "license_status")
ASSET_EDITABLE = ("method", "owner", "produced_at", "restore_types")

# 복원 종류 — 카드 배지·필터에 쓰는 통제 어휘. 피드백(2026-09-17): "복원본(팀 AI)" 한 가지가 아니라
# 2D→3D 변환 / 색 / 형태 처럼 무엇을 복원했는지 보이게.
RESTORE_TYPES = {
    "shape": "형태 복원", "color": "색 복원", "img2mesh": "2D→3D 변환",
    "symmetry": "회전대칭 복원", "pointr": "점군 완성 (PoinTr)",
}

LIKE_COLS = "search_text"
