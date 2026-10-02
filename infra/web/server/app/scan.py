"""staging/ (서버에서는 /srv/catalog/files/) 트리를 훑어 catalog.db 를 채운다.

트리 → 표 대응
  originals/<A>/{digital_obj,scan_ply,print_stl}  → assets <A>/3d/source
  originals/<A>/photo_2d/                          → assets <A>/2d/source
  restored/<A>/{digital_obj,scan_ply,print_stl,records} → assets <A>/3d/restored
  aihub/<A>/<variant>/pointcloud_las/              → assets <A>/3d/<variant>
  photos_2d/images/<relicId>_01.jpg                → artifacts <relicId> (emuseum) + assets <relicId>/2d/source

한글 정보 규칙 (비워두지 않는다)
  name_ko        : _meta.json 공식명. AIHub "실명 미상" 은 실측 기반 파생명으로 교체 (name_ko_source=derived)
  description_ko : 공식 설명 → 없으면 시대·재질·분류·크기·지정을 문장으로 조합 (description_source=derived)

사용:  python scan.py [--root DIR] [--db FILE] [--hash]
"""
import argparse, hashlib, json, os, re, sqlite3, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DIR_3D = ("digital_obj", "scan_ply", "print_stl", "pointcloud_las")
DIR_2D = ("photo_2d",)
DIR_DOC = ("records",)

VARIANT_KO = {
    "source": "원본", "restored": "복원본 (팀 AI)",
    "pointr_restored": "PoinTr 복원", "symmetry_restored": "회전대칭 복원",
    "pointr_raw_raw": "PoinTr 원출력", "pointr_raw_yup": "PoinTr 원출력 (Y-up)",
    "pointr_raw_output": "PoinTr 원출력", "shared_restore": "복원 (공유본)",
}

# 복원 종류 — "복원본(팀 AI)" 한 덩어리로는 무엇을 복원했는지 모른다는 피드백(2026-09-17).
# 형태/색/2D→3D 를 따로 표시한다. _meta.json 의 restore_types 가 있으면 그대로 쓰고,
# 없으면 variant 와 restoration_method 글에서 짐작한다 (사람이 웹에서 고치면 그 값이 이긴다).
RESTORE_TYPES = ("shape", "color", "img2mesh", "symmetry", "pointr")
# 복원본은 한 원본에 여러 개 올릴 수 있다 (2026-09-18).
#   restored/<유물ID>            → variant "restored"            (기존 — 그대로 둔다)
#   restored/<유물ID>__<슬러그>   → variant "restored__<슬러그>"   (여러 개)
# 슬러그가 asset_id 에 들어가므로 UNIQUE(artifact_id, media_type, variant) 가 그대로 여러 개를 허용한다.
RESTORED_SEP = "__"


def split_restored_dir(name):
    """폴더 이름을 (유물 ID, 슬러그 또는 None) 으로. 슬러그가 없으면 기존과 같다."""
    if RESTORED_SEP in name:
        base, slug = name.split(RESTORED_SEP, 1)
        slug = slug.strip()
        if base and slug:
            return base, slug
    return name, None


def restored_variant(slug):
    return "restored" if not slug else "restored" + RESTORED_SEP + slug


def variant_ko_of(variant, meta):
    """화면에 보일 이름. 여러 개일 때는 올릴 때 적은 라벨을, 없으면 슬러그를 쓴다."""
    if variant in VARIANT_KO:
        return VARIANT_KO[variant]
    if variant.startswith("restored" + RESTORED_SEP):
        slug = variant.split(RESTORED_SEP, 1)[1]
        lab = (meta or {}).get("restoration_label") or slug
        return "복원본 · %s" % lab
    return variant


VARIANT_TYPES = {"symmetry_restored": ("shape", "symmetry"),
                 "pointr_restored": ("shape", "pointr"),
                 "pointr_raw_raw": ("shape", "pointr"), "pointr_raw_yup": ("shape", "pointr"),
                 "pointr_raw_output": ("shape", "pointr")}
# 복원 방식 글에서 찾는 말. 왼쪽이 코드, 오른쪽이 그 코드로 볼 단서.
METHOD_HINTS = [
    ("color",    ("색", "컬러", "color", "cielab", "lab", "채색", "질감", "texture")),
    ("img2mesh", ("2d", "이미지", "사진", "image", "img2", "to 3d", "→3d", "->3d", "트리포", "tripo", "trellis", "las-comp", "lascomp")),
    ("symmetry", ("회전대칭", "대칭", "symmetry", "rotational")),
    ("pointr",   ("pointr", "포인터", "점군 완성", "point cloud completion")),
    ("shape",    ("형태", "형상", "결손", "채움", "shape", "geometry", "mesh", "완성", "complete")),
]


def restore_types_of(variant, meta):
    """복원 종류 코드들을 쉼표 문자열로. 원본은 빈 값."""
    if variant == "source":
        return None
    given = meta.get("restore_types") or meta.get("restoration_types")
    if given:
        got = [t.strip() for t in (given.split(",") if isinstance(given, str) else given)]
        keep = [t for t in got if t in RESTORE_TYPES]
        if keep:
            return ",".join(dict.fromkeys(keep))
    out = list(VARIANT_TYPES.get(variant, ()))
    text = " ".join(str(meta.get(k) or "") for k in
                    ("restoration_method", "method", "restoration_note", "note", "kind_ko")).lower()
    for code, words in METHOD_HINTS:
        if any(w in text for w in words):
            out.append(code)
    if not out:
        out = ["shape"]          # 무엇을 했는지 단서가 없으면 형태 복원으로 본다 (지금까지의 복원본은 전부 형태였다)
    return ",".join(dict.fromkeys(out))
KIND_KO = {
    "obj": "모델 (OBJ)", "glb": "모델 (GLB)", "mtl": "재질 (MTL)", "texture": "텍스처", "normalmap": "노멀맵",
    "ply": "점군·메시 (PLY)", "stl": "프린트용 (STL)", "las": "점군 (LAS)",
    "report": "복원 리포트", "image": "이미지", "photo": "사진", "other": "기타",
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# AIHub 는 공식 ID-문화재명 대응표가 없다. 점군을 실측해 파생명·설명을 만든다 (2026-09-07).
AIHUB_DERIVED = {
    "RR_07_01_EA_028": {
        "name_ko": "원형 토기편 (경상도 출토, AIHub EA_028)",
        "category": "도·토기류",
        "material": "토제 (점군 색상 기준 추정)",
        "size": "약 64.6 × 62.4 × 35.2 cm (점군 실측)",
        "description_ko": (
            "AIHub 문화유산 3D 데이터의 경상도 도·토기류 점군. 회전 대칭인 원형 토기의 일부로, "
            "위쪽 중앙에 규칙적인 원형 입구가 있고 아래쪽은 불규칙한 파단면으로 끝난다. "
            "측면에 큰 결손이 있어 팀의 회전대칭·PoinTr 복원 실험 대상으로 썼다. "
            "점 179,294개, 실측 크기 약 64.6 × 62.4 × 35.2 cm, 평균 색조는 어두운 회갈색이다. "
            "공식 문화재명은 제공되지 않아 형태로 이름을 붙였다."),
    },
    "RR_07_01_PA_033": {
        "name_ko": "석조 탑·비 부재 (경상도, AIHub PA_033)",
        "category": "탑·비",
        "material": "석재 (점군 색상 기준 추정)",
        "size": "약 82.6 × 220.7 × 89.4 cm (점군 실측)",
        "description_ko": (
            "AIHub 문화유산 3D 데이터의 경상도 탑·비류 점군. 한 방향으로 약 2.2 m 길게 뻗은 석조 부재로, "
            "밝은 회갈색의 석재 표면을 보인다. 점 1,488,637개, 실측 크기 약 82.6 × 220.7 × 89.4 cm. "
            "팀에서 PoinTr 복원과 회전대칭 복원을 각각 시험했다. "
            "공식 문화재명은 제공되지 않아 형태로 이름을 붙였다."),
    },
}


# 박물관 목록명이 한자만인 6건. 카탈로그의 다른명칭·목록명, 없으면 한자 독음으로 한글명을 준다.
# 원래 한자 표기는 alt_name 에 보존한다. (2026-09-07 확인)
NAME_KO_FIX = {
    "bon004164": "점토 인두 (진묘수 머리)",            # 粘土製人頭 · 전시명 '진묘수 머리'
    "bon012484": "청동 은입사 물가풍경무늬 정병",       # 靑銀製銀象嵌蒲柳水禽文甁 · 다른명칭
    "jub002084": "백자 달항아리",                     # 白磁大缸 · 목록명
    "kno000038": "주구토기",                          # 注口杯 · 목록명
    "PS0100100200100033100000": "금동 투조 장신구 (사리장엄구)",  # 金銅透彫裝身具 · 분류: 사리구
    "PS0100100200800081000000": "금동제 장신구 조각",            # 金銅製裝身具片
}
HANGUL = re.compile(r"[가-힣]")

# 공식 설명이 없는 유물의 한글 설명 (사람이 쓴 것). 키는 소장품번호 또는 e뮤지엄 relicId.
_dp = os.path.join(HERE, "descriptions_ko.json")
DESC_KO = {k: v for k, v in json.load(open(_dp, encoding="utf-8")).items() if not k.startswith("_")} if os.path.exists(_dp) else {}
# e뮤지엄에서 같은 소장품번호로 찾은 보조 정보 (다른명칭·출토지). work/fill_museum_desc.py 산출물.
_ep = os.path.join(HERE, "museum_extra_from_emuseum.json")
MUSEUM_EXTRA = json.load(open(_ep, encoding="utf-8")) if os.path.exists(_ep) else {}


def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def leaf(cat):
    """'종교신앙 - 불교 - 예배 - 불상' → '불상'"""
    return norm(cat).split(" - ")[-1] if cat else ""


def derive_description(m):
    """설명이 없을 때 구조화 필드를 문장으로 조합한다. 있는 것만 쓴다."""
    parts = []
    nm = norm(m.get("name_ko")) or norm(m.get("artifact_id"))
    seg = []
    if m.get("period"):
        seg.append(norm(m["period"]).replace("한국 - ", "") + " 시대의" if "한국" in m["period"] else norm(m["period"]))
    obj = leaf(m.get("category")) or "유물"
    parts.append("%s. %s %s이다." % (nm, " ".join(seg), obj) if seg else "%s. %s이다." % (nm, obj))
    if m.get("material"):
        parts.append("재질은 %s." % norm(m["material"]))       # '흙 - 경질' 처럼 계층 표기를 그대로 둔다
    if m.get("size"):
        parts.append("크기는 %s." % norm(m["size"]).rstrip("."))
    if m.get("designation"):
        d = norm(m["designation"])
        d = re.sub(r"National Treasure\s*(\d*)", lambda x: "국보 " + x.group(1), d)
        d = re.sub(r"^Treasure\s*(\d*)", lambda x: "보물 " + x.group(1), d)
        parts.append("%s로 지정되어 있다." % d.strip())
    if m.get("museum") or m.get("source_org"):
        parts.append("%s 소장." % norm(m.get("museum") or m.get("source_org")))
    return " ".join(parts)


def sha256_of(p, block=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def image_size(p):
    try:
        from PIL import Image
        with Image.open(p) as im:
            return im.size
    except Exception:
        return (None, None)


def file_kind(subdir, fname):
    ext = os.path.splitext(fname)[1].lower()
    low = fname.lower()
    if subdir in DIR_2D:
        return "photo"
    if subdir == "digital_obj":
        if ext == ".obj": return "obj"
        if ext == ".glb": return "glb"
        if ext == ".mtl": return "mtl"
        if "_normal" in low or "_nor" in low: return "normalmap"
        if ext in IMG_EXT: return "texture"
    if ext == ".ply": return "ply"
    if ext == ".stl": return "stl"
    if ext == ".las": return "las"
    if subdir in DIR_DOC:
        return "report" if ext in (".json", ".txt", ".md", ".npy") else ("image" if ext in IMG_EXT else "report")
    if ext in IMG_EXT: return "image"
    return "other"


def search_text(*vals):
    return " ".join(norm(str(v)) for v in vals if v).lower()


class Scanner:
    def __init__(self, root, db, do_hash=False):
        self.root, self.do_hash = root, do_hash
        fresh = not os.path.exists(db)
        self.cx = sqlite3.connect(db)
        # 이미 있는 표에 새 열(registered_at)을 붙인다 — v_cards 뷰가 참조하므로 schema.sql 보다 먼저 (db.migrate 와 같은 내용)
        have = {r[1] for r in self.cx.execute("PRAGMA table_info(assets)")}
        for col in ("registered_at", "restore_types"):
            if have and col not in have:
                self.cx.execute("ALTER TABLE assets ADD COLUMN %s TEXT" % col)
        self.cx.executescript(open(os.path.join(HERE, "schema.sql"), encoding="utf-8").read())
        # 사람이 웹에서 고친 열은 재스캔에 지워지지 않게 먼저 떠 둔다 (AS-AT-06)
        self.keep_art = {r[0]: r[1:] for r in self.cx.execute(
            "SELECT artifact_id, tags, note FROM artifacts WHERE tags IS NOT NULL OR note IS NOT NULL")}
        self.keep_ast = {r[0]: r[1:] for r in self.cx.execute(
            "SELECT asset_id, method, owner, produced_at, restore_types FROM assets "
            "WHERE method IS NOT NULL OR owner IS NOT NULL OR produced_at IS NOT NULL OR restore_types IS NOT NULL")}
        self.cx.execute("DELETE FROM files"); self.cx.execute("DELETE FROM assets"); self.cx.execute("DELETE FROM artifacts")
        self.n = {"artifacts": 0, "assets": 0, "files": 0, "bytes": 0}

    # ── 유물 ──
    def put_artifact(self, aid, m, source_org, museum=None):
        m = dict(m)
        derived = AIHUB_DERIVED.get(aid)
        name_src = m.get("name_ko_source") or "official"
        if derived:
            m.update(derived); name_src = "derived"
        name_ko = norm(m.get("name_ko")) or aid
        if not HANGUL.search(name_ko):                       # 한자만인 이름 → 한글명으로, 원표기는 alt_name 에
            fix = NAME_KO_FIX.get(m.get("accession")) or NAME_KO_FIX.get(aid)
            if fix:
                m["alt_name"] = " / ".join(x for x in (name_ko, norm(m.get("alt_name"))) if x)
                name_ko, name_src = fix, "derived"
        extra = MUSEUM_EXTRA.get(m.get("accession") or "", {})
        for k in ("alt_name", "excavation_site"):
            if extra.get(k) and not m.get(k):
                m[k] = extra[k]
        desc = norm(m.get("description_ko") or m.get("description"))
        desc_src = m.get("description_source") or ("derived" if derived else "official")
        if not desc:
            hand = DESC_KO.get(m.get("accession") or "") or DESC_KO.get(aid)
            if hand:
                desc, desc_src = norm(hand["text"]), hand.get("source", "derived")
            else:                                        # 마지막 수단 — 여기 걸리면 descriptions_ko.json 에 써 넣을 것
                m["museum"] = museum
                desc, desc_src = derive_description(m), "derived_template"
        row = dict(
            artifact_id=aid, accession=m.get("accession"), accession_ko=m.get("accession_ko"), slug=m.get("slug"),
            name_ko=name_ko, name_ko_source=name_src, alt_name=m.get("alt_name"),
            name_en=m.get("name_en"), name_en_source=m.get("name_en_source"),
            designation=m.get("designation"), period=m.get("period"), material=m.get("material"),
            category=m.get("category"), size=m.get("size"), excavation_site=m.get("excavation_site"),
            provenance=m.get("provenance"), description_ko=desc, description_source=desc_src,
            description_en=m.get("description_en"), source_org=source_org, museum=museum,
            detail_url=m.get("detail_url"), license_status=m.get("license_status") or "unverified",
            license_note=m.get("license_note"), kogl_type=m.get("kogl_type"), tags=None, note=None,
            search_text=search_text(aid, m.get("accession"), m.get("accession_ko"), name_ko, m.get("alt_name"),
                                    m.get("name_en"), m.get("designation"), m.get("period"), m.get("material"),
                                    m.get("category"), museum, desc[:200]),
        )
        cols = ",".join(row); q = ",".join("?" * len(row))
        self.cx.execute("INSERT OR REPLACE INTO artifacts(%s) VALUES(%s)" % (cols, q), list(row.values()))
        self.n["artifacts"] += 1
        return row

    # ── 카드 + 파일 ──
    def put_asset(self, art, media, variant, file_specs, preview_dir=None, meta=None):
        """file_specs: [(abs_path, rel_path, subdir)] — rel_path 는 files/ 기준 URL 경로
        meta: 그 폴더의 _meta.json. 복원본의 restoration_method/owner 를 카드에 싣는다."""
        if not file_specs:
            return
        aid = "%s/%s/%s" % (art["artifact_id"], media, variant)
        vko = variant_ko_of(variant, meta)
        # 미리보기: 3D 는 preview/<artifact>/<variant>.glb (Phase 2 산출물이 있을 때만)
        glb = png = None
        if media == "3d" and preview_dir:
            g = os.path.join(preview_dir, art["artifact_id"], variant + ".glb")
            p = os.path.join(preview_dir, art["artifact_id"], variant + ".png")
            glb = "preview/%s/%s.glb" % (art["artifact_id"], variant) if os.path.exists(g) else None
            png = "preview/%s/%s.png" % (art["artifact_id"], variant) if os.path.exists(p) else None
        rows, total = [], 0
        photos = [s for s in file_specs if s[2] in DIR_2D]
        for i, (ap, rp, sub) in enumerate(sorted(file_specs, key=lambda s: (s[2], s[1]))):
            fn = os.path.basename(ap); kind = file_kind(sub, fn); ext = os.path.splitext(fn)[1].lower()
            size = os.path.getsize(ap); total += size
            w = h = None
            if ext in IMG_EXT:
                w, h = image_size(ap)
            if kind == "photo":
                idx = [s[1] for s in sorted(photos, key=lambda s: s[1])].index(rp) + 1
                label = "사진 %d/%d" % (idx, len(photos))
                prev = rp; pv = 1
            elif kind in ("obj", "glb", "ply", "stl", "las"):
                label = KIND_KO[kind]; prev = glb; pv = 1 if glb else 0
            elif kind == "image":
                label = "이미지 · " + fn; prev = rp; pv = 1
            else:
                label = KIND_KO.get(kind, kind) + (" · " + fn if kind in ("report", "other") else "")
                prev = None; pv = 0
            rows.append((rp, aid, fn, None, kind, label, ext, size, w, h,
                         sha256_of(ap) if self.do_hash else None,
                         1 if os.path.exists(ap + ".gz") else 0, pv, prev, i,
                         time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(ap)))))
        thumb = (sorted(photos, key=lambda s: s[1])[0][1] if photos else png)
        if media == "3d":
            title = "%s · %s 3D" % (art["name_ko"], vko)
        else:
            title = "%s · 사진 %d장" % (art["name_ko"], len(photos)) if len(photos) != 1 else "%s · 사진" % art["name_ko"]
        sub_ko = " · ".join(x for x in (art.get("designation"), art.get("period"), art.get("material"), art.get("museum")) if x)
        # 복원 방식·만든 사람은 _meta.json 의 restoration_* 에서 가져온다.
        # (사람이 웹에서 고친 값은 keep_ast 로 이 뒤에 다시 덮어쓴다 — 손으로 적은 쪽이 이긴다)
        meta = meta or {}
        method = meta.get("restoration_method") or meta.get("method")
        owner = meta.get("restoration_owner") or meta.get("owner")
        # 등록 시각 — 목록의 기본 정렬(최신 등록순) 기준. UTC, 끝에 Z.
        # 웹에서 등록한 것은 _meta.json 의 uploaded_at(서버가 UTC 로 적음), 박물관 자료처럼 그게 없으면
        # 파일들 중 가장 늦은 수정 시각. 재스캔마다 DB 행은 지우고 다시 넣으므로 DB 의 시각은 못 쓴다.
        up = (meta.get("uploaded_at") or "").strip()
        if up:
            registered = up.replace(" ", "T")
            registered = registered if registered.endswith("Z") or "+" in registered else registered + "Z"
        else:
            registered = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(max(os.path.getmtime(ap) for ap, _, _ in file_specs)))
        rtypes = restore_types_of(variant, meta) if media == "3d" else None
        self.cx.execute(
            "INSERT OR REPLACE INTO assets(asset_id,artifact_id,media_type,variant,variant_ko,title_ko,subtitle_ko,"
            "thumb_path,preview_path,preview_kind,file_count,total_bytes,search_text,method,owner,registered_at,restore_types) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (aid, art["artifact_id"], media, variant, vko, title, sub_ko, thumb,
             glb if media == "3d" else None, "glb" if glb else ("image" if media == "2d" else "none"),
             len(rows), total, search_text(art["search_text"], vko, title, media, rtypes), method, owner, registered, rtypes))
        self.cx.executemany("INSERT OR REPLACE INTO files(path,asset_id,filename,original_name,kind,label_ko,ext,size,"
                            "width,height,sha256,has_gz,previewable,preview_path,sort_order,mtime) "
                            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        self.n["assets"] += 1; self.n["files"] += len(rows); self.n["bytes"] += total

    # ── 트리 순회 ──
    def walk_group(self, group, preview_dir):
        gdir = os.path.join(self.root, group)
        if not os.path.isdir(gdir):
            return
        for name in sorted(os.listdir(gdir)):
            adir = os.path.join(gdir, name)
            if not os.path.isdir(adir):
                continue
            mp = os.path.join(adir, "_meta.json")
            m = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else {"artifact_id": name}
            # 웹에서 등록한 유물은 _meta.json 의 kind 로 구분한다 (박물관 공개 자료와 섞이면 출처가 흐려진다)
            source_org = "aihub" if group == "aihub" else ("team" if m.get("kind_ko", "").startswith("원본 (팀") else "museum")
            museum = m.get("source_org") if source_org != "aihub" else "AIHub 문화유산 3D"
            aid, rslug = split_restored_dir(name) if group == "restored" else (name, None)
            art = self.put_artifact(name, m, source_org, museum) if group != "restored" else None
            if group == "restored":
                # 원본이 이미 넣어졌으면 그 행을 쓰고, 없으면 복원본 메타로 유물 행을 만든다
                cur = self.cx.execute("SELECT * FROM artifacts WHERE artifact_id=?", (aid,)).fetchone()
                if cur:
                    cols = [c[1] for c in self.cx.execute("PRAGMA table_info(artifacts)")]
                    art = dict(zip(cols, cur))
                else:
                    art = self.put_artifact(aid, m, source_org, museum)
            def collect(sub_filter):
                out = []
                for sub in sorted(os.listdir(adir)):
                    sp = os.path.join(adir, sub)
                    if not os.path.isdir(sp) or not sub_filter(sub):
                        continue
                    for r, _, fs in os.walk(sp):
                        for f in fs:
                            if f.endswith(".gz") or f.startswith("_"):
                                continue
                            ap = os.path.join(r, f)
                            out.append((ap, os.path.relpath(ap, self.root).replace("\\", "/"), sub))
                return out
            if group in ("originals", "restored"):
                variant = "source" if group == "originals" else restored_variant(rslug)
                self.put_asset(art, "3d", variant, collect(lambda s: s in DIR_3D or s in DIR_DOC), preview_dir, m)
                self.put_asset(art, "2d", variant, collect(lambda s: s in DIR_2D), preview_dir, m)
            elif group == "aihub":
                for sub in sorted(os.listdir(adir)):
                    sp = os.path.join(adir, sub)
                    if not os.path.isdir(sp):
                        continue
                    specs = []
                    for r, _, fs in os.walk(sp):
                        for f in fs:
                            if not f.endswith(".gz"):
                                ap = os.path.join(r, f)
                                specs.append((ap, os.path.relpath(ap, self.root).replace("\\", "/"),
                                              os.path.basename(r) if os.path.basename(r) != sub else "pointcloud_las"))
                    self.put_asset(art, "3d", sub, specs, preview_dir, m)

    def walk_emuseum(self):
        pdir = os.path.join(self.root, "photos_2d")
        ix = os.path.join(pdir, "index.json")
        if not os.path.exists(ix):
            return
        for r in json.load(open(ix, encoding="utf-8")):
            rid = r.get("relicId") or os.path.splitext(r["file"])[0]
            ap = os.path.join(pdir, "images", r["file"])
            if not os.path.exists(ap):
                continue
            m = dict(r)
            # 상세 페이지 설명 끝에 붙는 '연관단어 #태그 …' 는 설명이 아니므로 뗀다
            desc = re.split(r"\s*·?\s*연관단어\s*", r.get("description") or "")[0].strip() or None
            m.update({"artifact_id": rid, "accession": rid, "name_ko_source": "emuseum_official",
                      "description_ko": desc,
                      "description_source": "emuseum_official" if desc else None,
                      # 공공누리 1유형(출처표시)만 ok. 2~4유형은 상업적 이용·변경 제한이 있어 restricted.
                      "license_status": {1: "ok", 2: "restricted", 3: "restricted", 4: "restricted"}.get(r.get("kogl_type"), "unverified"),
                      "license_note": ("공공누리 %s유형: %s" % (r["kogl_type"], r.get("kogl_label", ""))) if r.get("kogl_type") else "소장처별 이용조건 확인 필요",
                      "source_org": r.get("museum")})
            art = self.put_artifact(rid, m, "emuseum", r.get("museum"))
            rp = os.path.relpath(ap, self.root).replace("\\", "/")
            self.put_asset(art, "2d", "source", [(ap, rp, "photo_2d")])

    def run(self, preview_dir):
        t0 = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())   # UTC — 화면에서 보는 사람 시각으로 바꾼다
        for g in ("originals", "restored", "aihub"):
            self.walk_group(g, preview_dir)
        self.walk_emuseum()
        # 사람이 고친 열 복원
        for aid, (tags, note) in self.keep_art.items():
            self.cx.execute("UPDATE artifacts SET tags=?, note=? WHERE artifact_id=?", (tags, note, aid))
        for aid, (method, owner, produced_at, rtypes) in self.keep_ast.items():
            # 사람이 고른 복원 종류가 있으면 그것이 이긴다 (없으면 방금 추정한 값을 둔다)
            self.cx.execute("UPDATE assets SET method=?, owner=?, produced_at=?, restore_types=COALESCE(?, restore_types) "
                            "WHERE asset_id=?", (method, owner, produced_at, rtypes, aid))
        self.cx.execute("INSERT INTO scans(started_at,finished_at,artifacts,assets,files,total_bytes) VALUES(?,?,?,?,?,?)",
                        (t0, time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), self.n["artifacts"], self.n["assets"], self.n["files"], self.n["bytes"]))
        self.cx.commit()
        return self.n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("CATALOG_FILES", os.path.expanduser("~/c201-assets/staging")))
    ap.add_argument("--db", default=os.environ.get("CATALOG_DB", os.path.expanduser("~/c201-assets/catalog.db")))
    ap.add_argument("--preview", default=os.environ.get("CATALOG_PREVIEW"), help="preview/ 디렉터리 (없으면 root/../preview)")
    ap.add_argument("--hash", action="store_true", help="sha256 계산 (느림)")
    a = ap.parse_args()
    preview = a.preview or os.path.join(os.path.dirname(a.root.rstrip("/\\")), "preview")
    t = time.time()
    n = Scanner(a.root, a.db, a.hash).run(preview)
    print("artifacts %d · assets %d · files %d · %.2f GB · %.1fs → %s"
          % (n["artifacts"], n["assets"], n["files"], n["bytes"] / 1073741824, time.time() - t, a.db))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
