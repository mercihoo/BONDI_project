-- 문화유산 3D 자료 카탈로그 스키마 (SQLite)
--
-- 유저플로우에 맞춘 3층 구조:
--   ① 2D / 3D 선택            → assets.media_type
--   ② 자료 목록 (한글명)        → assets  (카드 1장 = 1행, title_ko 필수)
--   ③ 검색                    → assets.search_text  LIKE
--   ④ 파일 선택 → 미리보기·다운로드 → files.preview_path / files.path
--
--   artifacts (유물 1점)  1 ─ N  assets (2D 카드·3D 카드)  1 ─ N  files (개별 파일)
--
-- 한글 정보는 비워두지 않는다. name_ko·description_ko 는 NOT NULL 이고,
-- 어디서 왔는지 *_source 열에 남긴다 (official | emuseum_official | derived).

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ───────────────────────────── 유물 ─────────────────────────────
CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id        TEXT PRIMARY KEY,          -- 폴더명. bon002789_pensive-bodhisattva-nt78 / RR_07_01_EA_028 / PS0100304800100897100000
  accession          TEXT,                      -- bon002789
  accession_ko       TEXT,                      -- 본관2789 / 제주 39827
  slug               TEXT,
  name_ko            TEXT NOT NULL,             -- 금동 반가 사유상
  name_ko_source     TEXT NOT NULL,             -- official | emuseum_official | derived
  alt_name           TEXT,                      -- 다른명칭 (한자 등)
  name_en            TEXT,
  name_en_source     TEXT,
  designation        TEXT,                      -- 국보/보물 표기
  period             TEXT,                      -- 한국 - 삼국
  material           TEXT,                      -- 금속 - 금동
  category           TEXT,                      -- 종교신앙 - 불교 - 예배 - 불상
  size               TEXT,                      -- 높이 81.5cm ...
  excavation_site    TEXT,                      -- 출토지
  provenance         TEXT,
  description_ko     TEXT NOT NULL,             -- 유물이 무엇인지 한글 설명
  description_source TEXT NOT NULL,             -- official | emuseum_official | derived
  description_en     TEXT,
  source_org         TEXT NOT NULL,             -- museum | aihub | emuseum
  museum             TEXT,                      -- 소장처 표시명 (국립중앙박물관 / 대가야박물관 ...)
  detail_url         TEXT,
  license_status     TEXT NOT NULL DEFAULT 'unverified',  -- ok | unverified | restricted
  license_note       TEXT,
  kogl_type          INTEGER,                   -- 공공누리 1~4 (e뮤지엄에서 확인된 경우)
  tags               TEXT,                      -- 사람이 채움, 쉼표 구분
  note               TEXT,                      -- 사람이 채움
  search_text        TEXT NOT NULL,             -- 검색용 결합 문자열 (소문자)
  created_at         TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ───────────────────────────── 카드 ─────────────────────────────
-- 화면의 목록 1줄. 같은 유물이라도 2D 카드와 3D 카드는 별개 행이고,
-- 3D 는 원본/복원본/실험출력 마다 카드가 따로 있다.
CREATE TABLE IF NOT EXISTS assets (
  asset_id      TEXT PRIMARY KEY,               -- <artifact_id>/<media_type>/<variant>
  artifact_id   TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  media_type    TEXT NOT NULL CHECK (media_type IN ('2d','3d')),
  variant       TEXT NOT NULL,                  -- source | restored | pointr_restored | symmetry_restored | ...
  variant_ko    TEXT NOT NULL,                  -- 원본 | 복원본 | PoinTr 복원 | 회전대칭 복원 ...
  title_ko      TEXT NOT NULL,                  -- 목록에 보이는 한 줄. "금동 반가 사유상 · 원본 3D"
  subtitle_ko   TEXT,                           -- "국보78 · 한국 - 삼국 · 금속 - 금동"
  method        TEXT,                           -- 복원 방식 (사람이 보정)
  owner         TEXT,                           -- 만든 사람 (사람이 채움)
  restore_types TEXT,                           -- 복원 종류 코드, 쉼표 구분: shape 형태 | color 색 | img2mesh 2D→3D | symmetry 회전대칭 | pointr 점군 완성.
                                                --   _meta.json restore_types 또는 method 글에서 추정. 사람이 고친 값은 재스캔에도 남는다
  produced_at   TEXT,
  registered_at TEXT,                            -- 등록 시각(UTC, 끝에 Z). 팀 등록은 _meta.json uploaded_at, 그 외는 파일 mtime. 목록 기본 정렬
  thumb_path    TEXT,                           -- 목록 썸네일 URL 경로
  preview_path  TEXT,                           -- 3D: preview/<artifact>/<variant>.glb  (Phase 2 생성)
  preview_kind  TEXT NOT NULL DEFAULT 'none',   -- glb | image | none
  file_count    INTEGER NOT NULL DEFAULT 0,
  total_bytes   INTEGER NOT NULL DEFAULT 0,
  tri_count     INTEGER,
  pt_count      INTEGER,
  search_text   TEXT NOT NULL,
  updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE (artifact_id, media_type, variant)
);

-- ───────────────────────────── 파일 ─────────────────────────────
-- 다운로드 단위. path 가 곧 URL (/files/<path>).
CREATE TABLE IF NOT EXISTS files (
  path          TEXT PRIMARY KEY,               -- originals/bon002789_.../photo_2d/bon002789_photo_01.jpg
  asset_id      TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
  filename      TEXT NOT NULL,
  original_name TEXT,                           -- 정규화 전 원명
  kind          TEXT NOT NULL,                  -- photo | obj | mtl | texture | normalmap | ply | stl | las | report | image | other
  label_ko      TEXT NOT NULL,                  -- "사진 1/6" "모델 (OBJ)" "점군 (PLY)" "프린트용 (STL)" "복원 리포트 (JSON)"
  ext           TEXT,
  size          INTEGER NOT NULL,
  width         INTEGER,                        -- 이미지일 때
  height        INTEGER,
  sha256        TEXT,
  has_gz        INTEGER NOT NULL DEFAULT 0,
  previewable   INTEGER NOT NULL DEFAULT 0,
  preview_path  TEXT,                           -- 사진: 자기 자신 / obj·ply·stl: 카드의 GLB / 그 외 NULL
  sort_order    INTEGER NOT NULL DEFAULT 0,
  mtime         TEXT,
  scanned_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_assets_media    ON assets(media_type);
CREATE INDEX IF NOT EXISTS idx_assets_artifact ON assets(artifact_id);
CREATE INDEX IF NOT EXISTS idx_files_asset     ON files(asset_id);
CREATE INDEX IF NOT EXISTS idx_files_kind      ON files(kind);
CREATE INDEX IF NOT EXISTS idx_files_sha       ON files(sha256);

-- ───────────────────────────── 즐겨찾기 ─────────────────────────────
-- 팀 공용 (인증이 없으니 개인별이 아니라 "팀이 찍은 것"). 재스캔은 assets 를 다시 만들지만 이 표는 건드리지 않는다.
-- FK 를 걸지 않는 이유: 재스캔 때 assets 행이 지워지며 CASCADE 로 함께 사라지는 것을 막기 위해.
CREATE TABLE IF NOT EXISTS favorites (
  asset_id    TEXT PRIMARY KEY,
  note        TEXT,                              -- 왜 찍었는지 (선택)
  created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 팀 피드백 게시판. 채팅처럼 쌓이고, 항목마다 처리 상태를 바꾼다.
-- favorites 와 같은 이유로 FK 를 걸지 않는다 — 재스캔이 이 표를 건드리면 안 된다.
CREATE TABLE IF NOT EXISTS feedback (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  body        TEXT NOT NULL,                     -- 자유롭게 적는다
  author      TEXT,                              -- 누가 (선택)
  status      TEXT NOT NULL DEFAULT 'new',       -- new 미확인 | seen 확인 | hold 대기 | doing 진행 | done 완료
  created_at  TEXT NOT NULL DEFAULT (datetime('now')),   -- UTC. 'localtime' 은 서버(UTC)와 보는 사람(KST)이 9시간 어긋났다
  updated_at  TEXT
);
CREATE INDEX IF NOT EXISTS ix_feedback_id ON feedback(id);

-- 목록 화면이 그대로 읽는 뷰 (스캔마다 다시 만들어 열 추가가 반영되게 한다)
DROP VIEW IF EXISTS v_cards;
CREATE VIEW v_cards AS
SELECT a.asset_id, a.media_type, a.variant, a.variant_ko, a.title_ko, a.subtitle_ko,
       a.method, a.owner, a.restore_types, a.produced_at, a.registered_at,
       a.thumb_path, a.preview_path, a.preview_kind, a.file_count, a.total_bytes, a.tri_count, a.pt_count,
       r.artifact_id, r.accession, r.name_ko, r.name_en, r.designation, r.period,
       r.material, r.category, r.source_org, r.museum, r.license_status, r.kogl_type, r.tags,
       -- VR 에서 볼 수 있는 유물 = 유물 태그에 'VR' 이 있는 것 (시연: 즐겨찾기에서 VR 가능한 것을 먼저 보여준 뒤 VR 로)
       CASE WHEN (',' || REPLACE(LOWER(COALESCE(r.tags,'')), ' ', '') || ',') LIKE '%,vr,%' THEN 1 ELSE 0 END AS vr,
       CASE WHEN f.asset_id IS NULL THEN 0 ELSE 1 END AS starred, f.created_at AS starred_at, f.note AS star_note,
       a.search_text
FROM assets a JOIN artifacts r ON r.artifact_id = a.artifact_id
              LEFT JOIN favorites f ON f.asset_id = a.asset_id;

-- 스캔 이력
CREATE TABLE IF NOT EXISTS scans (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at  TEXT NOT NULL,
  finished_at TEXT,
  artifacts   INTEGER, assets INTEGER, files INTEGER, total_bytes INTEGER,
  note        TEXT
);
