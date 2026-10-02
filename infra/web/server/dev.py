"""로컬 개발용 — 서버에서는 nginx 가 하는 정적 파일 서빙(/, /files, /preview)을 uvicorn 하나로 흉내낸다.

  cd server && uv run python dev.py            # http://127.0.0.1:8412/
환경변수: CATALOG_DB · CATALOG_FILES · CATALOG_PREVIEW (기본값은 ~/c201-assets/*)
"""
import os, sys
os.environ.setdefault("CATALOG_DB", os.path.expanduser("~/c201-assets/catalog.db"))
os.environ.setdefault("CATALOG_FILES", os.path.expanduser("~/c201-assets/staging"))
os.environ.setdefault("CATALOG_PREVIEW", os.path.expanduser("~/c201-assets/preview"))

import uvicorn
from fastapi.staticfiles import StaticFiles
from app.main import app, FILES_DIR, PREVIEW_DIR, ZIPS_DIR

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
os.makedirs(ZIPS_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=FILES_DIR, html=False), name="files")
app.mount("/preview", StaticFiles(directory=PREVIEW_DIR, html=False), name="preview")
app.mount("/zips", StaticFiles(directory=ZIPS_DIR, html=False), name="zips")
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8412
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
