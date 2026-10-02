"""텍스트 포맷 파일 옆에 .gz 를 만들어 nginx gzip_static 이 쓰게 한다 (AS-AT-08: 전송량 1/3).

서버에서 실행한다 (파일이 이미 거기 있으므로 다시 전송할 필요가 없다):
  cd /srv/catalog && uv run python app/make_gz.py            # files/ 전체, 없는 것만
  cd /srv/catalog && uv run python app/make_gz.py --force    # 전부 다시

압축 대상 판정 — 확장자만으로는 부족하다.
  .obj .mtl .json .md .csv .txt   항상 텍스트
  .ply                            헤더에 'format ascii' 일 때만
  .stl                            'solid' 로 시작하고 크기 != 84 + 50*삼각형수 일 때만 (바이너리 STL 도 'solid' 로 시작할 수 있다)
  .las                            바이너리지만 1.5~2배 줄어든다
  .jpg .png .glb .npy             안 함 (이미 압축 / 효과 없음)
.gz 는 원본보다 새 mtime 을 갖게 새로 쓴다 (gzip_static 은 오래된 .gz 를 무시한다).
"""
import gzip, os, shutil, struct, sys, time, argparse

ALWAYS = {".obj", ".mtl", ".json", ".md", ".csv", ".txt"}


def is_ascii_ply(p):
    with open(p, "rb") as f:
        return b"format ascii" in f.read(300)


def is_ascii_stl(p):
    with open(p, "rb") as f:
        if f.read(5) != b"solid":
            return False
        f.seek(80)
        n = struct.unpack("<I", f.read(4))[0]
    return os.path.getsize(p) != 84 + 50 * n


def want(p):
    ext = os.path.splitext(p)[1].lower()
    if ext in ALWAYS or ext == ".las":
        return True
    if ext == ".ply":
        return is_ascii_ply(p)
    if ext == ".stl":
        return is_ascii_stl(p)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("CATALOG_FILES", "/srv/catalog/files"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--level", type=int, default=6)
    a = ap.parse_args()
    n = skip = 0; before = after = 0; t0 = time.time()
    for r, _, fs in os.walk(a.root):
        for f in fs:
            if f.endswith(".gz"):
                continue
            p = os.path.join(r, f)
            if not want(p):
                continue
            gz = p + ".gz"
            if not a.force and os.path.exists(gz) and os.path.getmtime(gz) >= os.path.getmtime(p):
                skip += 1; continue
            with open(p, "rb") as fi, gzip.open(gz, "wb", compresslevel=a.level) as fo:
                shutil.copyfileobj(fi, fo, 1 << 20)
            n += 1; before += os.path.getsize(p); after += os.path.getsize(gz)
            if n % 50 == 0:
                print("  %d개 … %.1f GB → %.1f GB" % (n, before / 1e9, after / 1e9), flush=True)
    print("만듦 %d · 건너뜀 %d · %.2f GB → %.2f GB (%.0f%%) · %.1f분"
          % (n, skip, before / 1e9, after / 1e9, (after / before * 100) if before else 0, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
