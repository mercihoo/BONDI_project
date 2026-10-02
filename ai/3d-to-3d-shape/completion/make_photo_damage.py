# -*- coding: utf-8 -*-
"""
make_photo_damage.py — **사진 단계에서** 유물을 깨서 TRELLIS 로 3D 를 만든다.

왜
  §5.6.2 에서 sim-to-real 격차의 원인을 소거법으로 하나만 남겼다.

      학습 완형 = **온전한** 사진의 TRELLIS
      71489     = **깨진**   사진의 TRELLIS

  내가 통제할 수 있는 셋(파단면·손상 복잡도·표면 잡음)은 전부 기각됐다.
  남은 것은 **상류 경로가 다르다**는 것인데, 3D 를 깨서는 못 고친다.

  **사진을 먼저 깨고 TRELLIS 를 태우면** 학습 데이터가 실제와 같은 경로를 탄다.
  정답(complete)은 원본 사진의 TRELLIS(`trellis3d/*.glb`) 를 그대로 쓴다.

어떻게 깨나
  `run_trellis_batch.py` 주석: *"알파 있으면 rembg 건너뜀"*.
  그래서 **알파를 뚫으면 TRELLIS 가 그 자리를 배경으로 본다.**

    1. `pipe.preprocess_image()` 로 배경 제거된 RGBA 를 얻고
    2. 아가리 쪽 알파를 **불규칙한 경계로** 0 으로 만들고
    3. 그 PNG 를 TRELLIS 에 다시 넣는다

파이프라인 로딩은 직접 하지 않는다
  `_apply_patches()`(flex_gemm)와 `pipe.cuda()` 가 빠지면
  `Input type (cuda) and weight type (cpu)` 로 죽는다. 실제로 그렇게 한 번 죽었다.
  그래서 `run_trellis_batch.load_pipeline()` · `run_one()` 을 **그대로** 불러 쓴다.

사용
  <trellis venv>\\python.exe make_photo_damage.py --limit 3      # 사진만
  <trellis venv>\\python.exe make_photo_damage.py --run-trellis  # 3D 까지 (유물당 ~7분)
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# `gupdari-shape-prior` · `gupdari-71489` 는 **이 폴더 밖**에 있다 (원본 사진·완형 3D·v32 출력).
# 저장소 안에서 돌릴 때는 `RESTORE_ROOT` 로 그 폴더들이 모인 곳을 가리킨다.
# 안 걸면 예전처럼 이 스크립트의 부모 폴더를 본다.
ROOT = Path(__import__("os").environ.get("RESTORE_ROOT", HERE.parent))
# 산출물(점군·메시·체크포인트)은 수 GB 라 저장소에 안 넣는다.
# `ADAPOINTR_WORK` 로 저장소 밖을 가리킬 수 있다. 없으면 이 폴더 밑 work/.
WORK = Path(__import__("os").environ.get("ADAPOINTR_WORK", HERE / "work"))
PRIOR = ROOT / "gupdari-shape-prior"
PHOTOS = PRIOR / "images"
OUT_IMG = WORK / "photo_damaged"
OUT_GLB = WORK / "trellis3d_damaged"
# --tag 로 세트를 나눈다 (얕은/깊은 훼손 비교용)

from make_pairs import complete_slugs  # noqa: E402  — 완형 9점 목록을 공유한다


def load_trellis():
    q = PRIOR / "run_trellis_batch.py"
    spec = importlib.util.spec_from_file_location("run_trellis_batch", q)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def photo_of(slug: str):
    import csv
    import re
    import unicodedata
    with (PRIOR / "meta" / "artifacts.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            s = re.sub(r"\s+", "_", unicodedata.normalize("NFC", r["소장품번호"]).strip())
            if s == slug:
                p = PHOTOS / r["file"]
                return p if p.is_file() else None
    return None


def ragged_alpha(alpha, base, amp, rng):
    """유물 **위쪽**을 가로 위치마다 다른 높이까지 지운다.

    사진은 옆에서 본 투영이라 θ 를 못 쓴다. 대신 x 를 따라 저주파 파형으로
    자르는 높이를 흔든다 — 3D 의 `ragged` 패턴과 같은 취지다.
    """
    ys, xs = np.nonzero(alpha > 0)
    if len(ys) == 0:
        return alpha
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    H = y1 - y0 + 1

    xg = np.arange(alpha.shape[1])
    xn = (xg - x0) / max(1, x1 - x0) * 2 * np.pi
    k = int(rng.integers(2, 5))
    w = rng.uniform(0.4, 1.0, k)
    w = w / w.sum()
    wave = sum(wi * np.sin(fi * xn + pi) for wi, fi, pi in zip(
        w, rng.integers(1, 4, k), rng.uniform(0, 2 * np.pi, k)))
    cut_y = y0 + (base + amp * wave) * H

    out = alpha.copy()
    yy = np.arange(alpha.shape[0])[:, None]
    out[yy < cut_y[None, :]] = 0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=float, default=0.35, help="평균적으로 위에서 몇 할")
    ap.add_argument("--amp", type=float, default=0.08, help="가로 위치별 들쭉날쭉")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-trellis", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="", help="출력 폴더 접미사. 세트를 나눈다")
    args = ap.parse_args()

    global OUT_IMG, OUT_GLB
    if args.tag:
        OUT_IMG = WORK / ("photo_damaged" + args.tag)
        OUT_GLB = WORK / ("trellis3d_damaged" + args.tag)

    from PIL import Image

    keep, _ = complete_slugs(PRIOR / "meta" / "artifacts.csv")
    slugs = sorted(keep)
    if args.limit:
        slugs = slugs[: args.limit]
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    print("완형 %d점 · 위에서 %.0f%% ± %.0f%% 를 지운다"
          % (len(slugs), 100 * args.base, 100 * args.amp))

    RT = pipe = None
    made = []
    for i, slug in enumerate(slugs, 1):
        jpg = photo_of(slug)
        if jpg is None:
            print("  [건너뜀] 사진 없음: " + slug)
            continue
        rgba_path = OUT_IMG / (slug + "_rgba.png")
        dmg_path = OUT_IMG / (slug + "_damaged.png")

        if not rgba_path.is_file():
            if pipe is None:
                RT = load_trellis()
                pipe = RT.load_pipeline()
            pipe.preprocess_image(Image.open(jpg)).save(rgba_path)

        im = Image.open(rgba_path).convert("RGBA")
        arr = np.array(im)
        rng = np.random.default_rng(args.seed + i)
        a2 = ragged_alpha(arr[:, :, 3], args.base, args.amp, rng)
        kept = (a2 > 0).sum() / max(1, (arr[:, :, 3] > 0).sum())
        arr[:, :, 3] = a2
        Image.fromarray(arr).save(dmg_path)
        print("  %2d/%d %-14s 남은 화소 %.0f%%" % (i, len(slugs), slug, 100 * kept))
        made.append((slug, dmg_path))

    print("\n훼손 사진 %d장 → %s" % (len(made), OUT_IMG))
    if not args.run_trellis:
        print("확인 후  --run-trellis  로 3D 를 만든다 (유물당 약 7분)")
        return 0

    OUT_GLB.mkdir(parents=True, exist_ok=True)
    if pipe is None:
        RT = load_trellis()
        pipe = RT.load_pipeline()
    for i, (slug, png) in enumerate(made, 1):
        out = OUT_GLB / (slug + ".glb")
        if out.is_file():
            print("  [건너뜀] 이미 있음: " + slug)
            continue
        t0 = time.time()
        nv, nf = RT.run_one(pipe, png, out, 200000, 1024)
        print("  %2d/%d %-14s V=%d F=%d · %.1f분"
              % (i, len(made), slug, nv, nf, (time.time() - t0) / 60), flush=True)
    print("\n훼손 3D → " + str(OUT_GLB))
    return 0


if __name__ == "__main__":
    sys.exit(main())
