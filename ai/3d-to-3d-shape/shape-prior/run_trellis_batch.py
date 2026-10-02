r"""e뮤지엄 사진 43장을 TRELLIS.2-4B 로 3D 로 만든다.

왜
--
투창 개수를 **사진 한 장에서는 못 잰다** (`tuchang_from_photo.py` 참조).
읽히는 각도 범위가 `|θ| ≤ 58°` 뿐이라 N=3·4 면 한 단에 창이 평균 하나만 보인다.
주기를 재려면 둘은 보여야 하는데 짝이 없다.

**3D 면 그 한계가 없다.** 360° 를 다 보므로 창을 전부 셀 수 있다.
그리고 문헌에 개수가 적힌 12점이 **정답표**라, 만들어진 3D 로 `--nfold` 를 채점할 수 있다.
이 파이프라인 최초의 정답 있는 평가가 된다.

무엇을 만드나
-------------
유물마다 `<번호>.glb` 하나. 재질은 필요 없으므로 텍스처를 1024 로 줄이고
데시메이션도 낮춘다 — **여기서 볼 것은 기하뿐**이다.

돌리는 순서
-----------
1. 문헌에 개수가 적힌 것 (평가셋 · 최우선)
2. 나머지 `plain`
3. `handled` · `dish`

이어받기가 된다. 이미 GLB 가 있으면 건너뛴다. 중간에 끊겨도 다시 돌리면 이어진다.

사용:
  <trellis venv>\python.exe run_trellis_batch.py [--limit N] [--only-documented]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

# TRELLIS.2 설치 경로. 사람마다 다르므로 환경변수로 덮어쓴다.
#   PowerShell:  $env:TRELLIS2_HOME = "D:\trellis2-stableprojectorz_v22\code"
INSTALL = os.environ.get("TRELLIS2_HOME") or r"C:\Users\<USER>\Downloads\trellis2-stableprojectorz_v22\code"
if not os.path.isdir(INSTALL):
    raise SystemExit(f"TRELLIS.2 설치 폴더를 못 찾았다: {INSTALL}\n"
                     f"환경변수 TRELLIS2_HOME 에 trellis2 의 code 폴더 경로를 지정할 것")
os.environ.setdefault("HF_HOME", os.path.join(INSTALL, "models"))
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")
sys.path.insert(0, INSTALL)

HERE = Path(__file__).resolve().parent
OUT = HERE / "trellis3d"
HAN = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6}


def log(m):
    print(f"[TRELLIS] {m}", flush=True)


def documented(rec):
    s = rec.get("설명") or ""
    if "透窓" not in s:
        return None
    for pat in (r"각각\s*([0-9一二三四五六])\s*개", r"각\s*([0-9一二三四五六])\s*개",
                r"([0-9一二三四五六])\s*[개個]의?\s*[^,.。]{0,14}?透窓",
                r"透窓이?\s*([0-9一二三四五六])\s*[개個]"):
        m = re.search(pat, s)
        if m:
            g = m.group(1)
            return HAN.get(g, int(g) if g.isdigit() else None)
    return None


_SEEN: dict[str, str] = {}


def slug(rec):
    """소장품번호로 파일명을 만든다 — **다만 번호가 겹치는 유물이 있다.**

    `경주 4614` 가 둘이다 (`손잡이달린굽다리바리2-10.jpg` · `2-11.jpg`).
    처음엔 번호만 썼더니 뒤엣것이 '이미 있음' 으로 건너뛰어져 **한 점이 안 만들어졌다.**

    먼저 나온 것은 이름을 그대로 둔다 (이미 만들어 둔 42개와 어긋나지 않게).
    겹치는 것만 `_2`, `_3` 을 붙인다.
    """
    n = (rec.get("소장품번호") or Path(rec["file"]).stem).strip()
    base = re.sub(r"[^0-9A-Za-z가-힣]+", "_", n).strip("_")
    f = rec["file"]
    if _SEEN.get(base) in (None, f):
        _SEEN[base] = f
        return base
    k = 2
    while _SEEN.get(f"{base}_{k}") not in (None, f):
        k += 1
    _SEEN[f"{base}_{k}"] = f
    return f"{base}_{k}"


def load_pipeline():
    try:
        from pipeline_worker import _apply_patches
        _apply_patches()
    except Exception as e:
        log(f"경고: flex_gemm 패치 실패 — {e}")
    import torch
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 없음")
    log("파이프라인 로드 중 (TRELLIS.2-4B)")
    t0 = time.time()
    pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipe.cuda()
    log(f"로드 완료 {time.time()-t0:.1f}s")
    return pipe


def run_one(pipe, img_path: Path, out_path: Path, decimation: int, texsize: int):
    import o_voxel
    import torch
    from PIL import Image

    image = pipe.preprocess_image(Image.open(img_path))     # 알파 있으면 rembg 건너뜀
    torch.manual_seed(1)
    mesh = pipe.run(image)[0]
    nv, nf = int(len(mesh.vertices)), int(len(mesh.faces))
    mesh.attrs = mesh.attrs.float()
    for m in pipe.models.values():
        m.cpu()
    torch.cuda.empty_cache()
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation, texture_size=texsize,
        remesh=True, remesh_band=1, remesh_project=0, verbose=False,
    )
    glb.export(str(out_path))
    del mesh, glb
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return nv, nf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-documented", action="store_true")
    ap.add_argument("--decimation", type=int, default=200_000)
    ap.add_argument("--texsize", type=int, default=1024)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    recs = json.loads((HERE / "meta/artifacts.json").read_text(encoding="utf-8"))
    order = {"plain": 1, "dish": 2, "handled": 3}
    todo = []
    for r in recs:
        d = documented(r)
        todo.append((0 if d else 1, order.get(r.get("class"), 9), r, d))
    todo.sort(key=lambda t: (t[0], t[1], t[2].get("소장품번호") or ""))
    if a.only_documented:
        todo = [t for t in todo if t[3]]
    if a.limit:
        todo = todo[:a.limit]

    nd = sum(1 for t in todo if t[3])
    log(f"대상 {len(todo)}점 (문헌 개수 있는 것 {nd}점 우선) · 출력 {OUT}")

    pipe = None
    jl = OUT / "_runlog.jsonl"
    done = ok = fail = 0
    t_all = time.time()
    for i, (_, _, r, d) in enumerate(todo, 1):
        out = OUT / f"{slug(r)}.glb"
        tag = f"[{i}/{len(todo)}] {r.get('소장품번호')} ({r.get('class')})" \
              + (f" 문헌 {d}개" if d else "")
        if out.exists() and out.stat().st_size > 10_000:
            log(f"{tag} — 이미 있음, 건너뜀")
            done += 1
            continue
        if pipe is None:
            pipe = load_pipeline()
        t0 = time.time()
        try:
            nv, nf = run_one(pipe, HERE / "images" / r["file"], out, a.decimation, a.texsize)
            dt = time.time() - t0
            ok += 1
            log(f"{tag} — OK {dt:.0f}s · verts {nv:,} · {out.stat().st_size/1e6:.1f}MB")
            rec = dict(t=time.strftime("%H:%M:%S"), num=r.get("소장품번호"),
                       file=r["file"], glb=out.name, cls=r.get("class"),
                       documented=d, sec=round(dt, 1), verts=nv, faces=nf, ok=True)
        except Exception as e:
            fail += 1
            log(f"{tag} — 실패: {type(e).__name__}: {e}")
            traceback.print_exc()
            rec = dict(t=time.strftime("%H:%M:%S"), num=r.get("소장품번호"),
                       file=r["file"], cls=r.get("class"), documented=d,
                       ok=False, err=f"{type(e).__name__}: {e}")
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        with jl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done += 1

    log(f"끝 — 성공 {ok} · 실패 {fail} · 건너뜀 {done-ok-fail} · "
        f"총 {(time.time()-t_all)/60:.1f}분")
    log(f"로그: {jl}")


if __name__ == "__main__":
    main()
