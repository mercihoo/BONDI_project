"""
노드 합성 - 관측부는 원본, 채움부만 다른 판에서 가져온다 (MATERIAL.md)

왜 필요한가
  `texture_from_mesh.py texture` 는 **전 면**에 새 재질을 입힌다. 관측부까지 재생성되므로
  `NFR-ETH-003`(원본 무수정)이 재질 축에서 깨진다. 실측으로도 관측부 색이 ΔE 5.05(512) ·
  16.3(1024) 만큼 달라졌다 (MATERIAL.md §4).

  그런데 `to_glb` 를 노드마다 따로 돌려뒀으므로 **골라 쓰면 된다.** UV 마스크가 필요 없다.
  v28·v30·v31·v32 가 전부 이 스크립트 한 줄로 재현된다.

색 맞추기
  none      그대로 (v28 · v30)
  rgbgain   RGB 채널별 이득으로 평균을 맞춘다 (v31). **쓰지 말 것** —
            RGB→Lab 이 비선형이라 `a` 채널 공간 구조가 뒤틀린다. 질감 거리 6.7배 악화
  labshift  Lab 에서 평균만 평행이동 (v32, **권장**). 곱셈이 아니라 덧셈이라
            분산이 정의상 보존된다. 색차 dE 7.29 -> 1.48, 질감은 오히려 개선

사용
  # v28 = v24 관측부 + v27 채움 (512, 색 보정 없음)
  python compose_material.py --filled-from v27-cyl+biharm+n6+trellistex+up2 \
      --out v28-cyl+biharm+n6+trellisfill+up2 --glb-name restored_trellisfill.glb

  # v32 = 위 + Lab 평균이동  (현재 최적)
  python compose_material.py --filled-from v28-cyl+biharm+n6+trellisfill+up2 \
      --color-match labshift \
      --out v32-cyl+biharm+n6+trellisfill+labshift+up2 --glb-name restored_labshift.glb
"""

import argparse
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
CARRIED_DEFAULT = "v24-cyl+biharm+n6+nn+up2"

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass


def log(m):
    print(f"[CMPOSE] {m}", flush=True)


def find_glb(folder):
    d = os.path.join(OUT, folder)
    cand = [f for f in os.listdir(d) if f.endswith(".glb")]
    assert len(cand) == 1, f"{folder} 에 GLB 가 {len(cand)}개 — 하나여야 한다: {cand}"
    return os.path.join(d, cand[0])


# ── 색 맞추기 ─────────────────────────────────────────────────────

def _lab(u8):
    """uint8 RGB → float Lab (L∈[0,100], a·b∈[-128,127])."""
    import cv2
    l = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float64)
    l[..., 0] *= 100.0 / 255.0
    l[..., 1:] -= 128.0
    return l


def _rgb(l):
    import cv2
    x = l.copy()
    x[..., 0] *= 255.0 / 100.0
    x[..., 1:] += 128.0
    return cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def color_match(tex_fill, tex_ref, mode):
    """채움 텍스처를 관측 텍스처 평균색에 맞춘다. 아틀라스 빈칸(검정)은 건드리지 않는다."""
    if mode == "none":
        return tex_fill, {}
    mf = tex_fill.mean(2) > 12
    mr = tex_ref.mean(2) > 12

    if mode == "rgbgain":
        # v31 이 쓴 방식. 기록·재현용으로만 남긴다 — 질감을 망친다
        gain = tex_ref[mr].mean(0) / np.maximum(tex_fill[mf].mean(0), 1e-6)
        out = tex_fill.astype(np.float64).copy()
        out[mf] = np.clip(out[mf] * gain, 0, 255)
        log(f"  rgbgain {np.round(gain, 4).tolist()}  (권장하지 않음)")
        return out.astype(np.uint8), {"mode": mode, "gain": np.round(gain, 5).tolist()}

    if mode == "labshift":
        Lf, Lr = _lab(tex_fill), _lab(tex_ref)
        shift = Lr[mr].mean(0) - Lf[mf].mean(0)
        Ln = Lf.copy()
        Ln[mf] += shift
        pre = Ln[mf]
        clip = float(((pre[:, 0] < 0) | (pre[:, 0] > 100) |
                      (np.abs(pre[:, 1]) > 127) | (np.abs(pre[:, 2]) > 127)).mean())
        out = _rgb(Ln)
        out[~mf] = tex_fill[~mf]
        sd0, sd1 = Lf[mf].std(0), _lab(out)[mf].std(0)
        log(f"  labshift {np.round(shift, 3).tolist()} · 범위이탈 {100*clip:.3f}% · "
            f"표준편차 {np.round(sd0,3).tolist()} → {np.round(sd1,3).tolist()}")
        return out, {"mode": mode, "shift": np.round(shift, 4).tolist(),
                     "clip_frac": round(clip, 6)}

    raise SystemExit(f"모르는 color-match: {mode}")


# ── 합성 ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carried-from", default=CARRIED_DEFAULT,
                    help="관측부를 가져올 폴더. 기본은 형상 원본 v24")
    ap.add_argument("--filled-from", required=True, help="채움부를 가져올 폴더")
    ap.add_argument("--out", required=True, help="결과 폴더 이름 (out/ 아래)")
    ap.add_argument("--glb-name", default="restored_composed.glb")
    ap.add_argument("--color-match", default="none", choices=["none", "rgbgain", "labshift"])
    a = ap.parse_args()

    import trimesh
    from PIL import Image

    ga = trimesh.load(find_glb(a.carried_from), process=False, force="scene")
    gb = trimesh.load(find_glb(a.filled_from), process=False, force="scene")
    for name, sc, src in (("region_carried", ga, a.carried_from),
                          ("region_filled", gb, a.filled_from)):
        assert name in sc.geometry, f"{src} 에 {name} 노드가 없다"
    gc, gf = ga.geometry["region_carried"], gb.geometry["region_filled"]
    log(f"관측부 ← {a.carried_from}   ({len(gc.faces):,}면)")
    log(f"채움부 ← {a.filled_from}   ({len(gf.faces):,}면)")

    cm = {}
    if a.color_match != "none":
        tf = getattr(getattr(gf.visual, "material", None), "baseColorTexture", None)
        tc = getattr(getattr(gc.visual, "material", None), "baseColorTexture", None)
        assert tf is not None and tc is not None, "색 맞추기는 양쪽 다 텍스처가 있어야 한다"
        new, cm = color_match(np.asarray(tf.convert("RGB"), np.uint8),
                              np.asarray(tc.convert("RGB"), np.uint8), a.color_match)
        gf.visual.material.baseColorTexture = Image.fromarray(new)

    scene = trimesh.Scene()
    scene.add_geometry(gc, geom_name="region_carried")
    scene.add_geometry(gf, geom_name="region_filled")
    od = os.path.join(OUT, a.out)
    os.makedirs(od, exist_ok=True)
    p = os.path.join(od, a.glb_name)
    scene.export(p)

    # manifest — 형상 근거는 carried 쪽에서 물려받고, 재질 근거를 네 갈래로 적는다
    src_m = os.path.join(OUT, a.carried_from, "manifest.json")
    m = json.load(io.open(src_m, encoding="utf-8")) if os.path.exists(src_m) else {}
    fill_prov = m.get("provenance", {}).get("region_filled", "")
    tag = {"none": "", "labshift": " + Lab 평균 평행이동 (분산 보존, 사람이 건 보정)",
           "rgbgain": " + RGB 채널 이득 (질감 악화 — 재현용)"}[a.color_match]
    m["method"] = a.out.split("-", 1)[-1] if "-" in a.out else a.out
    m["provenance"] = {
        "region_carried": "observed (TRELLIS A단계, 무수정)",
        "region_carried_color": "observed (TRELLIS A단계 PBR, 무수정)",
        "region_filled": fill_prov,
        "region_filled_color": f"TRELLIS.2 텍스처링 (Trellis2TexturingPipeline 조립, AI 추정){tag}",
    }
    m["material_source"] = {"carried": a.carried_from, "filled": a.filled_from,
                            "color_match": cm or {"mode": "none"}}
    json.dump(m, io.open(os.path.join(od, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log(f"GLB → {p}  ({os.path.getsize(p)/1e6:.1f}MB · 노드 {len(scene.geometry)})")
    log("manifest 기록 완료")


if __name__ == "__main__":
    main()
