"""
재질 비교 — 색과 질감을 함께, v28 을 기준으로 (로드맵 S10)

왜
  "자연스러운가"는 결국 눈이 판정한다. 그래도 **눈 판정이 무엇을 보고 있었는지**를
  숫자로 잡아두면 다음 유물에서 1차 필터로 쓸 수 있다.

  v28 채택 이유는 *"채움이랑 표면 질감이 기존과 연결돼서"* 였다. 두 가지를 말한다.
    - **색**   채움면 색이 관측부와 이어지는가
    - **질감** 채움면 질감이 관측부와 **같은 종류**인가 (세밀함의 양이 아니다)

  앞서 쓴 단일 고주파 값은 절대 디테일 양만 재서 1024 를 이기게 만들었다.
  파워 스펙트럼 거리로 바꾸면 순위가 눈 판정과 맞는다. 여기서는 그걸
  **휘도만이 아니라 Lab 세 채널**로 넓히고 색차를 같이 본다.

무엇을 재나
  A. 내부 이질감   그 GLB 안에서 관측부 ↔ 채움부      ← "보기에 자연스러운가"
  B. 관측부 충실도 관측부가 v24 원본과 같은가           ← "원본을 지켰는가"

  A 만 보면 전 면을 새로 생성한 판(v27·v29)이 유리해진다 — 같은 모델이 만들었으니
  내부는 일관된다. B 가 그것을 잡는다. **둘을 같이 봐야 한다.**

기준
  v28 을 1.00 으로 놓고 상대값으로 적는다. 낮을수록 좋다.

사용
  python compare_material.py                    # 전 버전 비교 + 리포트/이미지
  python compare_material.py --ref v28-...      # 기준 바꾸기
"""

import argparse
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass


def log(m):
    print(f"[CMP] {m}", flush=True)


# 비교 대상. (폴더, GLB 파일명, 한 줄 설명)
VERSIONS = [
    ("v24-cyl+biharm+n6+nn+up2",                     "restored_direct.glb",        "최근접 정점색"),
    ("v27-cyl+biharm+n6+trellistex+up2",             "restored_trellistex.glb",    "전 면 TRELLIS 512"),
    ("v28-cyl+biharm+n6+trellisfill+up2",            "restored_trellisfill.glb",   "관측 원본 + 채움 512"),
    ("v29-cyl+biharm+n6+trellistex1024+up2",         "restored_trellistex1024.glb", "전 면 TRELLIS 1024"),
    ("v30-cyl+biharm+n6+trellisfill1024+up2",        "restored_trellisfill1024.glb", "관측 원본 + 채움 1024"),
    ("v31-cyl+biharm+n6+trellisfill1024+tonematch+up2", "restored_tonematch.glb",  "위 + RGB gain 톤보정"),
    ("v32-cyl+biharm+n6+trellisfill+labshift+up2",    "restored_labshift.glb",      "v28 + Lab 평균이동"),
]
ORIGIN = "v24-cyl+biharm+n6+nn+up2"        # 관측부 충실도의 원본


# ── 텍스처 읽기 · 유효 타일 ───────────────────────────────────────

def load_tex(glb, node):
    """노드의 baseColor 텍스처를 RGB 배열로. 정점색만 있는 노드는 None."""
    import trimesh
    s = trimesh.load(glb, process=False, force="scene")
    g = s.geometry.get(node)
    if g is None:
        return None
    t = getattr(getattr(getattr(g, "visual", None), "material", None), "baseColorTexture", None)
    return None if t is None else np.asarray(t.convert("RGB"), np.uint8)


def tiles(img, T=128, want=60, seed=0):
    """UV 아틀라스에서 **내용이 찬** 정사각 타일을 뽑는다.

    아틀라스는 조각 사이가 검게 비어 있어서 그대로 FFT 를 걸면 그 경계가 신호를 지배한다.
    거의 다 찬 타일만 고른다.
    """
    H, W, _ = img.shape
    if H < T or W < T:
        return []
    g = img.mean(2)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(want * 30):
        y, x = rng.integers(0, H - T), rng.integers(0, W - T)
        t = img[y:y + T, x:x + T]
        if (g[y:y + T, x:x + T] > 12).mean() > 0.97 and t.std() > 2:
            out.append(t)
        if len(out) >= want:
            break
    return out


# ── 색 ────────────────────────────────────────────────────────────

def lab_of(tile_list):
    """타일 묶음의 Lab. cv2 는 8bit RGB 에서 L∈[0,255], a,b∈[0,255](128 중심)로 준다."""
    import cv2
    a = np.concatenate([t.reshape(-1, 3) for t in tile_list], 0).astype(np.uint8)
    lab = cv2.cvtColor(a.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    lab[:, 0] *= 100.0 / 255.0            # L → [0,100]
    lab[:, 1:] -= 128.0                   # a,b → [-128,127]
    return lab


def delta_e76(lab1, lab2):
    """CIE76 색차. 평균 색 사이의 거리 — 지각 단위에 가깝다(ΔE 2~3 부터 눈에 보인다)."""
    return float(np.linalg.norm(lab1.mean(0) - lab2.mean(0)))


# ── 질감 ──────────────────────────────────────────────────────────

def spectrum(tile_list, ch, nb=24):
    """채널 ch 의 파워 스펙트럼 방사평균(정규화).

    ch: 0=L, 1=a, 2=b. 휘도만 보면 **색 얼룩의 공간 구조**를 놓친다.
    """
    import cv2
    if not tile_list:
        return None
    T = tile_list[0].shape[0]
    w = np.hanning(T)[:, None] * np.hanning(T)[None, :]
    acc = None
    for t in tile_list:
        lab = cv2.cvtColor(t, cv2.COLOR_RGB2LAB).astype(np.float64)
        x = lab[:, :, ch]
        P = np.abs(np.fft.fftshift(np.fft.fft2((x - x.mean()) * w))) ** 2
        acc = P if acc is None else acc + P
    P = acc / len(tile_list)
    c = T // 2
    yy, xx = np.mgrid[0:T, 0:T]
    r = np.hypot(yy - c, xx - c)
    b = np.linspace(1, T // 2, nb + 1)
    pr = np.array([P[(r >= b[i]) & (r < b[i + 1])].mean() for i in range(nb)])
    ssum = pr.sum()
    return pr / ssum if ssum > 0 else None


def spec_dist(p, q):
    """대칭 KL. 두 스펙트럼이 **같은 대역에 에너지를 두는가**를 잰다."""
    if p is None or q is None:
        return float("nan")
    p = np.maximum(p, 1e-12)
    q = np.maximum(q, 1e-12)
    return float(0.5 * ((p * np.log(p / q)).sum() + (q * np.log(q / p)).sum()))


# ── 한 버전 평가 ──────────────────────────────────────────────────

def evaluate(folder, fname, origin_tiles):
    glb = os.path.join(OUT, folder, fname)
    if not os.path.exists(glb):
        return None
    tc = load_tex(glb, "region_carried")
    tf = load_tex(glb, "region_filled")
    r = {"folder": folder, "glb": fname}
    if tc is None:
        r["error"] = "관측부 텍스처 없음"
        return r
    ct, ft = tiles(tc), (tiles(tf) if tf is not None else [])
    r["tiles"] = {"carried": len(ct), "filled": len(ft)}
    if not ft:
        r["note"] = "채움부에 텍스처 없음 (정점색). 내부 이질감 측정 불가"

    lc = lab_of(ct)
    r["carried_lab"] = [round(v, 2) for v in lc.mean(0)]

    # A. 내부 이질감 — 관측부 ↔ 채움부
    if ft:
        lf = lab_of(ft)
        r["filled_lab"] = [round(v, 2) for v in lf.mean(0)]
        r["internal_dE"] = round(delta_e76(lc, lf), 3)
        r["internal_spec"] = {}
        for ci, cn in enumerate("Lab"):
            r["internal_spec"][cn] = round(spec_dist(spectrum(ct, ci), spectrum(ft, ci)), 5)
        r["internal_spec_mean"] = round(float(np.mean(list(r["internal_spec"].values()))), 5)

    # B. 관측부 충실도 — v24 원본과 같은가
    r["carried_dE_vs_origin"] = round(delta_e76(lc, lab_of(origin_tiles)), 3)
    r["carried_spec_vs_origin"] = round(
        float(np.mean([spec_dist(spectrum(ct, i), spectrum(origin_tiles, i)) for i in range(3)])), 5)
    return r


# ── 그림 ──────────────────────────────────────────────────────────

def figure(reports, path):
    from PIL import Image, ImageDraw, ImageFont
    try:
        f = ImageFont.truetype(r"C:\Windows\Fonts\malgun.ttf", 14)
        fs = ImageFont.truetype(r"C:\Windows\Fonts\malgun.ttf", 12)
    except Exception:
        f = fs = ImageFont.load_default()

    rows = [r for r in reports if "error" not in r]
    S, PAD, TXT = 190, 10, 300
    W = PAD + S * 2 + PAD + TXT + PAD
    H = PAD + (S + PAD) * len(rows) + 30
    img = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(img)
    d.text((PAD, 8), "관측부 / 채움부  ·  v28 기준 상대값 (낮을수록 자연스럽다)", fill=(30, 30, 30), font=f)

    for i, r in enumerate(rows):
        y = 30 + i * (S + PAD)
        glb = os.path.join(OUT, r["folder"], r["glb"])
        for j, node in enumerate(("region_carried", "region_filled")):
            t = load_tex(glb, node)
            x = PAD + j * (S + 4)
            if t is None:
                d.rectangle([x, y, x + S, y + S], fill=(228, 228, 228))
                d.text((x + 12, y + S // 2 - 8), "정점색 (텍스처 없음)", fill=(120, 120, 120), font=fs)
                continue
            ts = tiles(t, T=128, want=1, seed=3)
            patch = Image.fromarray(ts[0] if ts else t[:128, :128])
            img.paste(patch.resize((S, S), Image.NEAREST), (x, y))
        d.text((PAD + 2, y + S - 16), "관측부", fill=(255, 255, 255), font=fs)
        d.text((PAD + S + 6, y + S - 16), "채움부", fill=(255, 255, 255), font=fs)

        tx = PAD + S * 2 + PAD + 6
        d.text((tx, y + 2), r["folder"].split("-")[0] + "  " + r.get("desc", ""), fill=(20, 20, 20), font=f)
        ln = y + 22
        for k, lab in (("rel_internal_dE", "내부 색차 ΔE"),
                       ("rel_internal_spec", "내부 질감차"),
                       ("rel_carried_dE", "관측부 색 변형"),
                       ("rel_carried_spec", "관측부 질감 변형")):
            v = r.get(k)
            raw = r.get(k.replace("rel_", ""), "")
            if v is None:
                if k + "_abs" not in r:
                    continue
                d.text((tx, ln), f"{lab:<14s} 원본 아님  ({raw})", fill=(190, 40, 40), font=fs)
                ln += 17
                continue
            col = (20, 120, 40) if v <= 1.02 else ((190, 110, 0) if v <= 2 else (190, 40, 40))
            if isinstance(raw, dict):
                raw = " ".join(f"{kk}{vv:.4f}" for kk, vv in raw.items())
            d.text((tx, ln), f"{lab:<14s} {v:5.2f}x   ({raw})", fill=col, font=fs)
            ln += 17
        if "note" in r:
            d.text((tx, ln), r["note"], fill=(150, 90, 0), font=fs)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path)
    log(f"그림 → {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="v28-cyl+biharm+n6+trellisfill+up2", help="기준 버전 폴더")
    ap.add_argument("--fig", default=os.path.join(OUT, "fig_material_compare.png"))
    args = ap.parse_args()

    og = os.path.join(OUT, ORIGIN, "restored_direct.glb")
    origin_tiles = tiles(load_tex(og, "region_carried"))
    log(f"원본 관측부 타일 {len(origin_tiles)} ({ORIGIN})")

    reports = []
    for folder, fname, desc in VERSIONS:
        r = evaluate(folder, fname, origin_tiles)
        if r is None:
            log(f"건너뜀 (없음) {folder}")
            continue
        r["desc"] = desc
        reports.append(r)

    ref = next((r for r in reports if r["folder"] == args.ref), None)
    assert ref is not None, f"기준 {args.ref} 을 못 찾았다"
    for r in reports:
        for key, rk in (("internal_dE", "rel_internal_dE"),
                        ("internal_spec_mean", "rel_internal_spec"),
                        ("carried_dE_vs_origin", "rel_carried_dE"),
                        ("carried_spec_vs_origin", "rel_carried_spec")):
            base, v = ref.get(key), r.get(key)
            if base is None or v is None:
                continue
            # 기준이 0 이면(관측부를 안 건드린 판) 배수가 의미 없다. 절대값으로 표시한다
            r[rk] = round(v / base, 3) if base > 1e-9 else (0.0 if v <= 1e-9 else None)
            if r[rk] is None:
                r[rk + "_abs"] = v

    log("")
    log(f"기준 = {args.ref}  (상대값 1.00). 낮을수록 좋다")
    log(f"{'버전':10s} {'설명':22s} {'내부색차':>9s} {'내부질감':>9s} {'관측색변형':>11s} {'관측질감변형':>13s}")
    log("  " + "-" * 84)
    for r in reports:
        if "error" in r:
            log(f"{r['folder'].split('-')[0]:10s} {r['error']}")
            continue
        def g(k):
            v = r.get(k)
            if v is None:
                return " 원본아님" if k + "_abs" in r else "     -   "
            return f"{v:8.2f}x"
        log(f"{r['folder'].split('-')[0]:10s} {r['desc']:22s} "
            f"{g('rel_internal_dE')} {g('rel_internal_spec')} "
            f"{g('rel_carried_dE')} {g('rel_carried_spec')}")
        json.dump(r, io.open(os.path.join(OUT, r["folder"], "material_report.json"), "w",
                             encoding="utf-8"), ensure_ascii=False, indent=2)

    log("")
    log("절대값 (ΔE 는 CIE76 · 질감은 대칭 KL, Lab 세 채널 평균)")
    for r in reports:
        if "error" in r:
            continue
        sp = r.get("internal_spec", {})
        spt = "  ".join(f"{k} {v:.4f}" for k, v in sp.items()) if sp else "-"
        log(f"  {r['folder'].split('-')[0]:6s} 내부 ΔE {str(r.get('internal_dE','-')):>7s} · "
            f"질감 {str(r.get('internal_spec_mean','-')):>8s}  [{spt}] | "
            f"관측부 vs 원본 ΔE {r['carried_dE_vs_origin']:6.3f} · 질감 {r['carried_spec_vs_origin']:.5f}")
    log("  각 폴더에 material_report.json 을 남겼다")

    figure(reports, args.fig)


if __name__ == "__main__":
    main()
