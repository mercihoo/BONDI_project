#!/usr/bin/env python3
"""ERS 를 **양쪽 방향**으로 — "여기는 비어 있어야 한다" 를 정하는 마스크를 만든다.

왜 필요한가 (2026-09-18 측정):
LaS-Comp 의 ERS 는 `pred_voxel[voxel_mask] = 1.0` 한 줄, IAS 의 BCE 도 **관측 복셀에만** 걸린다.
빈 공간에는 제약이 전혀 없어서 그 자리를 TRELLIS 의 생성 사전분포가 채운다. 실측:

    굽다리 구간(격자 1칸 = 1복셀)      채운 칸    남은 창
    ① 원본 점군 (모델 입력)            74.4%     7 개 (16x12 · 11x7 · 11x7 · 10x5 …)
    ② 1단계 복셀 64³                  92.7%     2 개
    ③ 2단계 LaS 메시                  97.1%     2 개

**①에서 ②로 갈 때 잃는다.** 투창은 7~16 복셀 폭이라 해상도 문제가 아니다 — 격자를 만드는 순간 메워진다.

어떻게 막는가:
투창은 **벽 안쪽에 갇힌 빈 구멍**이다 (사방이 관측된 표면으로 둘러싸임). 결손은 그렇지 않다 —
파단면을 따라 위·아래로 열려 있다. 그래서 굽다리 벽을 원통으로 펼쳐 **갇힌 빈 덩어리만** 골라
그 자리의 복셀을 "비어 있어야 한다" 로 확정한다. 결손부는 건드리지 않으므로 복원은 그대로 된다.

이 마스크를 어디에 쓰나:
ERS 는 `pred_voxel = (decoder(x_0) > 0.0)` 로 점유를 정한다. 그래서 저자 코드를 고치지 않고
**디코더 로짓을 이 칸에서만 큰 음수로 눌러** 두면 매 스텝 0 이 된다
(quantize_trellis 의 install_ers_empty_hook). 관측 복셀과는 서로소라 ERS 의 1 고정과 충돌하지 않는다.

좌표계: run_lascomp_image_condition_single.py 의 `ss` 와 **같은 인덱스 공간**이다 —
(1,1,64,64,64), 축 순서는 LaS 내부 프레임의 (x, y, z). 세로축은 격자에서 자동 검출한다.
"""
import numpy as np

try:
    from scipy import ndimage
except Exception as e:                                          # pragma: no cover
    raise SystemExit("scipy 필요: %s" % e)


def _radial_profile(occ, up):
    """축 up 의 각 슬라이스에서 점유 칸의 98 백분위 반지름."""
    prof = np.full(occ.shape[up], np.nan)
    for k in range(occ.shape[up]):
        ij = np.argwhere(np.take(occ, k, axis=up))
        if len(ij) < 5:
            continue
        d = ij - ij.mean(axis=0)
        prof[k] = np.percentile(np.hypot(d[:, 0], d[:, 1]), 98)
    return prof


def detect_up_axis(occ):
    """세로축을 격자에서 찾는다 — 굽다리 그릇은 그 축으로 '넓음 → 목(좁음) → 굽다리(넓음)' 이 된다.
    가로축으로 보면 가운데가 가장 넓은 단봉이라 안쪽 최소가 깊지 않다. 그 깊이로 고른다."""
    best, best_score = 2, -1.0
    for up in (0, 1, 2):
        prof = _radial_profile(occ, up)
        ok = ~np.isnan(prof)
        if ok.sum() < 16:
            continue
        n = len(prof)
        score = -1.0
        for i in range(int(n * 0.25), int(n * 0.8)):
            if np.isnan(prof[i]) or not ok[:i].any() or not ok[i:].any():
                continue
            a, b = np.nanmax(prof[:i]), np.nanmax(prof[i:])
            score = max(score, (min(a, b) - prof[i]) / max(min(a, b), 1e-6))
        if score > best_score:
            best, best_score = up, score
    return best, best_score


def find_neck(prof):
    """몸통·굽다리 경계 = 안쪽에서 반지름이 가장 좁아지는 슬라이스."""
    n = len(prof)
    lo, hi = int(n * 0.25), int(n * 0.8)
    band = prof[lo:hi]
    if np.all(np.isnan(band)):
        return int(n * 0.4)
    return lo + int(np.nanargmin(band))


def keep_empty_mask(occ, up=None, foot_lo_frac=0.03, min_cells=4, rad_pad=2.0, verbose=True):
    """(64,64,64) 점유에서 '비어 있어야 하는' 칸을 골라 같은 모양의 bool 배열로 돌려준다."""
    occ = np.asarray(occ).astype(bool)
    if up is None:
        up, sc = detect_up_axis(occ)
        if verbose:
            print("[keep_empty] 세로축 = 축 %d (목 깊이 점수 %.3f)" % (up, sc))
    lat = [a for a in (0, 1, 2) if a != up]
    prof = _radial_profile(occ, up)
    filled = np.argwhere(~np.isnan(prof))[:, 0]
    if not len(filled):
        return np.zeros_like(occ), {}
    z0, z1 = int(filled.min()), int(filled.max())
    lo = z0 + int(round(foot_lo_frac * (z1 - z0)))
    hi = find_neck(prof)
    if hi - lo < 4:
        if verbose:
            print("[keep_empty] 굽다리 구간이 너무 얇다 (슬라이스 %d~%d) — 마스크 없음" % (lo, hi))
        return np.zeros_like(occ), {}

    idx = np.argwhere(occ)
    band = idx[(idx[:, up] >= lo) & (idx[:, up] <= hi)]
    if len(band) < 50:
        return np.zeros_like(occ), {}
    center = band[:, lat].mean(axis=0)
    dxy = band[:, lat] - center
    r_band = np.hypot(dxy[:, 0], dxy[:, 1])
    r_med = float(np.median(r_band))
    NH = hi - lo + 1
    NA = max(8, int(round(2 * np.pi * max(r_med, 2.0))))         # 1 칸 ≈ 1 복셀 호길이
    ai = np.clip((((np.arctan2(dxy[:, 1], dxy[:, 0]) + np.pi) / (2 * np.pi)) * NA).astype(int) % NA, 0, NA - 1)
    hh = band[:, up] - lo
    g = np.zeros((NH, NA), dtype=bool)
    g[hh, ai] = True

    r_wall = np.full(NH, r_med)                                  # 슬라이스별 벽 반지름
    for k in range(NH):
        m = hh == k
        if m.any():
            r_wall[k] = float(np.median(r_band[m]))

    # 갇힌 빈 덩어리 = 투창. 각은 순환이라 3 배로 펼쳐 찾고 가운데 사본만 센다.
    lab, n = ndimage.label(np.concatenate([~g] * 3, axis=1))
    holes, seen = [], set()
    for k in range(1, n + 1):
        ys, xs = np.where(lab == k)
        mid = (xs >= NA) & (xs < 2 * NA)
        if not mid.any() or ys.min() == 0 or ys.max() == NH - 1 or len(ys) < min_cells:
            continue
        key = (int(ys.min()), int(ys.max()), int(xs[mid].min()) % NA)
        if key in seen:
            continue
        seen.add(key)
        holes.append((ys, xs % NA, len(ys), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)))

    mask = np.zeros_like(occ)
    base = {"up": int(up), "band": [int(lo), int(hi)], "grid": [int(NH), int(NA)],
            "center": [float(c) for c in center], "r_med": r_med, "rad_pad": rad_pad,
            "observed_voxels": int(occ.sum())}
    if not holes:
        if verbose:
            print("[keep_empty] 갇힌 빈 구멍 없음 — 마스크 없음")
        return mask, dict(base, holes=0, mask_voxels=0)

    hole_cells = set()
    for ys, xs, *_ in holes:
        hole_cells |= set(zip(ys.tolist(), xs.tolist()))

    cand = np.argwhere((~occ))
    cand = cand[(cand[:, up] >= lo) & (cand[:, up] <= hi)]
    cdxy = cand[:, lat] - center
    cr = np.hypot(cdxy[:, 0], cdxy[:, 1])
    cai = np.clip(((((np.arctan2(cdxy[:, 1], cdxy[:, 0]) + np.pi) / (2 * np.pi)) * NA).astype(int) % NA), 0, NA - 1)
    chh = cand[:, up] - lo
    pick = np.fromiter(((h, a) in hole_cells for h, a in zip(chh.tolist(), cai.tolist())),
                       dtype=bool, count=len(cand))
    pick &= np.abs(cr - r_wall[chh]) <= rad_pad
    take = cand[pick]
    mask[take[:, 0], take[:, 1], take[:, 2]] = True
    mask &= ~occ                                                 # 관측 칸은 절대 건드리지 않는다

    info = dict(base, holes=len(holes),
                hole_sizes=[(int(c), int(a), int(b)) for _, _, c, a, b in holes],
                mask_voxels=int(mask.sum()))
    if verbose:
        print("[keep_empty] 굽다리 슬라이스 %d~%d · 격자 %dx%d · 벽 반지름 중앙값 %.1f 복셀"
              % (lo, hi, NH, NA, r_med))
        print("[keep_empty] 갇힌 구멍 %d 개 (칸수, 각폭x높이폭): %s"
              % (len(holes), " · ".join("%d칸 %dx%d" % (c, a, b) for _, _, c, a, b in holes[:6])))
        print("[keep_empty] 비어 있어야 하는 복셀 %d 개 (관측 %d 개)" % (mask.sum(), occ.sum()))
    return mask, info


# ── 단독 점검용 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    ap = argparse.ArgumentParser(description="partial.ply → 64³ 점유 → 비어야 하는 마스크 (점검용)")
    ap.add_argument("--partial", required=True)
    ap.add_argument("--dataset", default="custom")
    ap.add_argument("--yz-flip", action="store_true", default=True)
    ap.add_argument("--no-yz-flip", dest="yz_flip", action="store_false")
    ap.add_argument("--no-normalize", dest="normalize", action="store_false", default=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--rad-pad", type=float, default=2.0)
    ap.add_argument("--min-cells", type=int, default=4)
    a = ap.parse_args()

    sys.path.insert(0, ".")
    from plyfile import PlyData
    from run_lascomp_image_condition_single import (load_points_from_ply, normalize_point_cloud,
                                                    voxelize_unit_cube)

    xyz = load_points_from_ply(PlyData.read(a.partial), a.dataset, "auto", yz_flip=a.yz_flip)
    if a.normalize:
        xyz, _, _ = normalize_point_cloud(xyz)
    ss, kept, _ = voxelize_unit_cube(xyz, resolution=64, open_upper_bound=False)
    occ = ss[0].cpu().numpy().astype(bool)
    print("점 %d → 점유 복셀 %d" % (len(xyz), occ.sum()))
    mask, info = keep_empty_mask(occ, rad_pad=a.rad_pad, min_cells=a.min_cells)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "keep_empty.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    def dump(arr, path):
        import trimesh
        p = np.argwhere(arr).astype(np.float64)
        if not len(p):
            print("  (빈 집합)", path)
            return
        trimesh.PointCloud((p + 0.5) / 64.0 - 0.5).export(str(path))
        print("  저장:", path, len(p), "점")

    dump(occ, out / "keep_empty_observed.ply")
    dump(mask, out / "keep_empty_mask.ply")
