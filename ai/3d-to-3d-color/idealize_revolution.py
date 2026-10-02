"""
idealize_revolution.py — 이상화 회전체 복원 (제작 당시 형상 버전)

빗살무늬토기처럼 회전체에 가까운 그릇 스캔에서 **축과 매끈한 반지름 곡선**을 맞추고, 깨지지 않은 회전체 메시(뾰족한 바닥,
벽 두께, 구연부)를 새로 만든 뒤, 스캔의 색과 새김 요철을 원통 좌표로 펴서 새 메시에 전사한다. 접합선·균열은 새 메시에
기하로 존재하지 않고, 색·요철은 접합선 띠와 검출된 균열을 좌우 결을 거울 복사해 채운다.
`restore_original_colour.py`(스캔 메시·텍스처를 고치는 노선)와는 다른 노선의 별도 버전.

  python idealize_revolution.py --model <obj> --out out_restore/poc_ssu022891_ideal --material earthenware
        [--texture <복원 텍스처 png>] [--crack-mask <png>] [--wall 10] [--tip pointed|scan] [--size 8192] [--reuse-raster]

출력: <stem>_restored_conserved_<tag>.glb, texture_restored_conserved_<tag>.png, normal_relief_<tag>.png, relief_<tag>.png,
      mask_coverage_<tag>.png, mask_crack_<tag>.png, profile_<tag>.png, raster_<tag>.npz(캐시), restore_<tag>.json
      (make_viewer_copies.py 가 그대로 읽음)
"""
import argparse
import json
import os
import time

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage as ndi
from scipy.signal import savgol_filter
from skimage import color as skcolor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from restore_original_colour import PRESETS, uv_seam_vertices, vertex_uv_lookup
from relief_maps import fill_nearest


# ----------------------------------------------------------------------------- axis + profile
def circle_fit(x, y):
    """Kasa algebraic circle fit with one MAD trim."""
    def solve(x, y):
        A = np.c_[x, y, np.ones_like(x)]
        sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
        cx, cy = sol[0] / 2, sol[1] / 2
        r = np.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9))
        return cx, cy, r
    cx, cy, r = solve(x, y)
    res = np.abs(np.hypot(x - cx, y - cy) - r)
    mad = np.median(res) + 1e-6
    keep = res < 3.0 * mad
    if keep.sum() >= 10:
        cx, cy, r = solve(x[keep], y[keep])
    return cx, cy, r


def basis(a):
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(a, t); e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    return e1, e2


def fit_axis(C, N, bin_mm=8.0, iters=4):
    """Axis (unit) and origin from per-slice circle fits of outward-facing face centres."""
    a = np.array([0, 0, 1.0]); o = C.mean(0)
    for _ in range(iters):
        p = C - o
        zc = p @ a
        rad = p - zc[:, None] * a
        r = np.linalg.norm(rad, axis=1)
        outward = (N * rad).sum(1) / np.maximum(r, 1e-9) > 0.3
        e1, e2 = basis(a)
        edges = np.arange(zc.min(), zc.max() + bin_mm, bin_mm)
        idx = np.digitize(zc, edges)
        cents, ws = [], []
        for b in np.unique(idx):
            sel = outward & (idx == b)
            if sel.sum() < 60:
                continue
            x, y = p[sel] @ e1, p[sel] @ e2
            cx, cy, rr = circle_fit(x, y)
            if not np.isfinite(rr) or rr <= 0 or rr > 1000:
                continue
            cents.append(o + cx * e1 + cy * e2 + zc[sel].mean() * a)
            ws.append(sel.sum())
        cents = np.array(cents); ws = np.array(ws, float)
        mu = (cents * ws[:, None]).sum(0) / ws.sum()
        X = (cents - mu) * np.sqrt(ws)[:, None]
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        a_new = vt[0]
        if a_new @ a < 0:
            a_new = -a_new
        a, o = a_new / np.linalg.norm(a_new), mu
    return a, o


def to_frame(P, a, o):
    """World -> axis frame (axis = +Z through the origin). Returns coords and the rotation R (rows = new basis)."""
    e1, e2 = basis(a)
    R = np.stack([e1, e2, a])
    return (P - o) @ R.T, R


def outer_profile(Cf, Nf, bin_mm=2.0):
    """Median outer radius per z bin from outward-facing face centres (inner surface and break faces excluded)."""
    r = np.hypot(Cf[:, 0], Cf[:, 1])
    outward = (Nf[:, 0] * Cf[:, 0] + Nf[:, 1] * Cf[:, 1]) / np.maximum(r, 1e-9) > 0.3
    z = Cf[:, 2]
    edges = np.arange(z.min(), z.max() + bin_mm, bin_mm)
    zc = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(z, edges) - 1, 0, len(zc) - 1)
    rmed = np.full(len(zc), np.nan); cnt = np.zeros(len(zc), int)
    for b in range(len(zc)):
        sel = outward & (idx == b)
        cnt[b] = sel.sum()
        if cnt[b] >= 15:
            rmed[b] = np.median(r[sel])
    return zc, rmed, cnt


def smooth_profile(zc, rmed, window_bins=21):
    good = np.isfinite(rmed)
    r_i = np.interp(zc, zc[good], rmed[good])
    w = min(window_bins | 1, (len(zc) // 2) * 2 - 1)
    return savgol_filter(r_i, w, 2)


def extend_tip(zc, r_s, fit_mm=30.0):
    """Extrapolate the lower profile linearly to r = 0 (pointed base); C1 at the junction."""
    sel = zc <= zc[0] + fit_mm
    slope, icpt = np.polyfit(zc[sel], r_s[sel], 1)      # r = slope*z + icpt
    if slope <= 1e-3:
        return zc, r_s, float(zc[0])
    z_tip = -icpt / slope
    z0 = zc[0]
    if z_tip >= z0 - 0.5:
        return zc, r_s, float(z0)
    z_ext = np.arange(z_tip, z0, zc[1] - zc[0])
    r_ext = np.clip(slope * z_ext + icpt, 0, None)
    return np.r_[z_ext, zc], np.r_[r_ext, r_s], float(z_tip)


def round_tip(zc, r_s, z_lo, r_lo, cap_mm=25.0):
    """Replace the bottom cap_mm of the profile by a circular arc (centre on the axis) fitted to the scan's own bottom
    (its lowest vertices: ssu022891 r ~ 10 mm at z_lo), so the base is rounded like the scan instead of a cone."""
    sel = zc <= zc[0] + cap_mm
    z = np.r_[zc[sel], z_lo]; r = np.r_[r_s[sel], r_lo]
    A = np.c_[2 * z, np.ones_like(z)]                       # r^2 + z^2 = 2*zc0*z + k  (circle centred on the axis)
    (zc0, k), *_ = np.linalg.lstsq(A, r * r + z * z, rcond=None)
    rho = float(np.sqrt(max(k + zc0 * zc0, 1e-6)))
    z_tip = float(zc0 - rho)
    z_join = zc[sel][-1]
    z_arc = np.linspace(z_tip, z_join, 40)
    r_arc = np.sqrt(np.clip(rho * rho - (z_arc - zc0) ** 2, 0, None))
    z_all = np.r_[z_arc[:-1], zc[~sel]]; r_all = np.r_[r_arc[:-1], r_s[~sel]]
    step = zc[1] - zc[0]
    z_u = np.arange(z_tip, z_all[-1], step)
    r_u = np.interp(z_u, z_all, r_all)
    r_u = savgol_filter(r_u, 11, 2)                         # smooth the arc / profile join
    r_u = np.clip(r_u, 0, None); r_u[0] = 0.0
    return z_u, r_u, z_tip


# ----------------------------------------------------------------------------- revolution mesh
def revolve(z, r, n_theta, v_lo, v_hi, z_range, flip=False):
    """Grid surface of revolution. UV: u = theta/2pi (seam column duplicated), v in [v_lo, v_hi] mapped from z over
    z_range (= the scanned z range, so the atlas does not depend on how far the base is extended)."""
    th = np.linspace(0, 2 * np.pi, n_theta + 1)
    Z, TH = np.meshgrid(z, th, indexing="ij")            # (nz, nt+1)
    R = np.repeat(r[:, None], n_theta + 1, axis=1)
    X = R * np.cos(TH); Y = R * np.sin(TH)
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    v = v_lo + (v_hi - v_lo) * np.clip((z - z_range[0]) / max(z_range[1] - z_range[0], 1e-9), 0, 1)
    UV = np.stack([(TH / (2 * np.pi)).ravel(), np.repeat(v[:, None], n_theta + 1, axis=1).ravel()], 1)
    nz, nt = len(z), n_theta + 1
    i = np.arange(nz - 1)[:, None] * nt + np.arange(n_theta)[None, :]
    q00, q01, q10, q11 = i, i + 1, i + nt, i + nt + 1
    F = np.concatenate([np.stack([q00, q10, q11], -1).reshape(-1, 3), np.stack([q00, q11, q01], -1).reshape(-1, 3)])
    if flip:
        F = F[:, ::-1]
    area = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    F = F[area > 1e-9]                                    # drop degenerate faces at a pole (r == 0 rows)
    return V, UV, F


def build_vessel(z, r_out, wall, n_theta, z_range, uv_gap=0.002, rim_taper_mm=25.0, rim_frac=0.6, rim_rings=6):
    """Outer shell + inner shell (offset by a wall that tapers to rim_frac over the top rim_taper_mm) + rounded rim
    (semicircle of the rim wall). Outer UV v in [0.5+gap, 1], inner in [0, 0.5-gap]."""
    Vo, UVo, Fo = revolve(z, r_out, n_theta, 0.5 + uv_gap, 1.0, z_range)
    t = np.clip((z - (z[-1] - rim_taper_mm)) / rim_taper_mm, 0, 1)
    wall_z = wall * (1.0 - (1.0 - rim_frac) * t)
    r_in = np.clip(r_out - wall_z, 0, None)
    keep = r_in > 0
    first = int(np.argmax(keep))
    z_in = np.r_[z[first] - 1e-3, z[first:]] if first > 0 else z
    r_in = np.r_[0.0, r_in[first:]] if first > 0 else r_in
    Vi, UVi, Fi = revolve(z_in, r_in, n_theta, 0.0, 0.5 - uv_gap, z_range, flip=True)
    nt = n_theta + 1
    th = np.linspace(0, 2 * np.pi, nt)
    # rounded rim: rings on a semicircle from the outer top ring (phi = 0) to the inner top ring (phi = pi)
    w_top = r_out[-1] - r_in[-1]
    r_mid, z_top = 0.5 * (r_out[-1] + r_in[-1]), z[-1]
    rim_V, rim_UV = [], []
    for phi in np.linspace(0, np.pi, rim_rings + 2)[1:-1]:
        rr = r_mid + 0.5 * w_top * np.cos(phi); zz = z_top + 0.5 * w_top * np.sin(phi)
        rim_V.append(np.stack([rr * np.cos(th), rr * np.sin(th), np.full(nt, zz)], 1))
        rim_UV.append(np.stack([th / (2 * np.pi), np.full(nt, (1.0 - uv_gap) if phi < np.pi / 2 else (0.5 - uv_gap - 0.001))], 1))
    V = np.concatenate([Vo, Vi] + rim_V); UV = np.concatenate([UVo, UVi] + rim_UV)
    top_o = np.arange(len(Vo) - nt, len(Vo))
    top_i = len(Vo) + np.arange(len(Vi) - nt, len(Vi))
    rings = [top_o] + [len(Vo) + len(Vi) + k * nt + np.arange(nt) for k in range(rim_rings)] + [top_i]
    j = np.arange(n_theta)
    Fr = np.concatenate([np.concatenate([np.stack([A[j], B[j], B[j + 1]], -1), np.stack([A[j], B[j + 1], A[j + 1]], -1)])
                         for A, B in zip(rings[:-1], rings[1:])])
    F = np.concatenate([Fo, Fi + len(Vo), Fr])
    m = trimesh.Trimesh(V, F, process=False)
    # orient: outer faces away from the axis, inner faces toward it, rim faces away from the rim arc centre
    fn = m.face_normals; c = m.triangles_center
    rad = np.c_[c[:, 0], c[:, 1], np.zeros(len(c))]; rad /= np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-9)
    n_o = len(Fo); n_i = len(Fi)
    wrong = np.zeros(len(F), bool)
    wrong[:n_o] = (fn[:n_o] * rad[:n_o]).sum(1) < 0
    wrong[n_o:n_o + n_i] = (fn[n_o:n_o + n_i] * rad[n_o:n_o + n_i]).sum(1) > 0
    cr = np.hypot(c[n_o + n_i:, 0], c[n_o + n_i:, 1])
    exp = rad[n_o + n_i:] * (cr - r_mid)[:, None] + np.c_[np.zeros(len(cr)), np.zeros(len(cr)), c[n_o + n_i:, 2] - z_top]
    wrong[n_o + n_i:] = (fn[n_o + n_i:] * exp).sum(1) < 0
    F2 = F.copy(); F2[wrong] = F2[wrong][:, ::-1]
    m = trimesh.Trimesh(V, F2, process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=UV)
    return m, (len(Vo), len(Vi))


# ----------------------------------------------------------------------------- cylindrical rasterisation of the scan
def rasterize(tri_px, attrs, zval, W, H, keep_max):
    """Software rasteriser: tri_px (F,3,2) pixel coords, attrs (F,3,K) interpolated, zval (F,3) depth (radius).
    keep_max=True keeps the largest radius per texel (outer wall), False the smallest (inner wall)."""
    K = attrs.shape[2]
    out = np.zeros((H, W, K), np.float32)
    zbuf = np.full((H, W), -np.inf if keep_max else np.inf, np.float32)
    cover = np.zeros((H, W), bool)
    for i in range(len(tri_px)):
        p = tri_px[i]
        x0 = int(np.floor(p[:, 0].min())); x1 = int(np.ceil(p[:, 0].max()))
        y0 = int(np.floor(p[:, 1].min())); y1 = int(np.ceil(p[:, 1].max()))
        if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
            continue
        x0, x1 = max(x0, 0), min(x1, W - 1); y0, y1 = max(y0, 0), min(y1, H - 1)
        xs = np.arange(x0, x1 + 1) + 0.5; ys = np.arange(y0, y1 + 1) + 0.5
        X, Y = np.meshgrid(xs, ys)
        (ax, ay), (bx, by), (cx, cy) = p
        d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(d) < 1e-12:
            continue
        l0 = ((by - cy) * (X - cx) + (cx - bx) * (Y - cy)) / d
        l1 = ((cy - ay) * (X - cx) + (ax - cx) * (Y - cy)) / d
        l2 = 1.0 - l0 - l1
        eps = -0.002
        inside = (l0 >= eps) & (l1 >= eps) & (l2 >= eps)
        if not inside.any():
            continue
        z = l0 * zval[i, 0] + l1 * zval[i, 1] + l2 * zval[i, 2]
        sub = zbuf[y0:y1 + 1, x0:x1 + 1]
        better = inside & ((z > sub) if keep_max else (z < sub))
        if not better.any():
            continue
        sub[better] = z[better]
        cover[y0:y1 + 1, x0:x1 + 1][better] = True
        a = l0[..., None] * attrs[i, 0] + l1[..., None] * attrs[i, 1] + l2[..., None] * attrs[i, 2]
        out[y0:y1 + 1, x0:x1 + 1][better] = a[better]
    return out, zbuf, cover


def unwrap_faces(Vf, Nf, faces, z_lo, z_hi, W, H, want_outward):
    """Scan faces -> (theta, z) pixel triangles for one wall; theta wrap handled by shifting + duplicating."""
    c = Vf[faces].mean(1)
    r = np.hypot(c[:, 0], c[:, 1])
    dot = (Nf[:, 0] * c[:, 0] + Nf[:, 1] * c[:, 1]) / np.maximum(r, 1e-9)
    sel = dot > 0.1 if want_outward else dot < -0.1
    F = faces[sel]
    th = np.arctan2(Vf[:, 1], Vf[:, 0]) % (2 * np.pi)
    T = th[F]                                              # (F,3)
    span = T.max(1) - T.min(1)
    wrap = span > np.pi
    T = np.where(wrap[:, None] & (T < np.pi), T + 2 * np.pi, T)
    x = T / (2 * np.pi) * W
    y = (1.0 - (Vf[F][..., 2] - z_lo) / (z_hi - z_lo)) * H
    px = np.stack([x, y], -1)
    return F, px, wrap


# ----------------------------------------------------------------------------- atlas repair
def masked_blur(x, mask, sigma):
    """Normalised-convolution Gaussian restricted to `mask`, computed on a block-mean downsample when sigma is large
    (8K atlas: sigma 340 px full-res took > 30 min; at 1/32 it is seconds). Result is bilinearly upsampled."""
    ds = max(1, int(sigma // 4))
    if ds == 1:
        num = ndi.gaussian_filter(x * mask, sigma)
        den = ndi.gaussian_filter(mask.astype(np.float32), sigma)
        return num / np.maximum(den, 1e-6)
    H, W = mask.shape
    Hp, Wp = -(-H // ds) * ds, -(-W // ds) * ds
    xm = np.zeros((Hp, Wp), np.float32); mm = np.zeros((Hp, Wp), np.float32)
    xm[:H, :W] = x * mask; mm[:H, :W] = mask
    xs = xm.reshape(Hp // ds, ds, Wp // ds, ds).mean((1, 3))
    ms = mm.reshape(Hp // ds, ds, Wp // ds, ds).mean((1, 3))
    num = ndi.gaussian_filter(xs, sigma / ds); den = ndi.gaussian_filter(ms, sigma / ds)
    low = num / np.maximum(den, 1e-6)
    up = ndi.zoom(low, ds, order=1)[:H, :W]
    return up.astype(np.float32)


def detect_cracks_atlas(L, relief, good, mm_px, relief_thr=-0.25, min_len_mm=40.0, max_thick_mm=4.0, hole_depth_mm=-1.0, hole_area_mm2=20.0):
    """Cracks / plaster fills inside a fragment and through-holes, on the unwrapped (theta, z) atlas. A crack is a groove
    in the radius residual (high-passed at 6 mm) that is also brighter than the local median (plaster), long (>= 40 mm)
    and thin (<= 4 mm). Comb strokes are grooves too but short and dark, so they fail the length or brightness test.
    Holes are deep (< -1 mm) compact components. 9/10 ssu022891 (2K test): the main in-fragment crack + 8 holes, no
    comb strokes; the brightness-only detector selected 69 % of the surface and the relief-only one the comb bands."""
    px = lambda mm: max(1, int(round(mm / mm_px)))
    top = L - ndi.grey_opening(L, size=(px(6), px(6)))
    t_med = float(np.median(top[good])) if good.any() else 0.0
    rel_hp = relief - masked_blur(relief, good, px(6))
    cand = good & (rel_hp < relief_thr) & (top > t_med)
    cand = ndi.binary_closing(cand, iterations=px(0.6))
    deep = good & (rel_hp < hole_depth_mm)
    lbl, n = ndi.label(cand | deep)
    if n == 0:
        return np.zeros_like(good)
    areas = np.bincount(lbl.ravel())[1:]
    objs = ndi.find_objects(lbl)
    keep = np.zeros_like(good)
    for i, sl in enumerate(objs):
        comp = lbl[sl] == i + 1
        h = sl[0].stop - sl[0].start; w = sl[1].stop - sl[1].start
        d = float(np.hypot(h, w))
        is_crack = d * mm_px >= min_len_mm and areas[i] / d * mm_px <= max_thick_mm
        is_hole = areas[i] * mm_px * mm_px >= hole_area_mm2 and float(rel_hp[sl][comp].mean()) < hole_depth_mm
        if is_crack or is_hole:
            keep[sl] |= comp
    return keep


def mirror_fill(img, good, band, sigma_low):
    """Fill `band` texels: low frequency from the (already interpolated) img, high frequency copied from the good texel
    mirrored across the band edge (q = 2*nearest - p), so seams and holes get real comb texture, not a smooth stripe."""
    H, W = good.shape
    idx = ndi.distance_transform_edt(~good, return_distances=False, return_indices=True)
    py, px = np.nonzero(band)
    if len(py) == 0:
        return img
    ny, nx = idx[0][py, px], idx[1][py, px]
    qy = np.clip(2 * ny - py, 0, H - 1); qx = (2 * nx - px) % W
    ok = good[qy, qx]
    qy = np.where(ok, qy, ny); qx = np.where(ok, qx, nx)
    out = img.copy()
    if img.ndim == 3:
        low = np.stack([fill_nearest(masked_blur(img[..., c], good, sigma_low), good) for c in range(img.shape[2])], -1)
    else:
        low = fill_nearest(masked_blur(img, good, sigma_low), good)
    detail = img - low
    out[py, px] = img[py, px] + detail[qy, qx]
    return out


def row_interp_fill(img, good):
    """Fill invalid texels row by row with periodic linear interpolation along theta (low frequency)."""
    H, W = good.shape
    out = img.copy()
    xs = np.arange(W)
    for y in range(H):
        g = good[y]
        if g.all() or not g.any():
            continue
        gx = xs[g]
        for c in range(img.shape[2]):
            vals = img[y, g, c]
            out[y, ~g, c] = np.interp(xs[~g], np.r_[gx - W, gx, gx + W], np.r_[vals, vals, vals])
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--material", default="earthenware", choices=sorted(PRESETS))
    ap.add_argument("--texture", default=None, help="colour source (default: the OBJ's own texture). Use the v1.6 restored texture for equalised fragment colours")
    ap.add_argument("--crack-mask", default=None, help="hand/SAM seam mask in scan texture space; those texels are refilled after transfer")
    ap.add_argument("--wall", type=float, default=None, help="wall thickness mm (default: measured outer-inner median radius gap, clipped 6..14)")
    ap.add_argument("--tip", default="round", choices=["round", "pointed", "scan"], help="round: circular arc fitted to the scan's own bottom (default); pointed: extrapolate the lower profile linearly to r=0 (cone); scan: stop where the scan stops")
    ap.add_argument("--size", type=int, default=8192, help="atlas size (outer wall = top half, inner = bottom half)")
    ap.add_argument("--n-theta", type=int, default=720)
    ap.add_argument("--z-step", type=float, default=1.5, help="mm between profile rings of the new mesh")
    ap.add_argument("--relief-mm", type=float, default=2.5, help="clip of the transferred relief (radius residual) for the normal map")
    ap.add_argument("--flatten", type=float, default=1.0, help="remove this fraction of the atlas's low-frequency colour (per-fragment tone / photographic shading), keeping the pattern. 0 = off")
    ap.add_argument("--flatten-mm", type=float, default=25.0, help="scale (Gaussian sigma, mm on the surface) below which colour variation is kept")
    ap.add_argument("--seam-mm", type=float, default=2.5, help="half-width (mm) of the fragment-join band that is refilled after transfer")
    ap.add_argument("--no-crack-detect", action="store_true", help="do not detect in-fragment cracks (bright + relief thin lines) on the atlas")
    ap.add_argument("--reuse-raster", action="store_true", help="reuse <out>/raster_<tag>.npz from a previous run (skip the 1-3 min rasterisation)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or f"{args.material}_ideal"
    preset = PRESETS[args.material]

    scene = trimesh.load(args.model, force="scene")
    mesh = trimesh.util.concatenate([g for g in scene.geometry.values()]) if len(scene.geometry) > 1 else list(scene.geometry.values())[0]
    tex_src = Image.open(args.texture).convert("RGB") if args.texture else mesh.visual.material.baseColorTexture.convert("RGB")
    tex = np.asarray(tex_src)
    Ht, Wt = tex.shape[:2]
    V = np.asarray(mesh.vertices, float); F = np.asarray(mesh.faces)
    C = mesh.triangles_center; N = mesh.face_normals
    print(f"{os.path.basename(args.model)}: {len(V)} verts, {len(F)} faces, texture {Wt}x{Ht}")

    # 1 axis + profile
    a, o = fit_axis(C, N)
    Vf, R = to_frame(V, a, o)
    Cf = Vf[F].mean(1)
    Nf = N @ R.T
    zc, rmed, cnt = outer_profile(Cf, Nf)
    r_s = smooth_profile(zc, rmed)
    z_scan_lo, z_scan_hi = float(Vf[:, 2].min()), float(Vf[:, 2].max())
    r_lo = float(np.median(np.hypot(*Vf[np.argsort(Vf[:, 2])[:200], :2].T)))       # radius of the scan's lowest vertices
    if args.tip == "pointed":
        zp, rp, z_tip = extend_tip(zc, r_s)
    elif args.tip == "round":
        zp, rp, z_tip = round_tip(zc, r_s, z_scan_lo, r_lo)
    else:
        zp, rp, z_tip = zc, r_s, float(zc[0])
    r_all = np.hypot(Cf[:, 0], Cf[:, 1])
    dot = (Nf[:, 0] * Cf[:, 0] + Nf[:, 1] * Cf[:, 1]) / np.maximum(r_all, 1e-9)
    wall = args.wall
    if wall is None:
        # wall = gap between the outer surface (20th pct radius) and the inner surface (80th pct) per 20 mm slice; medians
        # of both populations overstate it (11.5 mm on ssu022891) because break faces and misaligned fragments leak in
        ws = []
        for lo in np.arange(z_scan_lo + 30, z_scan_hi - 30, 20.0):
            s = (Cf[:, 2] >= lo) & (Cf[:, 2] < lo + 20)
            o_, i_ = s & (dot > 0.3), s & (dot < -0.3)
            if o_.sum() > 200 and i_.sum() > 200:
                ws.append(np.percentile(r_all[o_], 20) - np.percentile(r_all[i_], 80))
        wall = float(np.clip(np.median(ws), 4.0, 12.0)) if ws else 8.0
    print(f"   axis {np.round(a, 3)}  z range scan {z_scan_lo:.1f}..{z_scan_hi:.1f}  tip z {z_tip:.1f}  wall {wall:.1f} mm  rmax {rp.max():.1f}")

    # 2 mesh
    z_new = np.arange(z_tip, z_scan_hi, args.z_step); z_new = np.r_[z_new, z_scan_hi]
    r_new = np.interp(z_new, zp, rp)
    if args.tip in ("pointed", "round"):
        r_new[0] = 0.0
    ideal, (n_o, n_i) = build_vessel(z_new, r_new, wall, args.n_theta, (z_scan_lo, z_scan_hi))
    print(f"   ideal mesh {len(ideal.vertices)} verts, {len(ideal.faces)} faces")

    # 3 transfer: rasterise the scan into (theta, z) for outer + inner wall
    S = args.size; W, Hh = S, S // 2
    uv = np.asarray(mesh.visual.uv, float)
    seam_v = uv_seam_vertices(mesh)
    if args.crack_mask:
        cm = Image.open(args.crack_mask).convert("L")
        if cm.size != (Wt, Ht):
            cm = cm.resize((Wt, Ht), Image.NEAREST)
        seam_v |= vertex_uv_lookup(mesh, ndi.binary_dilation(np.asarray(cm) > 127, iterations=3))
    r_vert = np.hypot(Vf[:, 0], Vf[:, 1])
    attr_v = np.c_[uv, seam_v.astype(float)]                            # (V,3): su, sv, seam flag
    halves = {}
    cache = os.path.join(args.out, f"raster_{tag}.npz")
    reused = args.reuse_raster and os.path.exists(cache)
    if reused:
        zz = np.load(cache)
        for name in ("outer", "inner"):
            halves[name] = (zz[name + "_out"], zz[name + "_zb"], zz[name + "_cov"])
        print(f"   raster reused from {os.path.basename(cache)}")
    for name, outward, keep_max in (("outer", True, True), ("inner", False, False)):
        if name in halves:
            continue
        Fs, px, wrap = unwrap_faces(Vf, Nf, F, z_scan_lo, z_scan_hi, W, Hh, outward)   # atlas rows = scanned z range
        attrs = attr_v[Fs]; zval = r_vert[Fs]
        out, zb, cov = rasterize(px, attrs, zval, W, Hh, keep_max)
        if wrap.any():                                                  # faces crossing theta = 0: draw again shifted by -W
            px2 = px[wrap].copy(); px2[..., 0] -= W
            out2, zb2, cov2 = rasterize(px2, attrs[wrap], zval[wrap], W, Hh, keep_max)
            better = cov2 & ((zb2 > zb) if keep_max else (zb2 < zb))
            out[better] = out2[better]; zb[better] = zb2[better]; cov |= cov2
        halves[name] = (out, zb, cov)
        print(f"   {name}: {len(Fs)} faces rasterised, coverage {cov.mean():.1%}  {time.time() - t0:.0f}s")
    if not reused:
        np.savez(cache, **{f"{k}_{n}": v for k, (o_, z_, c_) in halves.items() for n, v in (("out", o_), ("zb", z_), ("cov", c_))})

    def sample(out, cov):
        sx = out[..., 0] * (Wt - 1); sy = (1.0 - out[..., 1]) * (Ht - 1)
        col = np.stack([ndi.map_coordinates(tex[..., c].astype(np.float32), [sy, sx], order=1, mode="nearest") for c in range(3)], -1)
        col[~cov] = 0
        return col

    atlas = np.zeros((S, S, 3), np.float32)
    good_all = np.zeros((S, S), bool)
    crack_all = np.zeros((S, S), bool)
    relief = np.zeros((Hh, W), np.float32)
    rows_z = z_scan_hi - (np.arange(Hh) + 0.5) / Hh * (z_scan_hi - z_scan_lo)
    r_row = np.interp(rows_z, zp, rp)
    du_mean = 2 * np.pi * float(r_row.mean()) / W                        # mm per texel along theta (mean row)
    crack_info = {}
    for k, name in enumerate(("outer", "inner")):
        out, zb, cov = halves[name]
        cov = ndi.binary_closing(cov, iterations=2) & ndi.binary_dilation(cov, iterations=2)
        seam = cov & (ndi.gaussian_filter(np.where(cov, out[..., 2], 0), 1.0) > 0.3)
        seam = ndi.binary_dilation(seam, iterations=max(1, int(round(args.seam_mm / du_mean))))
        col = sample(out, cov)
        # radius residual = relief (comb grooves, cracks, plaster ridges); its low frequency = profile misfit, removed
        res = np.where(cov, zb - r_row[:, None], 0.0).astype(np.float32)
        res = np.clip(res, -args.relief_mm, args.relief_mm)
        res = np.where(cov, res - masked_blur(res, cov, 6.0 / du_mean), 0.0)
        cracks = np.zeros_like(cov)
        if not args.no_crack_detect:
            L = skcolor.rgb2lab(np.clip(col, 0, 255) / 255.0)[..., 0].astype(np.float32)
            cracks = detect_cracks_atlas(L, res, cov & ~seam, du_mean)
            cracks = ndi.binary_dilation(cracks, iterations=max(1, int(round(1.5 / du_mean))))
            crack_info[name] = {"fraction_of_covered": float(cracks.sum() / max(cov.sum(), 1)), "components": int(ndi.label(cracks)[1])}
        good = cov & ~seam & ~cracks
        sparse = good.mean(1) < 0.3                                     # rows barely scanned (tip): rebuilt from the nearest dense row
        good[sparse] = False
        if args.flatten > 0:                                            # fragment tone / shading -> one flat body colour
            lab = skcolor.rgb2lab(np.clip(col, 0, 255) / 255.0).astype(np.float32)
            sig = args.flatten_mm / du_mean
            for c in range(3):
                low = masked_blur(lab[..., c], good, sig)
                mean_c = float(lab[..., c][good].mean())
                lab[..., c] = np.where(good, lab[..., c] + args.flatten * (mean_c - low), lab[..., c])
            col = (np.clip(skcolor.lab2rgb(lab), 0, 1) * 255).astype(np.float32)
        col = row_interp_fill(col, good)                                # low frequency across seams, cracks, holes
        band = cov & ~good & ~sparse[:, None]
        col = mirror_fill(col, good, band, 3.0 / du_mean)               # + mirrored comb texture
        any_row = good.any(1)
        if not any_row.all():                                           # unscanned rows (extended tip): flat clay colour of the nearest dense rows
            src = fill_nearest(np.arange(Hh), any_row)
            ref = np.arange(Hh)[any_row]
            base = np.stack([np.mean([col[r][good[r]].mean(0) for r in ref[np.argsort(np.abs(ref - y))[:20]]], 0) for y in np.flatnonzero(~any_row)])
            col[~any_row] = base[:, None, :]
        atlas[k * Hh:(k + 1) * Hh] = np.clip(col, 0, 255)
        good_all[k * Hh:(k + 1) * Hh] = good
        crack_all[k * Hh:(k + 1) * Hh] = cracks
        if name == "outer":
            res = np.where(good, res, 0.0)
            res = row_interp_fill(res[..., None], good)[..., 0]
            res = mirror_fill(res, good, band, 3.0 / du_mean)
            res[~any_row] = 0.0
            relief = res
    print(f"   atlas filled  {time.time() - t0:.0f}s")

    # normal map from the transferred relief (outer wall; inner flat)
    du_mm = 2 * np.pi * np.maximum(r_row, 1.0) / W                       # mm per texel along theta, per row
    dv_mm = (z_scan_hi - z_scan_lo) / Hh
    h = ndi.gaussian_filter(relief, 0.7)
    dcol = np.gradient(h, axis=1) / du_mm[:, None]
    drow = np.gradient(h, axis=0) / dv_mm
    nx = -dcol; ny = +drow; nz = np.ones_like(h)                         # glTF convention (see relief_maps.normal_from_height)
    n = np.stack([nx, ny, nz], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    nrm = np.full((S, S, 3), 128, np.uint8); nrm[..., 2] = 255
    nrm[:Hh] = ((n * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8)

    # 4 export in the scan's world frame
    stem = os.path.splitext(os.path.basename(args.model))[0]
    ideal.vertices = ideal.vertices @ R + o
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(np.clip(atlas, 0, 255).astype(np.uint8)),
                                              normalTexture=Image.fromarray(nrm), metallicFactor=float(preset["pbr"][0]),
                                              roughnessFactor=float(preset["pbr"][1]), doubleSided=True)
    ideal.visual = trimesh.visual.TextureVisuals(uv=ideal.visual.uv, material=mat)
    glb = os.path.join(args.out, f"{stem}_restored_conserved_{tag}.glb")
    ideal.export(glb, include_normals=True)
    Image.fromarray(np.clip(atlas, 0, 255).astype(np.uint8)).save(os.path.join(args.out, f"texture_restored_conserved_{tag}.png"))
    Image.fromarray(nrm).save(os.path.join(args.out, f"normal_relief_{tag}.png"))
    Image.fromarray((good_all * 255).astype(np.uint8)).save(os.path.join(args.out, f"mask_coverage_{tag}.png"))
    Image.fromarray((crack_all * 255).astype(np.uint8)).save(os.path.join(args.out, f"mask_crack_{tag}.png"))
    Image.fromarray(np.clip(relief / args.relief_mm * 127 + 128, 0, 255).astype(np.uint8)).save(os.path.join(args.out, f"relief_{tag}.png"))

    fig, ax = plt.subplots(1, 2, figsize=(11, 6))
    ax[0].scatter(rmed, zc, s=6, label="scan outer radius (median / 2 mm)")
    ax[0].plot(rp, zp, "r-", label="fitted profile")
    ax[0].plot(np.clip(rp - wall, 0, None), zp, "g--", label=f"inner (wall {wall:.1f} mm)")
    ax[0].set_xlabel("r [mm]"); ax[0].set_ylabel("z [mm]"); ax[0].axis("equal"); ax[0].legend(fontsize=8); ax[0].set_title("profile")
    prev = Image.fromarray(np.clip(atlas[:Hh], 0, 255).astype(np.uint8)).resize((1024, 512))
    ax[1].imshow(prev); ax[1].set_title("outer wall atlas (theta x z)"); ax[1].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, f"profile_{tag}.png"), dpi=110); plt.close(fig)

    rec = {"model": args.model, "material": args.material, "preset": {k: (list(v) if isinstance(v, tuple) else v) for k, v in preset.items()},
           "mode": "conserved", "method": "idealized_revolution", "texture_source": args.texture or "obj",
           "axis_world": a.tolist(), "origin_world": o.tolist(), "z_scan": [z_scan_lo, z_scan_hi], "z_tip": z_tip, "tip": args.tip,
           "wall_mm": wall, "r_max_mm": float(rp.max()), "n_theta": args.n_theta, "z_step_mm": args.z_step,
           "atlas": S, "coverage_outer": float(halves["outer"][2].mean()), "coverage_inner": float(halves["inner"][2].mean()),
           "ideal_faces": int(len(ideal.faces)), "cracks_detected": crack_info, "flatten": [args.flatten, args.flatten_mm], "seam_mm": args.seam_mm,
           "metrics": {}, "protect_dark": None, "chroma_strength": None, "tone": {}, "destain": {}, "normal_map": True,
           "crack_mask": args.crack_mask, "elapsed_s": time.time() - t0}
    json.dump(rec, open(os.path.join(args.out, f"restore_{tag}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"   wrote {os.path.basename(glb)} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
