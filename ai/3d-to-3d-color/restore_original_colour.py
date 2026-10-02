"""Rule-based "original colour" restoration of a GLB texture (design v1).

Conservation procedure mapped onto CIE Lab texture processing of the restored GLB:

  1 condition survey   valid texels (UV raster) -> material preset -> healthy-surface detection -> damage map -> protection mask
  2 corrosion removal  damaged texels: shift chroma (a, b) towards the healthy/preset target, keep L (conserved)
  3 refinishing        target = healthy-surface robust mean if enough healthy surface, else material preset
  4 shape stabilisation Taubin smoothing, vertex displacement clamped to 0.07 % of the bbox diagonal
  5 record             JSON (preset, thresholds, fractions, dE, displacement stats), masks, comparison figure

Everything produced here is an *inference* (ai_inferred). The damage map doubles as the FR-UI-008 "restored region" layer.

Usage
  python restore_original_colour.py --model in.glb --out out_dir --material bronze [--mode conserved|pristine|both] [--no-smooth]
Materials: gold gilt_bronze bronze iron silver celadon whiteware earthenware stoneware
"""
import argparse
import json
import os
import time

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix, diags, identity
from scipy.sparse.linalg import splu
from skimage import color as skcolor
from skimage.metrics import structural_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from color_match import rot, render_points, sample_for_render
from relief_maps import used_texel_mask, height_from_colour, fill_nearest, normal_from_height, masked_blur
try:
    import cv2
except ImportError:      # optional: crack inpainting falls back to nearest-fill
    cv2 = None

# ----------------------------------------------------------------------------- presets (design section 3; initial values, sources pending)
PRESETS = {
    # Chroma targets converted from the 8/27 PoC MATERIAL_PRESETS (OpenCV 8-bit Lab: a*, b* = value - 128; L* = value / 2.55).
    # PoC L values are kept as "pristine_L" (only used by --mode pristine); conserved mode never touches L.
    #  name          group      target (a, b)     pristine L  PBR (metallic, roughness)  score weights
    "gold":        dict(group="metal",   target_ab=(12.0, 58.0), pristine_L=70.0, pbr=(1.0, 0.55), w_b=0.5, w_a=0.3, plausible_de=25.0),
    "gilt_bronze": dict(group="metal",   target_ab=(14.0, 54.0), pristine_L=67.5, pbr=(1.0, 0.55), w_b=0.5, w_a=0.3, secondary_ab=(17.0, 52.0), plausible_de=25.0),
    "bronze":      dict(group="metal",   target_ab=(17.0, 52.0), pristine_L=57.0, pbr=(1.0, 0.40), w_b=0.5, w_a=0.3),   # 갓 주조한 적금색
    "iron":        dict(group="metal",   target_ab=(1.0, 6.0),   pristine_L=43.0, pbr=(1.0, 0.50), w_b=0.0, w_a=0.3),
    "silver":      dict(group="metal",   target_ab=(0.0, 4.0),   pristine_L=74.5, pbr=(1.0, 0.30), w_b=0.0, w_a=0.3),
    "celadon":     dict(group="ceramic", target_ab=(-5.0, 5.0),  pristine_L=67.5, pbr=(0.0, 0.18), w_c=0.05),
    "whiteware":   dict(group="ceramic", target_ab=(-2.0, 6.0),  pristine_L=80.0, pbr=(0.0, 0.12), w_c=0.05),           # PoC "porcelain"
    "earthenware": dict(group="ceramic", target_ab=(13.0, 30.0), pristine_L=50.0, pbr=(0.0, 0.75), w_c=0.05),           # 소성 적갈색 태토
    "stoneware":   dict(group="ceramic", target_ab=(0.0, 3.0),   pristine_L=62.0, pbr=(0.0, 0.60), w_c=0.05),           # 회청색 경질토기 (added for this project)
}
PRESET_BLEND_L = 0.30        # v2.0: the preset's weight for LIGHTNESS. The ground truth puts the best lightness at
                             # 0.30 (dL +0.18 metal / -0.13 celadon) but the best chroma at 0.00, so the two are split.
PRESET_BLEND = 0.10          # weight of the material preset for CHROMA (a, b). PoC used 0.62, kept
                             # unquestioned until the synthetic-corrosion ground truth (9/13) showed it is the single
                             # largest error source: the preset sits 11.8 dE from bon009435's real gilding colour, and
                             # moving the surface to it cost dE_ab 9.9 against the truth. At 0.30 the same test gives
                             # 4.9 (metal) and 2.25 (celadon), with the lightness error falling to +0.2 / -0.1.
                             # The plausibility gate (plausible_de) still discards a healthy colour that is patina.
HEALTHY_QUANTILE = 0.15      # top 15 % of the healthiness score seed the healthy mask
GROW_DE = 6.0                # region growth: accept texels within this dE_ab of the healthy colour
MIN_HEALTHY_FRAC = 0.03      # below this, fall back to the preset target
DAMAGE_DE_FULL = 30.0        # dE_ab at which damage degree d reaches 1
PROTECT_DE = 25.0            # ceramics: colour clusters this far from the glaze are decoration (철화·청화)
INLAY_CHROMA = 12.0          # metals: chroma above this in a cool hue (cyan/blue/purple/red) is an inlay, not patina (bon004740 turquoise: C ~19, L ~47, hue ~190)
INLAY_L = 35.0               # ... and not a dark shadow
NEAR_BLACK_L = 25.0          # near-black texels are decoration/deep incisions, never corrosion
INCISED_ATTEN = 0.3          # incised (low height) texels keep 30 % of the colour shift
CLAMP_FRAC = 0.0007          # vertex displacement cap = 0.07 % of bbox diagonal
CHROMA_VAR_KEEP = {"ceramic": 0.15, "metal": 0.10}   # PoC refinish 0.25; lowered (v1.4/1.5) so patina patches and worn-rim beige do not survive as colour blotches
WEAR_CONVEX_Q = 0.90         # ceramics: brown texels on the most convex 10 % of the surface (rims, teeth, feet) are wear, not painting
BAND_SIGMAS = (3.0, 12.0, 48.0, 192.0)   # v1.9: scales (px at atlas resolution) whose local variation is capped
GEOM_DETAIL_RES = 2048       # v1.8: the mesh relief is rasterised at this size for the detail test and upsampled;
                             # a full 8K raster of 380k faces costs minutes and the evidence is coarse enough
FILL_BAND_MM = 3.0           # v1.7: the colour / relief fill band around a join or crack, in mm on the surface (was ~1 px of dilation)
GROOVE_RINGS = 3             # v1.6/1.7: fragment-join / crack grooves are re-solved as a biharmonic surface over seam vertices + this many rings (ssu022891: edge 1.8 mm, 3 rings ~ 11 mm band)
BORDER_BAND_PX = 24          # v1.6: UV-chart border band whose low-frequency colour is replaced by the chart interior's (ssu022891: outer 8 px L 67.7 vs interior 73.6)
CRACK_GEOM_CAP = 0.02        # PoC crack_geom_max_dev_ratio: vertex move cap (bbox diagonal fraction) when smoothing crack/seam grooves
PLAUSIBLE_DE = 15.0          # healthy-surface colour must lie within this dE_ab of the preset, else it is patina, not original surface.
                             # Gold and gilt bronze use 25 (preset plausible_de): they do not turn green, and PoC blended them (bon004740 15.2, bon009435 20.1)
DAMAGE_REPORT_D = 0.2        # texels with damage degree >= this count as "damaged" in reported fractions / VR layer


# ----------------------------------------------------------------------------- helpers
def robust_norm(x, mask):
    v = x[mask]
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 + 1e-6
    return (x - med) / mad


def de_ab(lab, ab):
    return np.hypot(lab[..., 1] - ab[0], lab[..., 2] - ab[1])


def dominant_ab(lab, mask):
    """Mode of the (a, b) 2-D histogram over the mask, refined by the mean of nearby texels."""
    a, b = lab[..., 1][mask], lab[..., 2][mask]
    H, ae, be = np.histogram2d(a, b, bins=[np.arange(-60, 61, 1.0), np.arange(-60, 81, 1.0)])
    i, j = np.unravel_index(np.argmax(H), H.shape)
    mode = np.array([(ae[i] + ae[i + 1]) / 2, (be[j] + be[j + 1]) / 2])
    near = np.hypot(a - mode[0], b - mode[1]) < 4
    return np.array([a[near].mean(), b[near].mean()]) if near.sum() > 50 else mode


def remove_small(mask, min_frac, ref_area):
    lab_img, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab_img, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_frac * ref_area
    return keep[lab_img]


def healthy_mask(lab, valid, preset):
    """Design 2.1: healthiness score, top-quantile seed, growth by colour distance, island removal."""
    L_, a_, b_ = (robust_norm(lab[..., c], valid) for c in range(3))
    if preset["group"] == "metal":
        score = L_ + preset["w_b"] * b_ - preset["w_a"] * np.abs(a_)
        ref = None
    else:
        ref = dominant_ab(lab, valid)
        score = L_ - preset["w_c"] * de_ab(lab, ref)
    thr = np.quantile(score[valid], 1.0 - HEALTHY_QUANTILE)
    seed = valid & (score >= thr)
    seed = remove_small(seed, 0.001, valid.sum())
    healthy = seed.copy()
    if seed.any():
        ab_h = np.array([lab[..., 1][seed].mean(), lab[..., 2][seed].mean()])
        close = valid & (de_ab(lab, ab_h) < GROW_DE) & (L_ > -0.5)
        for _ in range(3):
            healthy = valid & close & (ndi.binary_dilation(healthy, iterations=2) | healthy)
    return healthy, score, ref


def protection_mask(lab, rgba, valid, healthy, preset, ref_ab, height, protect_dark="all", alpha_sidecar=None, crack=None, dark_thr_fixed=None, convex=None, protect_painted=True):
    """Design 4 + PoC rule: holes (alpha channel or <stem>_alpha.jpg sidecar), incised lines (height map),
    painted/inlaid decoration (ceramics only) and structural near-black.
    PoC threshold: L* below clamp(5th percentile, 15.7, 23.5) (8-bit L 40..60 / 2.55), or a fixed L* (PoC protect_dark_thr,
    e.g. 150/2.55 = 58.8 for the iron-painted porcelain duk006294). Cracks are never protected."""
    holes = valid & (rgba[..., 3] < 128) if rgba.shape[-1] == 4 else np.zeros_like(valid)
    if alpha_sidecar is not None:
        holes |= valid & (alpha_sidecar < 100)
    dark_thr = float(dark_thr_fixed) if dark_thr_fixed is not None else float(np.clip(np.percentile(lab[..., 0][valid], 5), 15.7, 23.5))
    apply_dark = protect_dark == "all" or (protect_dark == "ceramic" and preset["group"] == "ceramic")
    near_black = valid & (lab[..., 0] < dark_thr) if apply_dark else np.zeros_like(valid)
    if crack is not None:
        near_black &= ~crack
    painted = np.zeros_like(valid)
    wear = np.zeros_like(valid)
    # The "painted" rule protects any ceramic texel far enough from the reference colour, which is right for an
    # iron-painted porcelain and wrong for a scan whose stains are simply off-colour: on the Silla stoneware
    # gubdari_v2 it froze the blue cast on the inner wall (1.1 % of the surface, 100 % of the blue that survived).
    # --protect-painted off turns it off when the artefact is known to carry no pigment.
    if preset["group"] == "ceramic" and ref_ab is not None and protect_painted:
        painted = valid & ~healthy & (de_ab(lab, ref_ab) > PROTECT_DE)
        painted = remove_small(painted, 0.0005, valid.sum())
        if convex is not None:
            # brown on rims / teeth / feet (most convex part of the shape) is glaze worn through to the body, not a brush stroke
            warm = (lab[..., 1] > 2) & (lab[..., 2] > 8)
            thr = np.quantile(convex[valid], WEAR_CONVEX_Q)
            wear = painted & warm & (convex > thr)
            wear = ndi.binary_dilation(wear, iterations=3) & painted & warm
            painted &= ~wear
    inlay = np.zeros_like(valid)
    if preset["group"] == "metal":
        # inlaid jade / glass / enamel on metalwork: bright and saturated in a non-warm hue. Patina is also non-warm
        # but dark and dull, so chroma + lightness separate them (bon004740: turquoise inlay was gilded over in v1.3)
        a_, b_, L_ = lab[..., 1], lab[..., 2], lab[..., 0]
        chroma = np.hypot(a_, b_)
        hue = np.degrees(np.arctan2(b_, a_)) % 360.0          # gold/bronze ~ 70-95 deg
        non_warm = ((hue > 175) & (hue < 320)) | (hue < 20)   # green patina sits at 130-180 and stays out
        inlay = valid & non_warm & (chroma > INLAY_CHROMA) & (L_ > INLAY_L)
        inlay = ndi.binary_dilation(remove_small(inlay, 0.00003, valid.sum()), iterations=2)
    exclude = holes | near_black | painted | inlay
    low_h = valid & (height < np.quantile(height[valid], 0.10))     # deep = incised; attenuated, not excluded
    return exclude, low_h, {"holes": holes, "near_black": near_black, "painted": painted, "inlay": inlay, "wear": wear, "incised": low_h, "dark_thr_L": dark_thr}


def detect_cracks(lab, valid, dark_thr=12.0 * 100 / 255, min_diag_px=250, max_thickness_px=12.0, extra_cand=None):
    """PoC detect_and_fill_cracks, classification part: long (diag >= min_diag) and thin (area/diag <= max_thickness)
    dark or bright line components are cracks / fragment seams; short pattern strokes and thick bands are not."""
    L = lab[..., 0]
    base = ndi.gaussian_filter(fill_nearest(L, valid), max(8.0, float(max_thickness_px)))   # base must be wider than the seam to see a wide bright ridge
    cand = valid & (((base - L) > dark_thr) | ((L - base) > dark_thr * 1.3))
    if extra_cand is not None:
        cand |= extra_cand                                   # v1.5: grooves found in the mesh join the texture candidates
    cand = ndi.binary_closing(cand, structure=np.ones((3, 3)))
    lbl, n = ndi.label(cand, structure=np.ones((3, 3)))
    crack = np.zeros_like(valid)
    kept = 0
    if n:
        objs = ndi.find_objects(lbl)
        areas = ndi.sum(cand, lbl, index=np.arange(1, n + 1))
        for i, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            h_, w_ = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            diag = float(np.hypot(w_, h_))
            if diag >= min_diag_px and areas[i - 1] / max(diag, 1.0) <= max_thickness_px:
                crack[sl] |= lbl[sl] == i
                kept += 1
    core = ndi.binary_dilation(crack, iterations=4)
    near = ndi.binary_dilation(crack, iterations=15)
    halo = ((L - base) > 8.0) & near & valid
    fill = ndi.binary_closing(core | halo, structure=np.ones((7, 7))) & valid
    return fill, kept


def raster_vertex_scalar(uv, faces, val, W, H):
    """Gouraud-rasterise a per-vertex scalar into texture space (barycentric interpolation, no z-buffer: UV charts do
    not overlap). Returns the image and its coverage."""
    px = np.stack([uv[:, 0] * (W - 1), (1.0 - uv[:, 1]) * (H - 1)], 1)
    out = np.zeros((H, W), np.float32)
    cov = np.zeros((H, W), bool)
    P = px[faces]
    A = val[faces]
    x0 = np.floor(P[:, :, 0].min(1)).astype(int); x1 = np.ceil(P[:, :, 0].max(1)).astype(int)
    y0 = np.floor(P[:, :, 1].min(1)).astype(int); y1 = np.ceil(P[:, :, 1].max(1)).astype(int)
    for i in range(len(faces)):
        xa, xb = max(x0[i], 0), min(x1[i], W - 1)
        ya, yb = max(y0[i], 0), min(y1[i], H - 1)
        if xb < xa or yb < ya:
            continue
        a, b, c = P[i]
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-12:
            continue
        X, Y = np.meshgrid(np.arange(xa, xb + 1) + 0.5, np.arange(ya, yb + 1) + 0.5)
        l0 = ((b[1] - c[1]) * (X - c[0]) + (c[0] - b[0]) * (Y - c[1])) / d
        l1 = ((c[1] - a[1]) * (X - c[0]) + (a[0] - c[0]) * (Y - c[1])) / d
        l2 = 1.0 - l0 - l1
        ins = (l0 >= -0.002) & (l1 >= -0.002) & (l2 >= -0.002)
        if not ins.any():
            continue
        v = l0 * A[i, 0] + l1 * A[i, 1] + l2 * A[i, 2]
        out[ya:yb + 1, xa:xb + 1][ins] = v[ins]
        cov[ya:yb + 1, xa:xb + 1][ins] = True
    return out, cov


def mesh_residual(mesh, W, H, iters=40):
    """v1.7: signed distance (mm) from every vertex to a heavily smoothed copy of the same mesh, along the smooth
    surface normal, rasterised into texture space. Negative = groove (incision, crack, join), positive = ridge. This is
    the object's real relief, unlike the colour-derived height map, so it drives the re-embossing of a filled band."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces)
    diag = float(np.linalg.norm(mesh.bounding_box.extents))
    key = np.round(V / (diag * 1e-6)).astype(np.int64)
    _, uidx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    U = V[uidx]
    Fu = inv[F]
    e = np.concatenate([Fu[:, [0, 1]], Fu[:, [1, 2]], Fu[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    n = len(U)
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1.0)
    X = U.copy()
    for _ in range(iters):                                        # Taubin: the smooth reference keeps the overall shape
        X = X + 0.5 * (A @ X / deg[:, None] - X)
        X = X - 0.53 * (A @ X / deg[:, None] - X)
    fn = np.cross(X[Fu[:, 1]] - X[Fu[:, 0]], X[Fu[:, 2]] - X[Fu[:, 0]])
    acc = np.zeros((n, 3))
    for k in range(3):
        np.add.at(acc, Fu[:, k], fn)
    nv = acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
    res_v = np.einsum("ij,ij->i", U - X, nv)[inv]
    res, cov = raster_vertex_scalar(np.asarray(mesh.visual.uv, dtype=np.float64), F, res_v, W, H)
    return res, cov, nv[inv]


def mirror_fill(img, good, weight, chart, sigma_low):
    """v1.7: give a filled band the surrounding pattern instead of a smooth streak. The low frequency stays as it is
    (TELEA / harmonisation result); the high frequency is REPLACED, in proportion to `weight`, by the detail of the
    texel mirrored across the edge of the intact area (q = 2*nearest_good - p). On a repeating pattern (comb strokes)
    that continues the strokes. The mirror source must lie in the same UV chart, otherwise a neighbouring fragment's
    pattern would be pasted in. Replacing (not adding) matters: the band overlaps intact texture, and adding doubled
    its contrast (ssu022891 band detail sd 4.2 vs 2.7 outside)."""
    H, W = good.shape
    band = weight > 0.01
    if not band.any() or not good.any():
        return img, 0.0
    idx = ndi.distance_transform_edt(~good, return_distances=False, return_indices=True)
    py, px = np.nonzero(band)
    ny, nx = idx[0][py, px], idx[1][py, px]
    qy = np.clip(2 * ny - py, 0, H - 1)
    qx = np.clip(2 * nx - px, 0, W - 1)
    ok = good[qy, qx] & (chart[qy, qx] == chart[ny, nx]) & (chart[ny, nx] > 0)
    qy = np.where(ok, qy, ny)
    qx = np.where(ok, qx, nx)
    if img.ndim == 3:
        low = np.stack([fill_nearest(masked_blur(img[..., c], good, sigma_low), good) for c in range(img.shape[2])], -1)
    else:
        low = fill_nearest(masked_blur(img, good, sigma_low), good)
    detail = img - low
    w = weight[py, px]
    if img.ndim == 3:
        w = w[:, None]
    out = img.copy()
    out[py, px] = img[py, px] + w * (detail[qy, qx] - detail[py, px])
    return out, float(ok.mean())


def fill_cracks(lab, valid, fill, radius=4):
    """Fill crack / seam texels. v1.4: OpenCV TELEA inpainting (PoC method) on the 8-bit Lab image; the v1.1 nearest-fill +
    blur left the seam visible as a soft band. Falls back to nearest-fill when cv2 is missing."""
    if not fill.any():
        return lab
    if cv2 is not None:
        lab8 = np.empty(lab.shape, np.uint8)
        lab8[..., 0] = np.clip(lab[..., 0] * 2.55, 0, 255)
        lab8[..., 1] = np.clip(lab[..., 1] + 128, 0, 255)
        lab8[..., 2] = np.clip(lab[..., 2] + 128, 0, 255)
        mask8 = (fill & valid).astype(np.uint8) * 255
        out8 = cv2.inpaint(lab8, mask8, radius, cv2.INPAINT_TELEA)
        out = lab.copy()
        out[..., 0] = np.where(fill, out8[..., 0] / 2.55, lab[..., 0])
        out[..., 1] = np.where(fill, out8[..., 1].astype(np.float64) - 128, lab[..., 1])
        out[..., 2] = np.where(fill, out8[..., 2].astype(np.float64) - 128, lab[..., 2])
        return out
    src_mask = valid & ~fill
    out = lab.copy()
    for c in range(3):
        filled = fill_nearest(lab[..., c], src_mask)
        out[..., c] = np.where(fill, ndi.gaussian_filter(filled, 3), lab[..., c])
    return out


def structure_coherence(L, valid, sigma_grad=1.0, sigma_win=3.0):
    """0..1 per texel: how line-like (oriented) the local lightness detail is (structure tensor coherence x gradient
    strength). Engraved lines, fish-scale edges and pattern strokes are coherent; corrosion pits and speckle are not.
    v1.4: used to keep decoration detail while dropping corrosion detail (bon009435 lost its scale pattern in v1.3)."""
    Lf = fill_nearest(L, valid).astype(np.float32)
    gx = ndi.gaussian_filter(Lf, sigma_grad, order=(0, 1))
    gy = ndi.gaussian_filter(Lf, sigma_grad, order=(1, 0))
    Jxx = ndi.gaussian_filter(gx * gx, sigma_win)
    Jyy = ndi.gaussian_filter(gy * gy, sigma_win)
    Jxy = ndi.gaussian_filter(gx * gy, sigma_win)
    del gx, gy
    tr = Jxx + Jyy
    disc = np.sqrt(np.maximum(tr * tr / 4 - (Jxx * Jyy - Jxy * Jxy), 0))
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    coh = ((l1 - l2) / (l1 + l2 + 1e-6)) ** 2
    mag = np.sqrt(np.maximum(l1, 0))
    strength = np.clip(mag / (np.percentile(mag[valid], 95) + 1e-6), 0, 1)
    return (coh * strength).astype(np.float32)


def block_blur(x, sigma, mask=None):
    """Gaussian blur on a block-mean downsample when sigma is large (sigma 192 on an 8K atlas takes minutes at full
    resolution); the result is band-limited anyway, so the bilinear upsample loses nothing. With a mask it is a
    NORMALISED convolution - without that, the atlas background bleeds into every UV island and the band decomposition
    puts a step at each island border (v1.9 first try shifted the mean colour by 8 b* on duk003312)."""
    ds = max(1, int(sigma // 4))
    H, W = x.shape

    def pad(a):
        Hp, Wp = -(-H // ds) * ds, -(-W // ds) * ds
        o = np.zeros((Hp, Wp), np.float32)
        o[:H, :W] = a
        return o

    if ds == 1:
        if mask is None:
            return ndi.gaussian_filter(x.astype(np.float32), sigma)
        m = mask.astype(np.float32)
        num = ndi.gaussian_filter(x.astype(np.float32) * m, sigma)
        den = ndi.gaussian_filter(m, sigma)
        return (num / np.maximum(den, 1e-6)).astype(np.float32)
    if mask is None:
        small = pad(x).reshape(-1, ds, pad(x).shape[1] // ds, ds).mean((1, 3))
        lo = ndi.gaussian_filter(small, sigma / ds)
    else:
        m = mask.astype(np.float32)
        xs = pad(np.asarray(x, np.float32) * m)
        ms = pad(m)
        sh = (xs.shape[0] // ds, ds, xs.shape[1] // ds, ds)
        num = ndi.gaussian_filter(xs.reshape(sh).mean((1, 3)), sigma / ds)
        den = ndi.gaussian_filter(ms.reshape(sh).mean((1, 3)), sigma / ds)
        lo = num / np.maximum(den, 1e-6)
    out = ndi.zoom(lo, ds, order=1)
    o = np.zeros((H, W), np.float32)
    h, w = min(H, out.shape[0]), min(W, out.shape[1])
    o[:h, :w] = out[:h, :w]
    return o


def match_band_variation(lab, valid, d, ref_lab, strength=1.0, sigmas=BAND_SIGMAS, cap=1.0, healthy=None):
    """v1.9: the restored surface must not carry MORE local variation than the object itself had. Measured on the gilt
    bronze duk003312, the restoration was amplifying it instead of removing it: the standard deviation of the 12-48 px
    band went 4.4 -> 6.6 in L and 3.2 -> 6.8 in b, which is exactly the blotchiness seen in the viewer (the tone
    renormalisation, the destain lift and the chroma-follows-lightness gamut scaling each stretch the surviving
    contrast). Here every band of the result is compared, inside the damaged area, with the same band of the reference
    (the surviving healthy surface when there is enough of it, otherwise the original texture) and scaled down when it
    is larger. Bands are never scaled UP, and the residual beyond the coarsest sigma - the form shading - is left
    alone, so shape and decoration survive."""
    dmg = valid & (d >= DAMAGE_REPORT_D)
    if not dmg.any():
        return lab, None
    if healthy is None or healthy.sum() <= 0.03 * max(valid.sum(), 1):
        # no surviving original surface to compare with. The corroded texture is NOT a reference (its own variation IS
        # the damage), so capping against it would be meaningless: skip. Those artefacts need a reference corpus.
        return lab, {"skipped": "healthy surface < 3 %"}
    ref_mask = healthy
    info = {"reference": "healthy", "bands": []}
    out = lab.astype(np.float32).copy()
    # weight: constant inside the damaged area, feathered at its border. Weighting by the damage degree instead (the
    # first version) makes the subtracted field correlate with d, and since d correlates with colour that SHIFTS the
    # mean of the damaged area - it tripled the gilt bronze crown's dE (0.72 -> 2.17) with factors as mild as 0.88.
    w = (strength * ndi.gaussian_filter(dmg.astype(np.float32), 2.0)).astype(np.float32)
    for c in range(3):
        cur = out[..., c].astype(np.float32)
        ref = ref_lab[..., c].astype(np.float32)
        prev_cur, prev_ref = cur, ref
        for sg in sigmas:
            lo_cur, lo_ref = block_blur(cur, sg, valid), block_blur(ref, sg, valid)
            band_cur, band_ref = prev_cur - lo_cur, prev_ref - lo_ref
            s_cur = float(band_cur[dmg].std())
            s_ref = float(band_ref[ref_mask].std())
            f = min(1.0, cap * s_ref / max(s_cur, 1e-6))
            if f < 0.999:
                # scale the band around its own mean over the damaged area, so the mean colour cannot move
                out[..., c] -= w * (1.0 - f) * (band_cur - float(band_cur[dmg].mean()))
            if c == 0:
                info["bands"].append({"sigma_px": sg, "std_restored": round(s_cur, 3), "std_reference": round(s_ref, 3),
                                      "factor_L": round(f, 3)})
            prev_cur, prev_ref = lo_cur, lo_ref
        del prev_cur, prev_ref, lo_cur, lo_ref, band_cur, band_ref
    return out.astype(np.float64), info


def geom_support(mesh, W, H, mm_px, res_px=GEOM_DETAIL_RES, geom_res=None, detail_mm=3.0, win_mm=2.0, q=80):
    """v1.8: per texel 0..1 - is there REAL geometric relief here, at the scale of the surface detail? The mesh residual
    (signed distance to a heavily smoothed copy of the same mesh) is band-passed and turned into a local energy, then
    normalised by its 80th percentile. Colour high frequency with no geometric counterpart is a stain or a corrosion
    discolouration, not a tool mark, so the detail re-attachment can drop it instead of baking it into the result."""
    if geom_res is not None:
        res, cov, mm_s = geom_res, np.abs(geom_res) > 0, mm_px
    else:
        small = min(res_px, W)
        res, cov, _ = mesh_residual(mesh, small, small)
        mm_s = mm_px * (W / small)
    s_hi = max(1.0, detail_mm / mm_s)
    hp = res - ndi.gaussian_filter(res, s_hi)
    e = np.sqrt(np.maximum(ndi.gaussian_filter(hp * hp, max(1.0, win_mm / mm_s)), 0))
    ref = float(np.percentile(e[cov], q)) if cov.any() else 1.0
    sup = np.clip(e / max(ref, 1e-9), 0, 1).astype(np.float32)
    if sup.shape != (H, W):
        sup = ndi.zoom(sup, (H / sup.shape[0], W / sup.shape[1]), order=1).astype(np.float32)
        out = np.zeros((H, W), np.float32)
        h, w = min(H, sup.shape[0]), min(W, sup.shape[1])
        out[:h, :w] = sup[:h, :w]
        sup = out
    return np.clip(sup, 0, 1)


def reattach_detail(L_restored, L_orig, d, detail_preserve=0.9, sigma=2.0, coh=None, support=None, support_strength=1.0, gate="any"):
    """PoC reattach_detail: put the original high-frequency lightness (carving, tool marks) back on the restored surface.
    Attenuated where the damage degree is high (there the detail is corrosion noise) - unless the detail is coherent
    (line-like), which is decoration and is kept even inside corroded areas (v1.4)."""
    hi = L_orig - ndi.gaussian_filter(L_orig, sigma)
    atten = 1.0 - 0.9 * d
    keep = detail_preserve * (atten if coh is None else (coh + (1.0 - coh) * atten))
    if support is not None and support_strength > 0:
        # damaged texel + no geometric relief under it = stain: drop its "detail". Healthy texels are untouched.
        # v2.0: gate on "is this texel damaged at all" (d / DAMAGE_REPORT_D, saturating), not on the degree. The
        # synthetic ground truth showed 57 % of a stained area sits below d = 0.2, where the old d factor left the
        # corrosion speckle almost untouched, and the restored surface carried 1.8-2.4x the true band energy.
        g = np.clip(d / DAMAGE_REPORT_D, 0, 1) if gate == "any" else np.clip(d, 0, 1)
        keep = keep * (1.0 - support_strength * g * (1.0 - support))
    return L_restored + hi * keep - (L_restored - ndi.gaussian_filter(L_restored, sigma)) * 0.3


def curvature_texture(mesh, W, H, q=0.99, concave=False):
    """Per-texel curvature sharpness rasterised from the mesh (0..1). convex (default): how strongly a vertex sticks out -
    rims, teeth, feet, edges - where glaze wears through to the body (v1.4 wear vs painting). concave=True: grooves -
    fragment seams and incisions (v1.5 seam detection from geometry, since wide low-contrast seams escape the texture rule)."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces)
    _, inv = np.unique(np.round(V, 6), axis=0, return_inverse=True)
    inv = inv.ravel()
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    acc = np.zeros((inv.max() + 1, 3))
    for k in range(3):
        np.add.at(acc, inv[F[:, k]], fn)
    nv = acc[inv]
    nv /= np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-12)
    fnu = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    cen = V[F].mean(1)
    sharp = np.zeros(inv.max() + 1)
    cnt = np.zeros(inv.max() + 1)
    for k in range(3):
        vi = F[:, k]
        side = np.einsum("ij,ij->i", cen - V[vi], nv[vi])
        sel = side > 0 if concave else side < 0                            # centroid under the tangent plane = convex, above = concave
        val = (1.0 - np.einsum("ij,ij->i", fnu, nv[vi])) * sel
        np.add.at(sharp, inv[vi], val)
        np.add.at(cnt, inv[vi], 1.0)
    sharp = sharp / np.maximum(cnt, 1)
    face_val = sharp[inv[F]].mean(1)
    scale = np.quantile(face_val, q) + 1e-9
    grey = np.clip(face_val / scale * 255, 0, 255).astype(np.uint8)
    img = Image.new("L", (W, H), 0)
    drw = ImageDraw.Draw(img)
    uv = np.asarray(mesh.visual.uv, dtype=np.float64)
    px = np.stack([uv[:, 0] * (W - 1), (1.0 - uv[:, 1]) * (H - 1)], axis=1)
    for tri, g in zip(F, grey):
        drw.polygon([tuple(px[i]) for i in tri], fill=int(g))
    return np.asarray(img, dtype=np.float32) / 255.0


def destain(L, valid, d, strength=0.85, size_px=512, pct=90, down=8, keep_sigma=0.0):
    """Even out corrosion blotches (v1.2). Corrosion darkens the surface in patches of tens to hundreds of texels; the
    surviving bright surface between them is the local reference. Estimate that reference as a high-percentile envelope
    of L at 1/down resolution (percentile filter + blur), then lift damaged texels toward it by d*strength. Large-scale
    shading survives because the envelope follows it; fine detail is re-attached afterwards by reattach_detail."""
    H, W = L.shape
    Lf = fill_nearest(L, valid)
    small = np.asarray(Image.fromarray(Lf.astype(np.float32), mode="F").resize((W // down, H // down), Image.BOX))
    k = max(3, int(round(size_px / down)) | 1)
    env_s = ndi.percentile_filter(small, pct, size=k)
    env_s = ndi.gaussian_filter(env_s, sigma=k / 2)
    env = np.asarray(Image.fromarray(env_s.astype(np.float32), mode="F").resize((W, H), Image.BILINEAR))
    # v2.1: lifting each texel to the envelope flattens everything darker than the envelope that is SMALLER than the
    # window - on the gilt bronze bon009435 that is the shaded half of every chased scale (12-48 px): pattern
    # retention 0.69, and the synthetic ground truth measured those bands at 0.44-0.59 of the true surface.
    lift = np.clip(env - L, 0, None) * np.clip(d * 2.0, 0, 1) * strength    # full lift once the texel is clearly damaged
    if keep_sigma > 0:
        # Smooth the LIFT FIELD at the pattern scale instead of deciding it per texel. The mean lift is unchanged (a
        # blur preserves the mean), so the blotch is removed exactly as before; what changes is that texels finer than
        # keep_sigma are all moved by the same amount, so the chased pattern's shading rides along instead of being
        # flattened. (A first version decided the lift on the blurred L instead; that under-lifted the dark texels and
        # cost 0.7-4 L against the ground truth.) Normalised by `valid` so the atlas background cannot dilute the lift.
        m = valid.astype(np.float32)
        raw_total = float(lift.sum())
        lift = ndi.gaussian_filter(lift * m, keep_sigma) / np.maximum(ndi.gaussian_filter(m, keep_sigma), 1e-6) * m
        # The blur leaks some lift across the blotch border onto texels the healthy pass-through then resets, so the
        # blotch itself ended up 0.5 L short of the per-texel lift (ground truth dL -2.45 -> -3.00). Restore the total.
        lift *= raw_total / max(float(lift.sum()), 1e-6)
    return L + lift, env


def gamut_scale_ab(L, a, b, k=1.15):
    """Dark texels cannot hold a saturated colour in sRGB (a dark 'gold' clips to brown mud). Scale chroma so that
    C <= k*L, which approximates the sRGB gamut boundary for yellow/orange hues. Identity for bright texels."""
    C = np.hypot(a, b)
    f = np.clip(k * np.maximum(L, 0) / np.maximum(C, 1e-6), 0, 1)
    return a * f, b * f


def uv_seam_vertices(mesh):
    """Boolean per vertex: this position appears with more than one UV (a UV-chart border). On glued-fragment scans
    (ssu022891) the charts are the fragments, so the chart borders ARE the join seams (v1.5)."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    uv = np.asarray(mesh.visual.uv, dtype=np.float64)
    _, inv = np.unique(np.round(V, 6), axis=0, return_inverse=True)
    inv = inv.ravel()
    key = np.round(uv, 5)
    order = np.lexsort((key[:, 1], key[:, 0], inv))
    inv_s, key_s = inv[order], key[order]
    new_grp = np.r_[True, inv_s[1:] != inv_s[:-1]]
    new_uv = np.r_[True, np.any(key_s[1:] != key_s[:-1], axis=1)]
    distinct = new_grp | new_uv                       # first row of every (group, uv) pair
    counts = np.bincount(inv_s[distinct], minlength=inv.max() + 1)
    return counts[inv] >= 2


def harmonize_uv_seams(lab, mesh, valid, band_px=16):
    """Remove the colour step across UV-chart borders: every seam vertex is sampled on each of its charts, the samples are
    averaged, and each chart is pulled toward that average inside a feathered band along its border. Texture-space
    inpainting cannot cross chart borders, which is why fragment seams survived v1.4 (ssu022891)."""
    H, W = valid.shape
    seam = uv_seam_vertices(mesh)
    if not seam.any():
        return lab, 0.0
    V = np.asarray(mesh.vertices, dtype=np.float64)
    uv = np.asarray(mesh.visual.uv, dtype=np.float64)
    _, inv = np.unique(np.round(V, 6), axis=0, return_inverse=True)
    inv = inv.ravel()
    idx = np.flatnonzero(seam)
    px = np.clip((uv[idx, 0] % 1.0) * (W - 1), 0, W - 1).astype(int)
    py = np.clip(((1.0 - uv[idx, 1]) % 1.0) * (H - 1), 0, H - 1).astype(int)
    # sample each copy (3x3 mean, valid texels only)
    smp = np.zeros((len(idx), 3))
    cnt = np.zeros(len(idx))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy = np.clip(py + dy, 0, H - 1)
            xx = np.clip(px + dx, 0, W - 1)
            ok = valid[yy, xx]
            smp[ok] += lab[yy[ok], xx[ok]]
            cnt[ok] += 1
    good = cnt > 0
    smp[good] /= cnt[good, None]
    g = inv[idx]
    gsum = np.zeros((inv.max() + 1, 3))
    gcnt = np.zeros(inv.max() + 1)
    np.add.at(gsum, g[good], smp[good])
    np.add.at(gcnt, g[good], 1)
    ok2 = good & (gcnt[g] >= 2)
    delta = np.zeros((len(idx), 3))
    delta[ok2] = gsum[g[ok2]] / gcnt[g[ok2], None] - smp[ok2]
    # rasterise the per-copy deltas and spread them inside the border band
    dsum = np.zeros((H, W, 3))
    dcnt = np.zeros((H, W))
    np.add.at(dsum, (py[ok2], px[ok2]), delta[ok2])
    np.add.at(dcnt, (py[ok2], px[ok2]), 1)
    has = dcnt > 0
    field = np.where(has[..., None], dsum / np.maximum(dcnt, 1)[..., None], 0)
    band = valid & ~ndi.binary_erosion(valid, iterations=band_px)
    dist_in = ndi.distance_transform_edt(valid)                     # distance to the chart border
    w = np.clip(1.0 - dist_in / band_px, 0, 1) * band
    out = lab.copy()
    for c in range(3):
        fc = fill_nearest(field[..., c], has)
        fc = ndi.gaussian_filter(fc, 3)
        out[..., c] = lab[..., c] + w * fc
    return (out, float(np.abs(delta[ok2][:, 0]).mean())) if ok2.any() else (out, 0.0)


def equalize_chart_borders(lab, valid, band_px=BORDER_BAND_PX, sigma=12.0):
    """v1.6: on fragment scans every UV chart is one photographed fragment and its border band is darker than the interior
    (join shadow / plaster; ssu022891 outer 8 px L 67.7 vs 73.6 inside). harmonize_uv_seams only equalises the two SIDES of
    a join, so a band that is dark on both sides survived v1.5. Here the low-frequency colour of the band is replaced by
    the interior's (normalised-convolution local mean of interior texels), feathered from 1 at the border to 0 at band_px;
    the high-frequency pattern (comb strokes) is untouched."""
    dist_in = ndi.distance_transform_edt(valid)
    band = valid & (dist_in <= band_px)
    interior = valid & (dist_in > band_px)
    if not band.any() or not interior.any():
        return lab, None
    w = (np.clip(1.0 - (dist_in - 1.0) / band_px, 0, 1) * band).astype(np.float32)
    m_all = valid.astype(np.float32)
    m_int = interior.astype(np.float32)
    den_all = ndi.gaussian_filter(m_all, sigma)
    den_int = ndi.gaussian_filter(m_int, sigma)
    has_int = den_int > 0.02
    out = lab.copy()
    for c in range(3):
        ch = lab[..., c].astype(np.float32)
        low_all = ndi.gaussian_filter(np.where(valid, ch, 0.0), sigma) / np.maximum(den_all, 1e-6)
        low_int = ndi.gaussian_filter(np.where(interior, ch, 0.0), sigma) / np.maximum(den_int, 1e-6)
        low_int = fill_nearest(low_int, has_int)                    # band texels far from any interior texel: nearest interior mean
        out[..., c] = lab[..., c] + w * (low_int - low_all)
    return out, {"band_px": band_px, "band_fraction_of_valid": float(band.sum() / valid.sum()),
                 "mean_dL_in_band": float((out[..., 0] - lab[..., 0])[band].mean())}


def fill_grooves(mesh, seed_vertex, diag, rings=GROOVE_RINGS, cap_frac=None, eps=1e-6):
    """v1.6: flatten the groove / ridge of fragment joins and cracks. Seed vertices (UV-chart borders, crack-mask texels)
    grown by `rings` graph rings are freed; everything else is fixed and the free band is re-solved by minimising the
    graph-Laplacian energy ||L x||^2 (biharmonic surface = C1 hole filling). Taubin on one vertex ring (v1.4/1.5) moved
    0.02 mm on ssu022891 and left every join visible under lighting (dihedral 12 deg at joins vs 7.7 elsewhere)."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    key = np.round(V / (diag * 1e-6)).astype(np.int64)
    _, uniq_idx, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    U = V[uniq_idx]
    F = inverse[np.asarray(mesh.faces)]
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    n = len(U)
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    seed_u = np.zeros(n, bool)
    np.logical_or.at(seed_u, inverse, seed_vertex)
    free = seed_u.copy()
    for _ in range(rings):
        free |= (A @ free.astype(np.float64)) > 0
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    Lap = (identity(n, format="csr") - diags(1.0 / deg) @ A).tocsr()
    B = (Lap.T @ Lap).tocsr()
    f = np.flatnonzero(free)
    rhs = -(B @ U)[f]                                            # displacement formulation: (B_ff + eps I) d = -(B U)_f
    Bff = (B[f][:, f] + eps * identity(len(f))).tocsc()
    d_f = splu(Bff).solve(rhs)
    disp = np.zeros_like(U)
    disp[f] = d_f
    norm = np.linalg.norm(disp, axis=1)
    cap = (CLAMP_FRAC if cap_frac is None else cap_frac) * diag
    over = norm > cap
    disp[over] *= (cap / norm[over])[:, None]
    stats = {"method": "biharmonic", "rings": int(rings), "vertices_unique": int(n), "free_vertex_fraction": float(free.mean()),
             "seed_vertex_fraction": float(seed_u.mean()), "disp_mean_free": float(norm[f].mean()), "disp_p95_free": float(np.quantile(norm[f], 0.95)),
             "disp_max_before_clamp": float(norm.max()), "clamp_mm_equiv": cap, "clamped_fraction": float(over.mean())}
    return V + disp[inverse], stats


def taubin_smooth(mesh, protect_vertex, diag, iters=10, lam=0.5, mu=-0.53, cap_frac=None):
    """Taubin smoothing on position-merged vertices (UV seams stay closed), displacement clamped."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    key = np.round(V / (diag * 1e-6)).astype(np.int64)
    _, uniq_idx, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    U = V[uniq_idx]
    F = inverse[np.asarray(mesh.faces)]
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    n = len(U)
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    X = U.copy()
    for _ in range(iters):
        X = X + lam * (A @ X / deg[:, None] - X)
        X = X + mu * (A @ X / deg[:, None] - X)
    disp = X - U
    prot_u = np.zeros(n, bool)
    np.logical_or.at(prot_u, inverse, protect_vertex)
    disp[prot_u] = 0
    norm = np.linalg.norm(disp, axis=1)
    cap = (CLAMP_FRAC if cap_frac is None else cap_frac) * diag
    over = norm > cap
    disp[over] *= (cap / norm[over])[:, None]
    stats = {"vertices_unique": int(n), "disp_mean": float(norm.mean()), "disp_p95": float(np.quantile(norm, 0.95)),
             "disp_max_before_clamp": float(norm.max()), "clamp_mm_equiv": cap, "clamped_fraction": float(over.mean()),
             "protected_vertex_fraction": float(prot_u.mean())}
    return V + disp[inverse], stats


def vertex_uv_lookup(mesh, mask_img):
    """Boolean per vertex: does its UV land on a True texel of mask_img."""
    uv = np.asarray(mesh.visual.uv)
    h, w = mask_img.shape
    x = np.clip((uv[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(int)
    y = np.clip(((1.0 - uv[:, 1]) % 1.0) * (h - 1), 0, h - 1).astype(int)
    return mask_img[y, x]


def ssim_masked(a, b, mask):
    _, smap = structural_similarity(a, b, data_range=100.0, full=True)
    return float(smap[mask].mean())


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--material", required=True, choices=sorted(PRESETS))
    ap.add_argument("--mode", default="conserved", choices=["conserved", "pristine", "both"])
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--chroma", type=float, default=0.85, metavar="STRENGTH", help="PoC refinish chroma step: damaged texels move to the target colour by at least this fraction (factor = c + (1-c)*d); 0 = pure damage-degree weighting (v1.0/v1.1), PoC default 0.85")
    ap.add_argument("--tone", type=float, default=None, metavar="STRENGTH", help="PoC refinish tone step: renormalise L of non-protected texels so the mean moves to the material L (0.38*healthy + 0.62*preset) with contrast gain <= 1. Default: ceramics 0.8 (PoC), metals 0.3; 0 = keep L exactly (v1.0/v1.1)")
    ap.add_argument("--destain", type=float, default=None, metavar="STRENGTH", help="v1.3: lift dark corrosion blotches toward the local bright-surface envelope. Default: metals 0.85, ceramics 0 (off)")
    ap.add_argument("--chroma-var-keep", type=float, default=None, help="fraction of the a/b deviation kept at full strength (scaled by 1-d). Default ceramics 0.25, metals 0.10")
    ap.add_argument("--no-normal-map", action="store_true", help="v1.4: do not attach the pattern normal map (coherent relief from the original texture) to the GLB")
    ap.add_argument("--wear-mask", default=None, help="v1.5 (8-A semi-automatic): PNG mask of glaze worn through to the body (white). Those texels leave the painted-decoration protection and are restored with full destain")
    ap.add_argument("--crack-mask", default=None, help="v1.5: hand-painted crack/seam mask PNG (white = seam, texture resolution or any size); added to the crack fill without the length/thickness filter. For seams the colour and curvature rules cannot isolate (noisy earthenware)")
    ap.add_argument("--no-uv-seams", action="store_true", help="v1.5: with --remove-cracks, do not treat UV-chart borders as fragment joins (colour harmonisation across charts + groove smoothing)")
    ap.add_argument("--groove-rings", type=int, default=GROOVE_RINGS, help="v1.6: crack/seam vertices + this many rings are re-solved as a biharmonic surface (groove fill). 0 = v1.5 Taubin on the seam vertices only")
    ap.add_argument("--border-band", type=float, default=BORDER_BAND_PX, help="v1.6: UV-chart border band (px at texture resolution) whose low-frequency colour is replaced by the chart interior's. 0 = off")
    ap.add_argument("--stain-gate", default="any", choices=["any", "degree"], help="v2.0: 얼룩 판정을 손상 여부(any, 기본)로 할지 손상 정도(degree, v1.9 동작)로 할지")
    ap.add_argument("--keep-healthy", type=float, default=1.0, help="v2.0: 손상도가 0인 텍셀은 입력 그대로 내보낸다(1.0 = 완전 통과, 0 = v1.9 동작). 손상도 0~DAMAGE_REPORT_D 구간에서 부드럽게 섞인다")
    ap.add_argument("--band-match", type=float, default=None, help="v1.9: cap the local variation of the result, band by band, at what the reference surface has (healthy texels when >= 3 %%, else the original texture). 0 = off. Default: ceramics 1.0, metals 0 - a metal's \"healthy\" texels are bright patina as often as surviving gilding, and using them as the reference tripled the gilt bronze crown's dE")
    ap.add_argument("--band-cap", type=float, default=1.0, help="allowed ratio to the reference band std (1.0 = never exceed the reference)")
    ap.add_argument("--geom-detail", type=float, default=1.0, help="v1.8: use the MESH relief to decide which colour high frequency is real. On a damaged texel with no geometric relief under it the detail is a stain and is dropped, in proportion to this strength. 0 = v1.7 behaviour (colour coherence only)")
    ap.add_argument("--geom-detail-res", type=int, default=GEOM_DETAIL_RES, help="raster size for the mesh relief used by --geom-detail")
    ap.add_argument("--fill-band", type=float, default=FILL_BAND_MM, help="v1.7: width (mm on the surface) of the join / crack band whose pattern is rebuilt by mirroring the neighbouring texture. 0 = off (v1.6 behaviour: TELEA only, band flattened in the normal map)")
    ap.add_argument("--geom-relief", action="store_true", help="v1.7: rasterise the mesh relief (distance to a smoothed copy) even when it is only needed for re-embossing; implied by --reemboss and --geom-cracks")
    ap.add_argument("--geom-cracks", action="store_true", help="v1.7 (opt-in, review the mask): add long wide grooves of the MESH to the crack candidates. On combed earthenware this also selects deep decorative bands, so check mask_geom_crack_*.png before trusting it")
    ap.add_argument("--geom-crack-lo", type=float, default=-0.25, help="seed depth (mm below the local surface) for --geom-cracks")
    ap.add_argument("--geom-crack-hi", type=float, default=-0.10, help="grow depth (mm) for --geom-cracks (hysteresis)")
    ap.add_argument("--reemboss", type=float, default=1.0, help="v1.7: after the groove fill, displace the band vertices by the mirrored mesh relief so the pattern is real geometry (0 = off, normal map only)")
    ap.add_argument("--reemboss-mm", type=float, default=0.6, help="cap of the re-embossed displacement (mm)")
    ap.add_argument("--no-mesh-seams", action="store_true", help="v1.5: with --remove-cracks, do not add mesh grooves (concave curvature) as seam candidates")
    ap.add_argument("--mesh-seam-frac", type=float, default=0.05, help="fraction of the most concave texels used as seam candidates (default 0.05); long thin components survive the crack filter, comb strokes do not")
    ap.add_argument("--no-crack-smooth", action="store_true", help="v1.4: with --remove-cracks, skip the local Taubin smoothing of crack/seam vertices (PoC smooth_crack_geometry)")
    ap.add_argument("--destain-keep", type=float, default=0.0, metavar="SIGMA_PX", help="v2.1: 탈색 리프트를 이 sigma의 국소 평균 기준으로 정해, 그보다 가는 음영(무늬)은 그대로 둔다. 0 = 텍셀 단위 리프트(v1.2~v2.0)")
    ap.add_argument("--destain-geom", type=float, default=0.0, metavar="STRENGTH", help="v2.1: 메시 요철 지지도가 높은 텍셀은 어두움을 음영으로 보고 탈색 리프트를 (1 - STRENGTH*지지도)배로 줄인다. 0 = off")
    ap.add_argument("--destain-size", type=float, default=512, help="envelope window px (blotch scale to remove; larger keeps only coarser shading)")
    ap.add_argument("--normal-strength", type=float, default=1.0, help="노멀맵 요철 세기 배율. 기본 1.0은 95 백분위 기울기를 20도로 맞춘다 (석기처럼 거친 표면은 2~3이 필요)")
    ap.add_argument("--chroma-flatten", type=float, default=0.0, metavar="SIGMA_PX", help="색도를 '주변 평균이 목표색'이 되게 맞추고, 그보다 가는 알갱이 질감은 원본에서 그대로 살린다. 0 = off (완전 단색)")
    ap.add_argument("--chroma-grain", type=float, default=1.0, help="--chroma-flatten이 살릴 알갱이 세기 (1.0 = 원본과 같은 세기)")
    ap.add_argument("--tone-flatten", type=float, default=0.0, metavar="SIGMA_PX", help="이 크기보다 큰 밝기 얼룩을 없앤다: 각 텍셀을 '자기 주변 평균이 목표 밝기가 되도록' 이동. 결·물레자국 등 더 가는 구조는 그대로 남는다. 0 = off")
    ap.add_argument("--tone-grain", type=float, default=1.0, help="--tone-flatten이 살릴 밝기 알갱이 세기 (1.0 = 원본 그대로, 낮추면 창보다 작은 얼룩도 함께 옅어진다)")
    ap.add_argument("--tone-flatten-strength", type=float, default=1.0, help="--tone-flatten 적용 비율 (1.0 = 완전 평탄화)")
    ap.add_argument("--healthy-max-de", type=float, default=None, help="목표색에서 이 ΔE_ab보다 먼 텍셀은 건전부로 인정하지 않는다. 스캔 전체에 색조가 끼어 건전부 검출이 그 색조를 고르는 경우에 쓴다 (외부 기준색이 있을 때만 의미 있음)")
    ap.add_argument("--protect-painted", default="on", choices=["on", "off"], help="도자기의 '채색 장식' 보호 규칙(기준색에서 먼 색은 안료로 보고 보존). 안료가 없는 유물에서는 얼룩을 그대로 얼려버리므로 off")
    ap.add_argument("--protect-dark", default=None, choices=["all", "ceramic", "off"], help="structural near-black protection. Default by material group: ceramics 'all' (PoC adaptive threshold), metals 'off' (dark = corrosion, not decoration)")
    ap.add_argument("--protect-dark-thr", type=float, default=None, help="fixed L* threshold for dark protection instead of the adaptive clamp (PoC protect_dark_thr/2.55; duk006294 used 150 -> 58.8)")
    ap.add_argument("--remove-cracks", action="store_true", help="v1.1: detect long thin dark/bright lines (cracks, fragment seams) and fill them (PoC default off)")
    ap.add_argument("--crack-dark-thr", type=float, default=12.0 * 100 / 255, help="crack candidate: local darkening in L* (PoC crack_dark_thr/2.55; default 4.7, ssu022891 used 6 -> 2.35)")
    ap.add_argument("--crack-min-diag", type=float, default=250, help="crack component min bbox diagonal px (PoC crack_min_diag_px)")
    ap.add_argument("--crack-max-thickness", type=float, default=12.0, help="crack component max mean thickness px (PoC crack_max_thickness_px)")
    ap.add_argument("--alpha", default=None, help="alpha/hole map sidecar (default: <stem>_alpha.jpg next to the model if present)")
    ap.add_argument("--plausible-de", type=float, default=None, help="max dE_ab between the healthy median and the preset for the healthy colour to be trusted (blend); above it the preset is used. Default: preset value (gold/gilt_bronze 25, others 15). PoC had no such gate")
    ap.add_argument("--preset-blend-l", type=float, default=PRESET_BLEND_L, help="v2.0: 프리셋 비중을 밝기(L)에만 따로 적용. 채도는 --preset-blend")
    ap.add_argument("--preset-blend", type=float, default=PRESET_BLEND, help="weight of the preset vs the measured healthy colour when both are usable (PoC 0.62)")
    ap.add_argument("--target-ab", default=None, metavar="A,B", help="목표 색도를 직접 지정 (예: 실물 사진에서 측정한 값). 프리셋의 target_ab를 대체한다")
    ap.add_argument("--pristine-l", type=float, default=None, help="프리셋의 pristine_L을 대체 (톤 재정규화의 기준 밝기)")
    ap.add_argument("--pbr", default=None, metavar="METALLIC,ROUGHNESS", help="출력 GLB의 PBR을 프리셋 대신 이 값으로")
    ap.add_argument("--target", default="auto", choices=["auto", "healthy", "preset"], help="colour target: auto = healthy median if plausible for the material, else preset")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or args.material
    preset = PRESETS[args.material]
    plausible_de = args.plausible_de if args.plausible_de is not None else preset.get("plausible_de", PLAUSIBLE_DE)
    metal = preset["group"] == "metal"
    if args.reemboss > 0 and args.remove_cracks:
        args.geom_relief = True
    var_keep = args.chroma_var_keep if args.chroma_var_keep is not None else CHROMA_VAR_KEEP[preset["group"]]
    if args.protect_dark is None:
        args.protect_dark = "off" if metal else "all"
    if args.band_match is None:
        args.band_match = 0.0 if metal else 1.0     # v1.9: only ceramics have a trustworthy surviving original surface
    if args.tone is None:
        args.tone = 0.3 if metal else 0.8
    if args.destain is None:
        args.destain = 1.0 if metal else 0.7                 # ceramics (v1.5): lift dark worn rims / feet toward the surrounding glaze; protected painting is untouched
    if metal and args.destain_size == 512:
        args.destain_size = 1024                     # blotches on large plain areas (statue backs) exceed 512 px
    t0 = time.time()

    scene = trimesh.load(args.model, force="scene")
    for g in scene.geometry.values():          # OBJ+MTL come in as SimpleMaterial; convert so baseColorTexture exists
        if g.visual.kind == "texture" and not hasattr(g.visual.material, "baseColorTexture"):
            g.visual.material = g.visual.material.to_pbr()
    name, mesh = next((n, g) for n, g in scene.geometry.items() if g.visual.kind == "texture" and g.visual.material.baseColorTexture is not None)
    mat = mesh.visual.material
    rgba = np.asarray(mat.baseColorTexture.convert("RGBA"))
    rgb = rgba[..., :3]
    H, W = rgb.shape[:2]
    lab = skcolor.rgb2lab(rgb / 255.0)

    # ---- 1 condition survey ----
    valid = used_texel_mask(mesh, W, H)
    alpha_path = args.alpha or os.path.join(os.path.dirname(args.model), os.path.splitext(os.path.basename(args.model))[0] + "_alpha.jpg")
    alpha_sidecar = None
    if os.path.exists(alpha_path):
        a_img = Image.open(alpha_path).convert("L")
        if a_img.size != (W, H):
            a_img = a_img.resize((W, H), Image.BILINEAR)
        alpha_sidecar = np.asarray(a_img)
    groove_cand = None
    if args.remove_cracks and not args.no_mesh_seams:
        # seams of glued fragments are grooves in the scan mesh even where the texture shows little contrast (ssu022891)
        groove = curvature_texture(mesh, W, H, concave=True)
        groove_cand = valid & (groove > np.quantile(groove[valid], 1.0 - args.mesh_seam_frac))
        groove_cand = ndi.binary_dilation(groove_cand, iterations=1)
        Image.fromarray((groove * 255).astype(np.uint8)).save(os.path.join(args.out, f"mask_groove_{args.tag or args.material}.png"))
    crack_fill, n_cracks = (detect_cracks(lab, valid, args.crack_dark_thr, args.crack_min_diag, args.crack_max_thickness, groove_cand) if args.remove_cracks else (np.zeros_like(valid), 0))
    if args.crack_mask:
        cm = Image.open(args.crack_mask).convert("L")
        if cm.size != (W, H):
            cm = cm.resize((W, H), Image.NEAREST)
        manual = valid & (np.asarray(cm) > 127)
        crack_fill = crack_fill | ndi.binary_dilation(manual, iterations=3)
        n_cracks += int(ndi.label(manual)[1])
        args.remove_cracks = True                                   # manual seams always get filled, harmonised and smoothed
    mm_px = float(np.sqrt(float(mesh.area) / max(valid.sum(), 1)))                     # mm per texel on the surface
    px_mm = lambda mm: max(1, int(round(mm / mm_px)))
    geom_res = geom_cov = None
    if args.remove_cracks and (args.geom_relief or args.geom_cracks):
        geom_res, geom_cov, _ = mesh_residual(mesh, W, H)                              # v1.7: real relief from the mesh
        geom_res = np.where(geom_cov, geom_res, 0.0).astype(np.float32)
        Image.fromarray(np.clip(geom_res / 1.5 * 127 + 128, 0, 255).astype(np.uint8)).save(os.path.join(args.out, f"relief_mesh_{args.tag or args.material}.png"))
    if args.geom_cracks and geom_res is not None:
        # long wide grooves inside a fragment. Reviewed by eye: on ssu022891 this also selects the deep decorative
        # zigzag bands (they are grooves of the same width and depth), so it stays opt-in and is written out as a mask
        inner = ndi.binary_erosion(valid, iterations=px_mm(4))
        rel = geom_res - masked_blur(geom_res, valid, px_mm(8))
        r_ = max(1, px_mm(1.2) // 2)
        y_, x_ = np.ogrid[-r_:r_ + 1, -r_:r_ + 1]
        wide = ndi.grey_closing(rel, footprint=(x_ * x_ + y_ * y_ <= r_ * r_))         # thin comb strokes closed, wide grooves kept
        lbl_g, n_g = ndi.label(inner & (wide < args.geom_crack_lo))
        ids = np.unique(lbl_g[inner & (wide < args.geom_crack_hi)])
        cc = ndi.binary_closing(np.isin(lbl_g, ids[ids > 0]), iterations=px_mm(1.5))
        lbl_g, n_g = ndi.label(cc)
        objs_g = ndi.find_objects(lbl_g)
        areas_g = np.bincount(lbl_g.ravel())[1:] if n_g else np.zeros(0)
        geom_crack = np.zeros_like(valid)
        for i_, sl_ in enumerate(objs_g):
            dg_ = float(np.hypot(sl_[0].stop - sl_[0].start, sl_[1].stop - sl_[1].start))
            if dg_ * mm_px >= 30 and areas_g[i_] / dg_ * mm_px <= 6:
                geom_crack[sl_] |= lbl_g[sl_] == i_ + 1
        Image.fromarray((geom_crack * 255).astype(np.uint8)).save(os.path.join(args.out, f"mask_geom_crack_{args.tag or args.material}.png"))
        crack_fill = crack_fill | geom_crack
        n_cracks += int(ndi.label(geom_crack)[1])
    if args.remove_cracks and crack_fill.any():
        lab = fill_cracks(lab, valid, crack_fill)
    seam_info = None
    if args.remove_cracks and not args.no_uv_seams:
        lab, seam_dL = harmonize_uv_seams(lab, mesh, valid)
        seam_info = {"mean_abs_dL_across_seams": seam_dL}
        if args.border_band > 0:
            lab, band_info = equalize_chart_borders(lab, valid, args.border_band)      # v1.6
            seam_info["border_band"] = band_info
    # v1.7: the band that lost its pattern (join border + filled cracks) gets the neighbouring pattern mirrored into it
    fill_band = np.zeros_like(valid)
    band_w = None
    mirror_info = None
    if args.remove_cracks and args.fill_band > 0:
        b = px_mm(args.fill_band)
        core = crack_fill & valid if crack_fill.any() else np.zeros_like(valid)        # texels whose pattern is really gone
        if not args.no_uv_seams:
            core = core | (valid & ~ndi.binary_erosion(valid, iterations=px_mm(1.5)))  # chart-border ring = the join itself
        fill_band = ndi.binary_dilation(core, iterations=b) & valid
        dist_core = ndi.distance_transform_edt(~core)
        band_w = np.where(fill_band, np.clip(1.0 - dist_core / max(b, 1), 0, 1), 0.0).astype(np.float32)
        band_w[core] = 1.0                                                             # full replacement in the core, feathered outward
        chart, _ = ndi.label(valid)
        src = valid & ~ndi.binary_dilation(core, iterations=max(1, px_mm(0.5)))        # mirror source = intact texture
        lab, mirror_ok = mirror_fill(lab, src, band_w, chart, sigma_low=px_mm(3.0))
        mirror_info = {"band_mm": args.fill_band, "core_fraction_of_valid": float(core.sum() / max(valid.sum(), 1)),
                       "band_fraction_of_valid": float(fill_band.sum() / max(valid.sum(), 1)), "mirror_source_found": mirror_ok}
        rgb = np.where(valid[..., None], (np.clip(skcolor.lab2rgb(lab), 0, 1) * 255 + 0.5).astype(np.uint8), rgb)
        rgba = np.concatenate([rgb, rgba[..., 3:4]], axis=-1)                          # height map below must see the filled pattern
    healthy, score, ref_ab = healthy_mask(lab, valid, preset)
    healthy_frac = float(healthy.sum() / max(valid.sum(), 1))
    height, _, _ = height_from_colour(rgb, valid, 1.0, 16.0)
    coh = structure_coherence(lab[..., 0], valid)                       # v1.4: line-like detail = decoration
    geom_sup, geom_detail_info = None, None
    if args.geom_detail > 0:
        geom_sup = geom_support(mesh, W, H, mm_px, args.geom_detail_res, geom_res)    # v1.8: real relief, from the mesh
        Image.fromarray((geom_sup * 255).astype(np.uint8)).save(os.path.join(args.out, f"mask_geom_support_{args.tag or args.material}.png"))
    if args.remove_cracks and crack_fill.any():
        coh[ndi.binary_dilation(crack_fill, iterations=8)] = 0.0
    convex = curvature_texture(mesh, W, H) if preset["group"] == "ceramic" else None
    exclude, incised, prot_parts = protection_mask(lab, rgba, valid, healthy, preset, ref_ab, height, args.protect_dark, alpha_sidecar, crack_fill if args.remove_cracks else None, args.protect_dark_thr, convex,
                                                   protect_painted=(args.protect_painted == "on"))
    dark_thr_L = prot_parts.pop("dark_thr_L")
    wear_manual = None
    if args.wear_mask:
        wm = Image.open(args.wear_mask).convert("L")
        if wm.size != (W, H):
            wm = wm.resize((W, H), Image.NEAREST)
        wear_manual = valid & (np.asarray(wm) > 127)
        exclude &= ~wear_manual                                   # worn body is damage, never decoration
        prot_parts["wear"] = prot_parts.get("wear", np.zeros_like(valid)) | wear_manual
    healthy &= ~exclude
    damage = valid & ~healthy & ~exclude

    # Explicit overrides: the reference is a photograph of THIS artefact, which beats any material-wide preset.
    if args.target_ab:
        preset = dict(preset, target_ab=tuple(float(v) for v in args.target_ab.split(",")))
    if args.pristine_l is not None:
        preset = dict(preset, pristine_L=float(args.pristine_l))
    if args.pbr:
        preset = dict(preset, pbr=tuple(float(v) for v in args.pbr.split(",")))
    preset_ab = np.array(preset["target_ab"], dtype=float)
    healthy_ab = np.array([np.median(lab[..., 1][healthy]), np.median(lab[..., 2][healthy])]) if healthy.any() else preset_ab
    healthy_vs_preset = float(np.hypot(*(healthy_ab - preset_ab)))
    if args.target == "healthy":
        target_ab, target_src = healthy_ab, "healthy surface median"
    elif args.target == "auto" and healthy_frac >= MIN_HEALTHY_FRAC and healthy_vs_preset <= plausible_de:
        target_ab = (1 - args.preset_blend) * healthy_ab + args.preset_blend * preset_ab
        target_src = f"blend ab {1-args.preset_blend:.2f}*healthy + {args.preset_blend:.2f}*preset, L {1-args.preset_blend_l:.2f}/{args.preset_blend_l:.2f}"
    else:
        target_ab = preset_ab
        why = (f"healthy fraction {healthy_frac:.1%} < {MIN_HEALTHY_FRAC:.0%}" if healthy_frac < MIN_HEALTHY_FRAC
               else f"healthy colour dE_ab {healthy_vs_preset:.1f} from preset > {plausible_de:.0f} (bright patina, not original surface)")
        target_src = f"preset {args.material} ({why})" if args.target == "auto" else f"preset {args.material} (forced)"
        if args.target == "auto" and healthy_frac >= MIN_HEALTHY_FRAC and healthy_vs_preset > plausible_de:
            healthy = valid & (de_ab(lab, preset_ab) < 10) & ~exclude   # keep only texels that really look like the original material
            healthy_frac = float(healthy.sum() / max(valid.sum(), 1))
            damage = valid & ~healthy & ~exclude
    if args.healthy_max_de is not None:
        # "Healthy" means surviving ORIGINAL surface. The detector picks it from the scan alone (top 15 % of a robust
        # score), so when the whole scan carries a cast it happily elects the cast as healthy - on gubdari_v0 the
        # surviving blue was exactly the healthy mask (10.8 %), and healthy texels get no chroma correction at all
        # (the refinish is multiplied by `damage`). With an external reference colour we can say it outright:
        # a texel further than this from the target is not original surface.
        healthy &= de_ab(lab, target_ab) < args.healthy_max_de
        healthy_frac = float(healthy.sum() / max(valid.sum(), 1))
        damage = valid & ~healthy & ~exclude
    d = np.clip(de_ab(lab, target_ab) / DAMAGE_DE_FULL, 0, 1) * damage
    if wear_manual is not None:
        d = np.where(wear_manual & damage, 1.0, d)
    d[incised & damage] *= INCISED_ATTEN
    damage_frac = float((d >= DAMAGE_REPORT_D).sum() / max(valid.sum(), 1))   # reported / VR layer: meaningfully damaged texels only
    protect_frac = float(exclude.sum() / max(valid.sum(), 1))

    # ---- 2 corrosion removal + 3 refinishing (conserved) ----
    lab_new = lab.copy()
    mean_ab_dmg = np.array([lab[..., 1][damage].mean(), lab[..., 2][damage].mean()]) if damage.any() else target_ab
    f = (args.chroma + (1 - args.chroma) * d) * damage      # PoC chroma_strength floor, damage-degree weighted above it
    # PoC refinish: a' = target + 0.25*(a - mean)  -> colour blotches (green/brown patches) are compressed to 25 %, not kept
    for ch, t_, m_ in ((1, target_ab[0], mean_ab_dmg[0]), (2, target_ab[1], mean_ab_dmg[1])):
        x = lab[..., ch]
        lab_new[..., ch] = (1 - f) * x + f * (t_ + var_keep * (1.0 - d) * (x - m_))   # v1.4: clearly damaged texels converge fully (worn celadon rims stayed beige at 25 %)
    # dark-spot cleaning: small pits much darker than their surroundings, inside the damage mask only
    L_fill = fill_nearest(lab[..., 0], valid)
    L_med = ndi.median_filter(L_fill, size=7)
    spots = damage & ((L_med - lab[..., 0]) > 15)
    lab_spots, n_sp = ndi.label(spots)
    if n_sp:
        sizes = ndi.sum(spots, lab_spots, index=np.arange(1, n_sp + 1))
        small = np.zeros(n_sp + 1, bool)
        small[1:] = sizes <= 30
        spots = small[lab_spots]
        lab_new[..., 0][spots] = L_med[spots]
    spot_frac = float(spots.sum() / max(valid.sum(), 1))

    # PoC refinish, lightness part: mean L of the (non-protected) surface -> material L, contrast compressed (gain <= 1)
    work = valid & ~exclude
    tone_info = {"strength": args.tone}
    if args.tone > 0 and work.any():
        L_h = lab[..., 0][healthy] if healthy.any() else lab[..., 0][work]
        tgt_L = (1 - args.preset_blend_l) * float(L_h.mean()) + args.preset_blend_l * preset["pristine_L"]   # v2.0: 밝기는 별도 비중
        mu, sig = float(lab[..., 0][work].mean()), float(lab[..., 0][work].std()) + 1e-6
        gain = min(float(L_h.std()) / sig * 1.6, 1.0) * 0.75 + 0.25
        L_tone = tgt_L + (lab[..., 0] - mu) * gain
        lab_new[..., 0] = np.where(work, lab[..., 0] * (1 - args.tone) + L_tone * args.tone, lab_new[..., 0])
        tone_info.update(target_L=tgt_L, surface_mean_L_before=mu, gain=gain, surface_mean_L_after=float(lab_new[..., 0][work].mean()))
    destain_info = {"strength": args.destain, "size_px": args.destain_size}
    if args.destain > 0:
        L_lift, env = destain(lab_new[..., 0], valid, d, strength=args.destain, size_px=args.destain_size, keep_sigma=args.destain_keep)
        if args.destain_geom > 0 and geom_sup is not None:
            # v2.1: where the mesh has relief under the texel, darkness is shading, not corrosion - scale the lift down
            L_lift = lab_new[..., 0] + (L_lift - lab_new[..., 0]) * (1.0 - args.destain_geom * geom_sup)
        L_lift = np.where(valid, reattach_detail(L_lift, lab[..., 0], d, detail_preserve=0.8, sigma=1.5, coh=coh,
                                                 support=geom_sup, support_strength=args.geom_detail, gate=args.stain_gate), L_lift)   # coherent (pattern) detail kept, speckle dropped
        destain_info["mean_lift_L_damage"] = float((L_lift - lab_new[..., 0])[damage].mean()) if damage.any() else 0.0
        destain_info.update(keep_sigma=args.destain_keep, geom=args.destain_geom)
        lab_new[..., 0] = np.where(damage, L_lift, lab_new[..., 0])
    if args.chroma_flatten > 0:
        # The refinish drives every damaged texel to ONE colour, so the surface loses the grain a fired body actually
        # has - light and dark specks, a faint drift of hue - and an average of a speckled surface reads as flat khaki
        # mud. Here the target is imposed on the LOCAL MEAN instead: each texel keeps the original's deviation from
        # its own neighbourhood (the grain) while the neighbourhood average becomes the target colour.
        mflat = valid.astype(np.float32)
        for ch, tgt in ((1, target_ab[0]), (2, target_ab[1])):
            orig = lab[..., ch].astype(np.float32) * mflat
            lo = block_blur(orig, args.chroma_flatten) / np.maximum(block_blur(mflat, args.chroma_flatten), 1e-6)
            grain = (lab[..., ch] - lo) * args.chroma_grain
            lab_new[..., ch] = np.where(valid, tgt + grain, lab_new[..., ch])
    if args.tone_flatten > 0:
        # --tone renormalises the MEAN lightness but keeps the local contrast, so a stain that survived as a light or
        # dark patch stays a patch. Here every texel is shifted so that its own local mean (over tone_flatten px)
        # becomes the target lightness: variation slower than that scale disappears, everything finer - the grain,
        # the throwing rings, the relief - is carried through untouched. Normalised convolution, so the atlas
        # background cannot bleed in at the island borders.
        mflat = valid.astype(np.float32)
        Lc = lab_new[..., 0].astype(np.float32) * mflat
        lo = block_blur(Lc, args.tone_flatten) / np.maximum(block_blur(mflat, args.tone_flatten), 1e-6)
        tgt_flat = args.pristine_l if args.pristine_l is not None else preset["pristine_L"]
        w_flat = np.clip(args.tone_flatten_strength, 0, 1)
        # Shifting by (target - local mean) keeps EVERY structure finer than the window, blotch and grain alike. On the
        # blue-grey jan_v0 the surviving patches were 50-200 px, i.e. inside the window, so flattening alone left the
        # surface mottled (L std 5.5). --tone-grain scales that finer-than-window part, exactly as --chroma-grain does
        # for colour: below 1.0 the small blotches fade while the true surface relief, which is far finer, stays.
        flat = tgt_flat + (lab_new[..., 0] - lo) * args.tone_grain
        lab_new[..., 0] = np.where(valid, lab_new[..., 0] * (1 - w_flat) + flat * w_flat, lab_new[..., 0])
        tone_info["flatten_sigma_px"] = args.tone_flatten
        tone_info["flatten_strength"] = w_flat
        tone_info["L_std_after_flatten"] = float(lab_new[..., 0][valid].std())
    band_info = None
    if args.band_match > 0:
        lab_new, band_info = match_band_variation(lab_new, valid, d, lab, strength=args.band_match,
                                                  cap=args.band_cap, healthy=healthy)     # v1.9
    # v2.0: a texel with no damage must come out exactly as it went in. Design section 7 asks for dE <= 1 on the
    # healthy surface; the synthetic test measured 1.21 (gilt bronze) because tone, destain and the band cap all touch
    # it a little. Blend back with the same saturating ramp used for the stain gate, so there is no step at the border.
    passthrough = None
    if args.keep_healthy > 0 and healthy.any():
        # the pass-through must key on the HEALTHY mask, not on the damage degree. The degree is a colour measure, so a
        # faint stain scores low on it: keying on it preserved exactly the mild corrosion that step 1 is meant to clean
        # (celadon dE 2.25 -> 3.40, band ratio 2.10 -> 3.77 in the first try). The healthy mask is the algorithm's own
        # statement that a texel is surviving original surface, which is what "leave it alone" should mean.
        w_h = ndi.gaussian_filter(healthy.astype(np.float32), 3.0) * args.keep_healthy
        w_h = np.clip(w_h, 0, 1)[..., None]
        before = lab_new.copy()
        lab_new = np.where(valid[..., None], lab * w_h + lab_new * (1.0 - w_h), lab_new)
        passthrough = {"strength": args.keep_healthy, "mode": "healthy_mask",
                       "healthy_fraction_of_valid": float(healthy.sum() / max(valid.sum(), 1)),
                       "mean_pullback_dE_ab": float(np.hypot(*(before - lab_new)[..., 1:].transpose(2, 0, 1))[healthy].mean())}
    # gamut: dark texels cannot carry the full target chroma in sRGB -> scale a,b with L (damage texels only)
    a_g, b_g = gamut_scale_ab(lab_new[..., 0], lab_new[..., 1], lab_new[..., 2])
    lab_new[..., 1] = np.where(damage, a_g, lab_new[..., 1])
    lab_new[..., 2] = np.where(damage, b_g, lab_new[..., 2])

    def to_rgba(lab_arr):
        rgb_out = (np.clip(skcolor.lab2rgb(lab_arr), 0, 1) * 255 + 0.5).astype(np.uint8)
        rgb_out[~valid] = rgb[~valid]
        return np.concatenate([rgb_out, rgba[..., 3:4]], axis=-1)

    outputs = {}
    if args.mode in ("conserved", "both"):
        outputs["conserved"] = to_rgba(lab_new)
    pristine_skipped = None
    if args.mode in ("pristine", "both") and healthy_frac < MIN_HEALTHY_FRAC:
        # pristine maps the damaged L distribution onto the healthy one; with < 3 % healthy surface that map is meaningless
        # (flat mustard result on corroded metals), so the pristine output is not produced at all
        pristine_skipped = f"healthy fraction {healthy_frac:.1%} < {MIN_HEALTHY_FRAC:.0%}: no reference distribution"
        print(f"   pristine skipped: {pristine_skipped}")
    if args.mode in ("pristine", "both") and pristine_skipped is None:
        lab_p = lab_new.copy()
        if healthy.any() and damage.any():
            q = np.linspace(0, 1, 501)
            src_q = np.quantile(lab[..., 0][damage], q)
            dst_q = np.quantile(lab[..., 0][healthy], q)
            L_map = np.interp(lab[..., 0], src_q, dst_q)
            lab_p[..., 0] = np.where(damage, lab[..., 0] + d * (L_map - lab[..., 0]), lab_p[..., 0])
            lab_p[..., 0] = np.where(valid, reattach_detail(lab_p[..., 0], lab[..., 0], d, coh=coh, support=geom_sup,
                                                            support_strength=args.geom_detail, gate=args.stain_gate), lab_p[..., 0])   # PoC step 4
        outputs["pristine"] = to_rgba(lab_p)

    # ---- 4 shape stabilisation ----
    diag = float(np.linalg.norm(mesh.bounding_box.extents))
    smooth_stats = None
    if not args.no_smooth:
        prot_v = vertex_uv_lookup(mesh, exclude | incised)
        new_V, smooth_stats = taubin_smooth(mesh, prot_v, diag)
        vol_before = float(mesh.volume) if mesh.is_watertight else None
        mesh.vertices = new_V
        smooth_stats["volume_change_pct"] = (float((mesh.volume - vol_before) / vol_before * 100) if vol_before else None)

    crack_geom = None
    if args.remove_cracks and not args.no_crack_smooth and (crack_fill.any() or not args.no_uv_seams):
        # PoC smooth_crack_geometry: the seam is a groove in the mesh too; smooth only the vertices under the crack band
        crack_v = vertex_uv_lookup(mesh, ndi.binary_dilation(crack_fill, iterations=6))
        if not args.no_uv_seams:
            crack_v |= uv_seam_vertices(mesh)                      # chart borders = fragment joins: smooth the groove too
        if args.groove_rings > 0:
            new_V, crack_geom = fill_grooves(mesh, crack_v, diag, rings=args.groove_rings, cap_frac=CRACK_GEOM_CAP)   # v1.6
        else:
            new_V, crack_geom = taubin_smooth(mesh, ~crack_v, diag, iters=20, cap_frac=CRACK_GEOM_CAP)             # v1.4/1.5
            crack_geom["method"] = "taubin"
        mesh.vertices = new_V
        crack_geom["crack_vertex_fraction"] = float(crack_v.mean())
        if args.reemboss and geom_res is not None and fill_band.any():
            # the biharmonic fill leaves the band smooth; put the neighbouring relief back as real geometry so the
            # pattern survives in the silhouette and under any lighting, not only in the normal map (v1.7)
            chart_r, _ = ndi.label(valid)
            res_m, _ = mirror_fill(geom_res, valid & ~fill_band, band_w, chart_r, sigma_low=px_mm(3.0))
            res_m = np.clip(np.where(fill_band, res_m, 0.0), -args.reemboss_mm, args.reemboss_mm)
            amp_tex = ndi.gaussian_filter(band_w, px_mm(1.0))                          # feathered, so the band edge has no step
            uvv = np.asarray(mesh.visual.uv)
            xs = np.clip((uvv[:, 0] % 1.0) * (W - 1), 0, W - 1)
            ys = np.clip(((1.0 - uvv[:, 1]) % 1.0) * (H - 1), 0, H - 1)
            amp = ndi.map_coordinates(amp_tex, [ys, xs], order=1, mode="nearest")
            dv = ndi.map_coordinates(res_m, [ys, xs], order=1, mode="nearest") * amp * args.reemboss
            nrm_v = np.asarray(mesh.vertex_normals)
            mesh.vertices = np.asarray(mesh.vertices) + nrm_v * dv[:, None]
            crack_geom["reemboss"] = {"strength": args.reemboss, "cap_mm": args.reemboss_mm,
                                      "mean_abs_mm": float(np.abs(dv[amp > 0.05]).mean()) if (amp > 0.05).any() else 0.0}

    # ---- pattern normal map (v1.4): coherent relief of the ORIGINAL texture, so decoration reads as geometry even where
    # the colour texture was flattened by destain / tone. Pits and speckle (incoherent) are damped out.
    normal_img = None
    if not args.no_normal_map:
        h_c = 0.5 + (height - 0.5) * np.clip(coh * 2.0, 0, 1)
        if args.remove_cracks and mirror_info is None:                    # v1.6 behaviour: no mirrored pattern -> flatten
            if crack_fill.any():
                h_c[ndi.binary_dilation(crack_fill, iterations=8)] = 0.5  # filled seams must not come back as grooves in the normal map
            if not args.no_uv_seams:
                h_c[valid & ~ndi.binary_erosion(valid, iterations=max(8, int(args.border_band)))] = 0.5
        g = np.hypot(np.gradient(h_c, axis=1), np.gradient(h_c, axis=0))
        p95 = float(np.percentile(g[valid], 95))
        # Scale the SLOPE, not the angle: tan(20 deg * strength) turns negative past strength 4.5 (the angle passes
        # 90 deg), which silently inverted the relief at 5x and flattened it at 8x.
        k_n = float(np.tan(np.deg2rad(20.0)) * args.normal_strength / max(p95, 1e-9))
        nrm_u8, _, _ = normal_from_height(h_c, k_n)
        nrm_u8 = fill_nearest(nrm_u8, valid)
        normal_img = Image.fromarray(nrm_u8, mode="RGB")
        normal_img.save(os.path.join(args.out, f"normal_pattern_{tag}.png"))
        mat.normalTexture = normal_img

    # ---- 5 record + exports ----
    stem = os.path.splitext(os.path.basename(args.model))[0]
    written = {}
    mat.metallicFactor, mat.roughnessFactor = float(preset["pbr"][0]), float(preset["pbr"][1])   # material preset PBR (design 3절)
    key_ = np.round(mesh.vertices, 6)
    _, inv_ = np.unique(key_, axis=0, return_inverse=True)
    inv_ = inv_.ravel()
    fn_ = np.cross(mesh.vertices[mesh.faces[:, 1]] - mesh.vertices[mesh.faces[:, 0]], mesh.vertices[mesh.faces[:, 2]] - mesh.vertices[mesh.faces[:, 0]])
    acc_ = np.zeros((inv_.max() + 1, 3))
    for k_ in range(3):
        np.add.at(acc_, inv_[mesh.faces[:, k_]], fn_)
    n_ = acc_[inv_]
    mesh.vertex_normals = n_ / np.maximum(np.linalg.norm(n_, axis=1, keepdims=True), 1e-12)   # smooth across UV-seam duplicates
    for mode, tex in outputs.items():
        mat.baseColorTexture = Image.fromarray(tex, "RGBA")
        p = os.path.join(args.out, f"{stem}_restored_{mode}_{tag}.glb")
        scene.export(p)
        Image.fromarray(tex[..., :3]).save(os.path.join(args.out, f"texture_restored_{mode}_{tag}.png"))
        written[mode] = p
    Image.fromarray((np.clip(coh, 0, 1) * 255).astype(np.uint8)).save(os.path.join(args.out, f"mask_coherence_{tag}.png"))
    if convex is not None:
        Image.fromarray((convex * 255).astype(np.uint8)).save(os.path.join(args.out, f"mask_convex_{tag}.png"))
    for k, m in (("valid", valid), ("healthy", healthy), ("protect", exclude), ("incised", incised)):
        Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(args.out, f"mask_{k}_{tag}.png"))
    Image.fromarray((d * 255 + 0.5).astype(np.uint8)).save(os.path.join(args.out, f"mask_damage_{tag}.png"))

    # metrics (design section 7)
    if geom_sup is not None:
        dmg = d >= DAMAGE_REPORT_D
        geom_detail_info = {"strength": args.geom_detail, "raster": int(min(args.geom_detail_res, W)),
                            "support_mean_damage": float(geom_sup[valid & dmg].mean()) if (valid & dmg).any() else 0.0,
                            "support_mean_healthy": float(geom_sup[healthy].mean()) if healthy.any() else 0.0}
    lab_out = skcolor.rgb2lab(outputs.get("conserved", outputs.get("pristine"))[..., :3] / 255.0)
    metrics = {
        "L_ssim_valid": ssim_masked(lab[..., 0], lab_out[..., 0], valid) if "conserved" in outputs else None,
        "healthy_dE_mean": float(np.linalg.norm((lab_out - lab)[healthy], axis=-1).mean()) if healthy.any() else None,
        "damage_dE_ab_to_target_before": float(de_ab(lab, target_ab)[damage].mean()) if damage.any() else None,
        "damage_dE_ab_to_target_after": float(de_ab(lab_out, target_ab)[damage].mean()) if damage.any() else None,
        "healthy_fraction": healthy_frac, "damage_fraction": damage_frac, "protect_fraction": protect_frac,
        "protect_parts_fraction": {k: float(v.sum() / max(valid.sum(), 1)) for k, v in prot_parts.items()},
        "dark_spot_fraction": spot_frac,
        "dark_protect_threshold_L": dark_thr_L, "alpha_sidecar_used": alpha_sidecar is not None,
        "geom_detail": geom_detail_info, "band_match": band_info, "passthrough": passthrough,
        "cracks": {"enabled": bool(args.remove_cracks), "components": int(n_cracks), "fill_fraction": float(crack_fill[valid].mean()) if valid.any() else 0.0},
    }
    record = {"model": args.model, "material": args.material, "preset": preset, "mode": args.mode, "geometry": name,
              "texture_size": [W, H], "valid_texel_fraction": float(valid.mean()),
              "target_ab": target_ab.tolist(), "target_source": target_src, "glaze_dominant_ab": None if ref_ab is None else ref_ab.tolist(),
              "thresholds": {"healthy_quantile": HEALTHY_QUANTILE, "grow_dE": GROW_DE, "min_healthy_frac": MIN_HEALTHY_FRAC,
                             "damage_dE_full": DAMAGE_DE_FULL, "protect_dE": PROTECT_DE, "near_black_L": NEAR_BLACK_L,
                             "incised_atten": INCISED_ATTEN, "clamp_frac": CLAMP_FRAC, "plausible_dE": plausible_de, "damage_report_d": DAMAGE_REPORT_D},
              "healthy_vs_preset_dE_ab": healthy_vs_preset, "target_mode": args.target, "protect_dark": args.protect_dark, "protect_painted": args.protect_painted, "healthy_max_de": args.healthy_max_de, "protect_dark_thr_fixed_L": args.protect_dark_thr, "chroma_strength": args.chroma, "chroma_var_keep": var_keep, "pristine_skipped": pristine_skipped, "normal_map": normal_img is not None, "normal_strength": args.normal_strength, "chroma_flatten": args.chroma_flatten, "tone_flatten": args.tone_flatten, "crack_geometry": crack_geom, "uv_seams": seam_info, "pattern_mirror": mirror_info, "wear_mask": args.wear_mask, "crack_mask": args.crack_mask, "crack_params": {"dark_thr": args.crack_dark_thr, "min_diag": args.crack_min_diag,
              "max_thickness": args.crack_max_thickness, "fill_band": args.fill_band, "groove_rings": args.groove_rings}, "tone": tone_info, "destain": destain_info,
              "metrics": metrics, "smoothing": smooth_stats, "outputs": written,
              "provenance": "ai_inferred - colour restoration hypothesis (rule-based v1); damage map = FR-UI-008 layer",
              "elapsed_s": time.time() - t0}
    with open(os.path.join(args.out, f"restore_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    # ---- figure: masks + before/after renders ----
    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(2, 6)
    panels = [("base colour (before)", rgb), ("healthiness score", np.where(valid, score, np.nan)),
              ("healthy mask", healthy), ("damage degree d", np.where(valid, d, np.nan)),
              ("protect (excluded)", exclude), ("restored texture", outputs.get("conserved", outputs.get("pristine"))[..., :3])]
    for j, (title, img) in enumerate(panels):
        a = fig.add_subplot(gs[0, j])
        if img.ndim == 3:
            a.imshow(img)
        else:
            a.imshow(img, cmap="viridis" if img.dtype != bool else "gray")
        a.set_title(title, fontsize=10)
        a.axis("off")
    before_tex = rgb
    after_tex = outputs.get("conserved", outputs.get("pristine"))[..., :3]
    Rup = rot("x", 90)
    views = [("front", Rup), ("3/4", rot("z", 40) @ Rup), ("back", rot("z", 180) @ Rup)]
    pts_b, nrm_b, col_b = sample_for_render(mesh, before_tex)
    pts_a, nrm_a, col_a = sample_for_render(mesh, after_tex)
    ext = np.ptp(pts_b, axis=0)
    centre = (pts_b.min(0) + pts_b.max(0)) / 2
    lim = 0.56 * max(ext)
    for j, (vname, R) in enumerate(views):
        render_points(fig.add_subplot(gs[1, j]), pts_b, nrm_b, col_b, R, f"BEFORE - {vname}", centre, lim)
        render_points(fig.add_subplot(gs[1, j + 3]), pts_a, nrm_a, col_a, R, f"RESTORED ({args.material}, conserved) - {vname}", centre, lim)
    fig.suptitle(f"Original-colour restoration v1 - {stem} - material={args.material} - target {target_src}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"restore_compare_{tag}.png"), dpi=85)
    plt.close(fig)

    print(f"{stem} [{args.material}] valid {valid.mean():.1%} of atlas | healthy {healthy_frac:.1%} | damage {damage_frac:.1%} | protect {protect_frac:.1%} "
          f"| target ab=({target_ab[0]:.1f},{target_ab[1]:.1f}) from {target_src}")
    print(f"   L-SSIM {metrics['L_ssim_valid']}  damage dE_ab to target {metrics['damage_dE_ab_to_target_before']:.1f} -> {metrics['damage_dE_ab_to_target_after']:.1f}"
          if metrics["damage_dE_ab_to_target_before"] is not None else "   no damage texels")
    if smooth_stats:
        print(f"   smoothing: disp mean {smooth_stats['disp_mean']*1e3:.3f}e-3, p95 {smooth_stats['disp_p95']*1e3:.3f}e-3, cap {smooth_stats['clamp_mm_equiv']*1e3:.3f}e-3 (units of model), clamped {smooth_stats['clamped_fraction']:.1%}")
    print("   wrote", ", ".join(os.path.basename(p) for p in written.values()), f"| {record['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
