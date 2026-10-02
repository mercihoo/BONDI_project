"""Compare a generated 2D->3D mesh against a ground-truth scan.

Usage: python compare.py --gen GEN.obj --gt GT.obj [--gt-extra a.ply b.stl] --out OUTDIR
All metrics are reported in the GT's native units (assumed mm) after a
similarity (rotation + translation + uniform scale) alignment of GEN onto GT.
"""
import argparse
import itertools
import json
import os
import time

import numpy as np
import trimesh
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_SAMPLE = 200_000
RNG = np.random.default_rng(0)


def load_mesh(path):
    m = trimesh.load(path, force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.dump()])
    return m


def mesh_stats(m, name):
    ext = m.bounding_box.extents
    return {
        "name": name,
        "vertices": int(len(m.vertices)),
        "faces": int(len(m.faces)),
        "bbox_extents": [float(x) for x in ext],
        "bbox_diagonal": float(np.linalg.norm(ext)),
        "watertight": bool(m.is_watertight),
        "surface_area": float(m.area),
    }


def sample(m, n=N_SAMPLE):
    pts, _ = trimesh.sample.sample_surface(m, n, seed=0)
    return np.asarray(pts, dtype=np.float64)


def axis_rotations():
    """All 24 proper rotations that permute/flip axes."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i, p in enumerate(perm):
                R[i, p] = signs[i]
            if np.isclose(np.linalg.det(R), 1.0):
                mats.append(R)
    return mats


def nn_dist(a, tree_b):
    d, _ = tree_b.query(a, k=1, workers=-1)
    return d


def umeyama(src, dst, with_scale=True):
    """Similarity transform (s, R, t) minimising ||s R src + t - dst||."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = (D * np.diag(S)).sum() / var_s if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def icp_similarity(src, dst, tree_dst, init_s, init_R, init_t, iters=60, trim=0.9, tol=1e-6, s_bounds=None):
    """Trimmed similarity ICP. Returns (s, R, t, rms). s_bounds=(lo, hi) clamps the scale so a
    partial / flat mesh cannot shrink onto a patch of the GT surface (degenerate free-scale solution)."""
    s, R, t = init_s, init_R.copy(), init_t.copy()
    prev = np.inf
    rms = np.inf
    for _ in range(iters):
        cur = (s * (R @ src.T)).T + t
        d, idx = tree_dst.query(cur, k=1, workers=-1)
        keep = d <= np.quantile(d, trim)  # drop worst fraction (outliers / missing parts)
        s, R, t = umeyama(src[keep], dst[idx[keep]], with_scale=True)
        if s_bounds is not None and not (s_bounds[0] <= s <= s_bounds[1]):
            s_c = float(np.clip(s, *s_bounds))
            s = s_c
            _, R, t = umeyama(src[keep] * s, dst[idx[keep]], with_scale=False)  # rigid re-fit at clamped scale
        rms = float(np.sqrt((d[keep] ** 2).mean()))
        if abs(prev - rms) < tol:
            break
        prev = rms
    return s, R, t, rms


def align(gen_pts, gt_pts, gt_diag, scale_tol=0.2):
    """Try 24 axis-aligned inits, run similarity ICP from each, keep best.
    Scale is initialised from the bbox-diagonal ratio and clamped to +-scale_tol around it."""
    tree_gt = cKDTree(gt_pts)
    g_c = gen_pts.mean(0)
    t_c = gt_pts.mean(0)
    g_diag = np.linalg.norm(np.ptp(gen_pts, axis=0))
    s0 = gt_diag / g_diag
    s_bounds = (s0 * (1 - scale_tol), s0 * (1 + scale_tol))
    sub = gen_pts[RNG.choice(len(gen_pts), 20_000, replace=False)]
    best = None
    for R0 in axis_rotations():
        t0 = t_c - s0 * R0 @ g_c
        s, R, t, rms = icp_similarity(sub, gt_pts, tree_gt, s0, R0, t0, iters=40, s_bounds=s_bounds)
        if best is None or rms < best[3]:
            best = (s, R, t, rms, R0)
    s, R, t, rms, R0 = best
    s, R, t, rms = icp_similarity(gen_pts[RNG.choice(len(gen_pts), 60_000, replace=False)],
                                  gt_pts, tree_gt, s, R, t, iters=80, trim=0.95, s_bounds=s_bounds)
    return s, R, t, rms, R0, tree_gt


def apply(pts, s, R, t):
    return (s * (R @ pts.T)).T + t


def metrics(gen_al, gt_pts, tree_gt, gt_diag):
    tree_gen = cKDTree(gen_al)
    d_g2t = nn_dist(gen_al, tree_gt)   # accuracy: gen -> gt
    d_t2g = nn_dist(gt_pts, tree_gen)  # completeness: gt -> gen
    out = {
        "chamfer_L1_mm": float((d_g2t.mean() + d_t2g.mean()) / 2),
        "chamfer_L2_mm2": float(((d_g2t ** 2).mean() + (d_t2g ** 2).mean()) / 2),
        "accuracy_mean_mm": float(d_g2t.mean()),
        "completeness_mean_mm": float(d_t2g.mean()),
        "accuracy_median_mm": float(np.median(d_g2t)),
        "completeness_median_mm": float(np.median(d_t2g)),
        "hausdorff_mm": float(max(d_g2t.max(), d_t2g.max())),
        "p95_gen_to_gt_mm": float(np.quantile(d_g2t, 0.95)),
        "p95_gt_to_gen_mm": float(np.quantile(d_t2g, 0.95)),
        "fscore": {},
    }
    for frac in (0.01, 0.02, 0.05):
        tau = frac * gt_diag
        p = float((d_g2t < tau).mean())
        r = float((d_t2g < tau).mean())
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        out["fscore"][f"tau_{int(frac * 100)}pct_{tau:.1f}mm"] = {"precision": p, "recall": r, "fscore": f}
    return out, d_g2t, d_t2g


def profile(pts, up_axis, nbins=60):
    """Axisymmetric profile: median radius per height bin about the mean axis."""
    h = pts[:, up_axis]
    others = [i for i in range(3) if i != up_axis]
    c = pts[:, others].mean(0)
    r = np.linalg.norm(pts[:, others] - c, axis=1)
    edges = np.linspace(h.min(), h.max(), nbins + 1)
    idx = np.digitize(h, edges) - 1
    hs, rs = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() > 20:
            hs.append(0.5 * (edges[b] + edges[b + 1]))
            rs.append(np.median(r[m]))
    return np.array(hs), np.array(rs)


def plot_views(gt_pts, gen_al, d_g2t, gt_ext, up_axis, out, gen_label):
    others = [i for i in range(3) if i != up_axis]
    views = [("front", others[0], up_axis), ("side", others[1], up_axis), ("top", others[0], others[1])]
    sub_gt = gt_pts[RNG.choice(len(gt_pts), 40_000, replace=False)]
    sel = RNG.choice(len(gen_al), 40_000, replace=False)
    sub_gen, sub_d = gen_al[sel], d_g2t[sel]
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    vmax = np.quantile(d_g2t, 0.98)
    sc = None
    for row, (name, ax_x, ax_y) in enumerate(views):
        a = axes[row, 0]
        a.scatter(sub_gt[:, ax_x], sub_gt[:, ax_y], s=0.3, c="#555")
        a.set_title(f"GT scan - {name}")
        a = axes[row, 1]
        a.scatter(sub_gen[:, ax_x], sub_gen[:, ax_y], s=0.3, c="#c33")
        a.set_title(f"{gen_label} (aligned) - {name}")
        a = axes[row, 2]
        sc = a.scatter(sub_gen[:, ax_x], sub_gen[:, ax_y], s=0.3, c=sub_d, cmap="viridis", vmin=0, vmax=vmax)
        a.set_title(f"Error gen->GT [mm] - {name}")
        for a in axes[row]:
            a.set_aspect("equal")
            a.set_xlabel("xyz"[ax_x] + " [mm]")
            a.set_ylabel("xyz"[ax_y] + " [mm]")
            lim = max(gt_ext) * 0.6
            cx = gt_pts[:, ax_x].mean()
            cy = gt_pts[:, ax_y].mean()
            a.set_xlim(cx - lim, cx + lim)
            a.set_ylim(cy - lim, cy + lim)
    fig.colorbar(sc, ax=axes[:, 2], shrink=0.6, label="distance to GT [mm]")
    fig.suptitle(f"GT scan vs {gen_label} (similarity-aligned, GT units = mm)")
    fig.savefig(os.path.join(out, "views.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_profile(gt_pts, gen_al, up_axis, d_g2t, d_t2g, out, gen_label):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    others = [i for i in range(3) if i != up_axis]
    sub_gt = gt_pts[RNG.choice(len(gt_pts), 30_000, replace=False)]
    sub_gen = gen_al[RNG.choice(len(gen_al), 30_000, replace=False)]
    a = axes[0]
    a.scatter(sub_gt[:, others[0]], sub_gt[:, up_axis], s=0.3, c="#555", label="GT scan")
    a.scatter(sub_gen[:, others[0]], sub_gen[:, up_axis], s=0.3, c="#c33", alpha=0.5, label=gen_label)
    a.set_aspect("equal")
    a.legend(markerscale=20)
    a.set_title("Overlay (front)")
    a.set_xlabel("mm")
    a.set_ylabel("mm")
    a = axes[1]
    for pts, col, lab in ((gt_pts, "#555", "GT scan"), (gen_al, "#c33", gen_label)):
        hs, rs = profile(pts, up_axis)
        a.plot(rs, hs, color=col, label=lab)
        a.plot(-rs, hs, color=col)
    a.set_aspect("equal")
    a.legend()
    a.set_title("Axisymmetric profile (median radius vs height)")
    a.set_xlabel("radius [mm]")
    a.set_ylabel("height [mm]")
    a = axes[2]
    bins = np.linspace(0, max(np.quantile(d_g2t, 0.99), np.quantile(d_t2g, 0.99)), 60)
    a.hist(d_g2t, bins=bins, alpha=0.6, label="gen -> GT (accuracy)")
    a.hist(d_t2g, bins=bins, alpha=0.6, label="GT -> gen (completeness)")
    a.set_xlabel("nearest-neighbour distance [mm]")
    a.set_ylabel("count")
    a.legend()
    a.set_title("Distance distributions")
    fig.savefig(os.path.join(out, "overlay_profile_hist.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--gt-extra", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen-name", default="generated")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    gen = load_mesh(args.gen)
    gt = load_mesh(args.gt)
    report = {"gen_mesh": mesh_stats(gen, args.gen), "gt_mesh": mesh_stats(gt, args.gt), "gt_extra": []}
    gt_pts = sample(gt)
    gen_pts = sample(gen)
    gt_ext = np.ptp(gt_pts, axis=0)
    gt_diag = float(np.linalg.norm(gt_ext))
    up_axis = int(np.argmax(gt_ext))  # pottery: tallest extent is the vertical axis
    report["gt_up_axis"] = "xyz"[up_axis]

    # noise floor: how far are the GT's sibling files (PLY scan / STL) from the OBJ?
    tree_gt = cKDTree(gt_pts)
    for p in args.gt_extra:
        try:
            m = load_mesh(p)
            pts = sample(m, 100_000)
            d1 = nn_dist(pts, tree_gt)
            d2 = nn_dist(gt_pts, cKDTree(pts))
            st = mesh_stats(m, p)
            st["chamfer_L1_vs_gt_mm"] = float((d1.mean() + d2.mean()) / 2)
            st["hausdorff_vs_gt_mm"] = float(max(d1.max(), d2.max()))
            report["gt_extra"].append(st)
        except Exception as e:  # noqa: BLE001
            report["gt_extra"].append({"name": p, "error": str(e)})

    s, R, t, rms, R0, tree_gt = align(gen_pts, gt_pts, gt_diag)
    gen_al = apply(gen_pts, s, R, t)
    report["alignment"] = {
        "scale_gen_to_mm": float(s),
        "icp_rms_mm": rms,
        "init_axis_rotation": R0.tolist(),
        "rotation": R.tolist(),
        "translation": t.tolist(),
        "gen_extents_after_align_mm": [float(x) for x in np.ptp(gen_al, axis=0)],
        "gt_extents_mm": [float(x) for x in gt_ext],
    }
    m, d_g2t, d_t2g = metrics(gen_al, gt_pts, tree_gt, gt_diag)
    report["metrics_mm"] = m
    try:
        pitch = gt_diag / 64

        def occ(pts):
            v = np.floor((pts - gt_pts.min(0)) / pitch).astype(int)
            return set(map(tuple, v))

        a, b = occ(gen_al), occ(gt_pts)
        report["metrics_mm"]["surface_voxel_iou_64"] = len(a & b) / len(a | b)
    except Exception as e:  # noqa: BLE001
        report["metrics_mm"]["surface_voxel_iou_64"] = f"error: {e}"

    gen_al_mesh = gen.copy()
    gen_al_mesh.vertices = apply(np.asarray(gen.vertices, dtype=np.float64), s, R, t)
    gen_al_mesh.export(os.path.join(args.out, "generated_aligned_to_gt.obj"))

    plot_views(gt_pts, gen_al, d_g2t, gt_ext, up_axis, args.out, args.gen_name)
    plot_overlay_profile(gt_pts, gen_al, up_axis, d_g2t, d_t2g, args.out, args.gen_name)
    report["elapsed_s"] = time.time() - t0
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
