"""Compare several 2D->3D outputs against one GT scan and render a single summary figure.

python multi_compare.py --gt GT.obj --out OUTDIR --model TripoSR=path/mesh.obj --model SF3D=path/mesh.glb ...
Each model mesh is aligned to the GT with compare.align (similarity ICP over 24 axis inits),
scored with compare.metrics, exported as <name>_aligned_to_gt.obj, and drawn in the figure
(front / three-quarter / side / top point renders with the model's own colours + vertical cross-section).
"""
import argparse
import json
import os
import time

import numpy as np
import trimesh
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare import load_mesh, sample, align, apply, metrics, nn_dist

RNG = np.random.default_rng(0)


def load_with_colours(path):
    """Load mesh; return (mesh, colour source). Colour source is either
    ("vertex", per-vertex RGB 0..1) or ("texture", (uv per vertex, HxWx3 image 0..1)) or None."""
    m = trimesh.load(path, force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        geoms = list(m.dump())
        m = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
    col = None
    try:
        vis = m.visual
        if vis.kind == "texture":
            img = getattr(vis.material, "baseColorTexture", None) or getattr(vis.material, "image", None)
            if img is not None and vis.uv is not None:
                col = ("texture", (np.asarray(vis.uv, dtype=np.float64), np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0))
            else:
                vis = vis.to_color()
        if col is None and vis.kind == "vertex":
            col = ("vertex", np.asarray(vis.vertex_colors)[:, :3] / 255.0)
    except Exception:  # noqa: BLE001
        col = None
    return m, col


def sample_coloured(m, col, n):
    """Sample surface points; colour each point from vertex colours or by barycentric UV texture lookup."""
    pts, fidx = trimesh.sample.sample_surface(m, n, seed=1)
    pts = np.asarray(pts)
    nrm = m.face_normals[fidx]
    c = None
    if col is not None and col[0] == "vertex":
        c = col[1][m.faces[fidx]].mean(1)
    elif col is not None and col[0] == "texture":
        uv, img = col[1]
        tri = m.triangles[fidx]
        bary = trimesh.triangles.points_to_barycentric(tri, pts)
        uv_p = (bary[:, :, None] * uv[m.faces[fidx]]).sum(1)
        h, w = img.shape[:2]
        x = np.clip((uv_p[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(int)
        y = np.clip(((1.0 - uv_p[:, 1]) % 1.0) * (h - 1), 0, h - 1).astype(int)
        c = img[y, x]
    return pts, nrm, c


def rot(axis, deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


VIEWS = [("front", np.eye(3)), ("3/4", rot("x", 20) @ rot("z", 35)), ("side", rot("z", 90)), ("top", rot("x", -90))]
LIGHT = np.array([0.4, -0.6, 0.7]) / np.linalg.norm([0.4, -0.6, 0.7])


def draw_views(axes_row, pts, nrm, col, label, lim_c, lim):
    for a, (name, R) in zip(axes_row, VIEWS):
        p = pts @ R.T
        nn = nrm @ R.T
        order = np.argsort(p[:, 1])
        shade = np.clip(nn @ LIGHT, 0, 1) * 0.7 + 0.3
        if col is None:
            c = shade[order]
            a.scatter(p[order, 0], p[order, 2], s=0.6, c=c, cmap="gray", vmin=0, vmax=1)
        else:
            c = np.clip(col[order] * (0.6 + 0.4 * shade[order][:, None]), 0, 1)
            a.scatter(p[order, 0], p[order, 2], s=0.6, c=c)
        a.set_aspect("equal")
        a.set_title(f"{label} - {name}", fontsize=10)
        cx = (lim_c @ R.T)
        a.set_xlim(cx[0] - lim, cx[0] + lim)
        a.set_ylim(cx[2] - lim, cx[2] + lim)
        a.set_xticks([])
        a.set_yticks([])


def draw_section(ax, pts, axis_xy, label, colour, lim_z, lim_x):
    m = np.abs(pts[:, 1] - axis_xy[1]) < 4.0
    ax.scatter(pts[m, 0] - axis_xy[0], pts[m, 2], s=1.0, c=colour)
    ax.set_aspect("equal")
    ax.set_title(f"{label} - section", fontsize=10)
    ax.set_xlim(-lim_x, lim_x)
    ax.set_ylim(lim_z[0] - 10, lim_z[1] + 10)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", default=[], help="NAME=path (repeatable)")
    ap.add_argument("--timing", action="append", default=[], help="NAME=seconds wall time (optional, repeatable)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    timings = dict(kv.split("=", 1) for kv in args.timing)

    gt, gt_col = load_with_colours(args.gt)
    gt_pts = sample(gt)
    tree_gt = cKDTree(gt_pts)
    gt_ext = np.ptp(gt_pts, axis=0)
    gt_diag = float(np.linalg.norm(gt_ext))
    axis_gt = gt_pts[:, :2].mean(0)
    centre = np.array([axis_gt[0], axis_gt[1], gt_pts[:, 2].mean()])
    lim = max(gt_ext) * 0.58
    lim_z = (gt_pts[:, 2].min(), gt_pts[:, 2].max())
    gt_pv, gt_nv, _ = sample_coloured(gt, None, 120_000)

    rows = []
    results = {"gt": {"path": args.gt, "extents_mm": [float(x) for x in gt_ext]}, "models": {}}
    for kv in args.model:
        name, path = kv.split("=", 1)
        t0 = time.time()
        m, col = load_with_colours(path)
        pts = sample(m)
        s, R, t, rms, R0, _ = align(pts, gt_pts, gt_diag)
        al = apply(pts, s, R, t)
        met, d_g2t, d_t2g = metrics(al, gt_pts, tree_gt, gt_diag)
        # front/back split about the camera axis (y after alignment, see symmetry_test.py)
        d = nn_dist(al, tree_gt)
        y = al[:, 1] - al[:, 1].mean()
        halves = sorted([float(d[y > 0].mean()), float(d[y <= 0].mean())])
        m_al = m.copy()
        m_al.vertices = apply(np.asarray(m.vertices, dtype=np.float64), s, R, t)
        out_obj = os.path.join(args.out, f"{name}_aligned_to_gt.obj")
        try:
            if col is not None and col[0] == "vertex":
                m_al.visual = trimesh.visual.ColorVisuals(m_al, vertex_colors=(col[1] * 255).astype(np.uint8))
            elif col is not None and col[0] == "texture":
                out_obj = os.path.join(args.out, f"{name}_aligned_to_gt.glb")
            m_al.export(out_obj)
        except Exception as e:  # noqa: BLE001
            print("export failed", name, e)
        pv, nv, cv = sample_coloured(m_al, col, 120_000)
        rows.append((name, pv, nv, cv, al))
        f = list(met["fscore"].values())
        results["models"][name] = {
            "path": path, "vertices": int(len(m.vertices)), "faces": int(len(m.faces)), "watertight": bool(m.is_watertight),
            "has_colour": col is not None, "scale_to_mm": float(s), "icp_rms_mm": rms,
            "tilt_deg": float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))),
            "extents_after_align_mm": [float(x) for x in np.ptp(al, axis=0)],
            "chamfer_L1_mm": met["chamfer_L1_mm"], "accuracy_mean_mm": met["accuracy_mean_mm"], "completeness_mean_mm": met["completeness_mean_mm"],
            "accuracy_median_mm": met["accuracy_median_mm"], "completeness_median_mm": met["completeness_median_mm"],
            "hausdorff_mm": met["hausdorff_mm"], "p95_gen_to_gt_mm": met["p95_gen_to_gt_mm"],
            "fscore_1pct": f[0]["fscore"], "fscore_2pct": f[1]["fscore"], "fscore_5pct": f[2]["fscore"],
            "front_back_mean_err_mm": halves, "wall_time_s": float(timings[name]) if name in timings else None,
            "compare_elapsed_s": time.time() - t0,
        }
        print(f"{name:10s} chamfer={met['chamfer_L1_mm']:6.2f}  F@1%={f[0]['fscore']:.3f}  F@2%={f[1]['fscore']:.3f}  F@5%={f[2]['fscore']:.3f}  "
              f"hausdorff={met['hausdorff_mm']:6.1f}  V={len(m.vertices)}  F={len(m.faces)}  colour={col is not None}")

    # ---------- figure: one row per model + GT ----------
    n = len(rows) + 1
    fig, axes = plt.subplots(n, 5, figsize=(21, 4.3 * n))
    if n == 1:
        axes = axes[None, :]
    draw_views(axes[0, :4], gt_pv, gt_nv, None, "GT scan", centre, lim)
    draw_section(axes[0, 4], gt_pts, axis_gt, "GT scan", "#333", lim_z, lim)
    palette = ["#c33", "#27a", "#2a7", "#a72"]
    for i, (name, pv, nv, cv, al) in enumerate(rows, start=1):
        draw_views(axes[i, :4], pv, nv, cv, name, centre, lim)
        draw_section(axes[i, 4], al, al[:, :2].mean(0), name, palette[(i - 1) % len(palette)], lim_z, lim)
    fig.suptitle("GT scan vs 2D->3D models, all aligned into the GT frame (mm). Right column: vertical cross-section, |y| < 4 mm", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "multi_compare.png"), dpi=90)
    plt.close(fig)

    # ---------- metric bars ----------
    names = list(results["models"].keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.arange(len(names))
    for a, key, title in ((axes[0], "chamfer_L1_mm", "Chamfer L1 [mm]  (lower = better)"),
                          (axes[1], "hausdorff_mm", "Hausdorff (max error) [mm]  (lower = better)")):
        vals = [results["models"][k][key] for k in names]
        b = a.bar(x, vals, color=palette[: len(names)])
        a.set_xticks(x)
        a.set_xticklabels(names)
        a.set_title(title)
        for r_, v in zip(b, vals):
            a.text(r_.get_x() + r_.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    a = axes[2]
    w = 0.27
    for j, (key, lab) in enumerate((("fscore_1pct", "F @ 1% (5.4 mm)"), ("fscore_2pct", "F @ 2% (10.7 mm)"), ("fscore_5pct", "F @ 5% (26.8 mm)"))):
        vals = [results["models"][k][key] for k in names]
        b = a.bar(x + (j - 1) * w, vals, w, label=lab)
        for r_, v in zip(b, vals):
            a.text(r_.get_x() + r_.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    a.set_xticks(x)
    a.set_xticklabels(names)
    a.set_ylim(0, 1.1)
    a.set_title("F-score  (higher = better)")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "multi_compare_metrics.png"), dpi=100)
    plt.close(fig)

    with open(os.path.join(args.out, "multi_compare.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ---------- markdown table ----------
    lines = ["| 모델 | Chamfer | 정확도 중앙값 | 완전성 중앙값 | F@1% | F@2% | F@5% | Hausdorff | 정점/면 | 기울기 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k in names:
        r = results["models"][k]
        lines.append(f"| {k} | {r['chamfer_L1_mm']:.1f} mm | {r['accuracy_median_mm']:.1f} mm | {r['completeness_median_mm']:.1f} mm | {r['fscore_1pct']:.2f} | {r['fscore_2pct']:.2f} | {r['fscore_5pct']:.2f} | {r['hausdorff_mm']:.0f} mm | {r['vertices']:,} / {r['faces']:,} | {r['tilt_deg']:.0f}° |")
    with open(os.path.join(args.out, "multi_compare_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
