"""Does a 'perfect rotational symmetry' rule fix the smeared back side?

Builds surfaces of revolution from silhouettes and scores them against the GT scan:
  A. TripoSR raw (aligned)                      -- baseline from compare.py
  B. TripoSR front silhouette revolved, solid    -- symmetry rule only
  C. TripoSR front silhouette revolved, hollow   -- symmetry + open top + wall thickness rule
  D. GT's own profile revolved, hollow           -- how asymmetric the real pot is (floor for any symmetric model)
Also splits the TripoSR error into camera-facing half vs back half.

python symmetry_test.py --gen-aligned generated_aligned_to_gt.obj --gt GT.obj --out OUTDIR [--wall 15]
"""
import argparse
import json
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare import load_mesh, sample, metrics, nn_dist

RNG = np.random.default_rng(0)


def silhouette_profile(pts, axis_xy, nbins=80, mode="max", zmin=None, zmax=None):
    """Radius per height bin. mode=max -> outer silhouette, mode=min -> inner wall."""
    z = pts[:, 2]
    r = np.linalg.norm(pts[:, :2] - axis_xy, axis=1)
    zmin = z.min() if zmin is None else zmin
    zmax = z.max() if zmax is None else zmax
    edges = np.linspace(zmin, zmax, nbins + 1)
    idx = np.digitize(z, edges) - 1
    zs, rs = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() < 30:
            continue
        rb = r[m]
        # robust extreme: 99th / 1st percentile instead of hard max/min
        rs.append(np.quantile(rb, 0.99) if mode == "max" else np.quantile(rb, 0.01))
        zs.append(0.5 * (edges[b] + edges[b + 1]))
    return np.array(zs), np.array(rs)


def revolve_zup(profile_rz, sections=180):
    """profile_rz: (N,2) array of (r, z) forming a closed or open polyline. Returns z-up mesh."""
    # trimesh.creation.revolve: 2D profile (x=r, y=h) revolved about the 2D Y axis, output already Z-up
    m = trimesh.creation.revolve(profile_rz, sections=sections)
    return m


def solid_from_silhouette(zs, rs, axis_xy):
    prof = [(0.0, zs[0])] + list(zip(rs, zs)) + [(0.0, zs[-1])]
    m = revolve_zup(np.array(prof))
    m.apply_translation([axis_xy[0], axis_xy[1], 0])
    return m


def shell_from_profiles(zs_out, rs_out, zs_in, rs_in, axis_xy, floor_z):
    """Hollow vessel: outer wall up, across the rim, inner wall down, closed at the inner floor."""
    outer = list(zip(rs_out, zs_out))                       # bottom -> rim
    inner = list(zip(rs_in[::-1], zs_in[::-1]))             # rim -> inner floor
    prof = [(0.0, zs_out[0])] + outer + inner + [(0.0, floor_z)]
    m = revolve_zup(np.array(prof))
    m.apply_translation([axis_xy[0], axis_xy[1], 0])
    return m


def score(name, mesh, gt_pts, tree_gt, gt_diag):
    pts = sample(mesh)
    m, d_g2t, d_t2g = metrics(pts, gt_pts, tree_gt, gt_diag)
    m = {k: v for k, v in m.items()}
    m["fscore_2pct"] = list(m["fscore"].values())[1]["fscore"]
    m["fscore_1pct"] = list(m["fscore"].values())[0]["fscore"]
    print(f"{name:45s} chamfer={m['chamfer_L1_mm']:6.2f} mm  F@2%={m['fscore_2pct']:.3f}  hausdorff={m['hausdorff_mm']:6.1f}")
    return m, pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-aligned", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wall", type=float, default=15.0, help="assumed wall thickness for rule C [mm]")
    ap.add_argument("--front-sign", type=int, default=0, help="+1/-1: which y half faces the camera; 0 = auto (lower error half)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gt = load_mesh(args.gt)
    gen = load_mesh(args.gen_aligned)
    gt_pts = sample(gt)
    gen_pts = sample(gen)
    tree_gt = cKDTree(gt_pts)
    gt_diag = float(np.linalg.norm(np.ptp(gt_pts, axis=0)))
    axis_gt = gt_pts[:, :2].mean(0)
    axis_gen = gen_pts[:, :2].mean(0)
    report = {"wall_mm": args.wall}

    # ---- A. raw TripoSR, split front/back -------------------------------------------------
    mA, _ = score("A  TripoSR raw (aligned)", gen, gt_pts, tree_gt, gt_diag)
    d = nn_dist(gen_pts, tree_gt)
    y = gen_pts[:, 1] - axis_gen[1]
    halves = {"+y": d[y > 0], "-y": d[y <= 0]}
    front = args.front_sign if args.front_sign else (1 if halves["+y"].mean() < halves["-y"].mean() else -1)
    fkey, bkey = ("+y", "-y") if front > 0 else ("-y", "+y")
    report["A_raw"] = {"chamfer_L1_mm": mA["chamfer_L1_mm"], "fscore_1pct": mA["fscore_1pct"], "fscore_2pct": mA["fscore_2pct"], "hausdorff_mm": mA["hausdorff_mm"],
                       "front_half": fkey, "front_mean_err_mm": float(halves[fkey].mean()), "front_median_err_mm": float(np.median(halves[fkey])),
                       "back_mean_err_mm": float(halves[bkey].mean()), "back_median_err_mm": float(np.median(halves[bkey]))}
    print(f"   front half ({fkey}) mean err {halves[fkey].mean():.2f} mm | back half ({bkey}) mean err {halves[bkey].mean():.2f} mm")

    # ---- silhouettes ------------------------------------------------------------------------
    # camera-facing silhouette of TripoSR: use the front half only, outer radius per height
    front_pts = gen_pts[(y > 0) if front > 0 else (y <= 0)]
    zs_g, rs_g = silhouette_profile(front_pts, axis_gen, nbins=80)
    zs_t_out, rs_t_out = silhouette_profile(gt_pts, axis_gt, nbins=80, mode="max")
    zs_t_in, rs_t_in = silhouette_profile(gt_pts, axis_gt, nbins=80, mode="min")
    gt_floor_z = float(np.quantile(gt_pts[:, 2], 0.0005)) + args.wall  # inner floor ~ wall above the tip

    # ---- B. TripoSR silhouette revolved, solid ------------------------------------------------
    mB_mesh = solid_from_silhouette(zs_g, rs_g, axis_gen)
    mB, _ = score("B  symmetry rule: TripoSR silhouette, solid", mB_mesh, gt_pts, tree_gt, gt_diag)

    # ---- C. TripoSR silhouette revolved, hollow shell with assumed wall ----------------------
    rs_in_rule = np.clip(rs_g - args.wall, 0.5, None)
    keep = rs_in_rule > 2.0
    floor_z = float(zs_g[0] + args.wall)
    mC_mesh = shell_from_profiles(zs_g, rs_g, zs_g[keep], rs_in_rule[keep], axis_gen, floor_z)
    mC, _ = score(f"C  symmetry + open top + {args.wall:.0f} mm wall", mC_mesh, gt_pts, tree_gt, gt_diag)

    # ---- D. GT itself made perfectly symmetric ---------------------------------------------
    keep_in = rs_t_in > 2.0
    mD_mesh = shell_from_profiles(zs_t_out, rs_t_out, zs_t_in[keep_in], rs_t_in[keep_in], axis_gt, gt_floor_z)
    mD, _ = score("D  GT revolved (real pot's own asymmetry)", mD_mesh, gt_pts, tree_gt, gt_diag)

    for k, m in (("B_sym_solid", mB), ("C_sym_shell", mC), ("D_gt_symmetrised", mD)):
        report[k] = {"chamfer_L1_mm": m["chamfer_L1_mm"], "fscore_1pct": m["fscore_1pct"], "fscore_2pct": m["fscore_2pct"], "hausdorff_mm": m["hausdorff_mm"],
                     "accuracy_mean_mm": m["accuracy_mean_mm"], "completeness_mean_mm": m["completeness_mean_mm"]}
    mB_mesh.export(os.path.join(args.out, "rule_B_symmetric_solid.obj"))
    mC_mesh.export(os.path.join(args.out, "rule_C_symmetric_shell.obj"))

    # ---- figure ------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
    a = axes[0]
    a.plot(rs_t_out, zs_t_out, "k", label="GT outer wall")
    a.plot(-rs_t_out, zs_t_out, "k")
    a.plot(rs_t_in, zs_t_in, "k--", label="GT inner wall")
    a.plot(-rs_t_in, zs_t_in, "k--")
    a.plot(rs_g, zs_g, "r", label=f"TripoSR front silhouette ({fkey} half)")
    a.plot(-rs_g, zs_g, "r")
    a.plot(rs_in_rule[keep], zs_g[keep], "r:", label=f"rule C inner wall (outer - {args.wall:.0f} mm)")
    a.plot(-rs_in_rule[keep], zs_g[keep], "r:")
    a.set_aspect("equal")
    a.set_xlabel("radius [mm]")
    a.set_ylabel("height [mm]")
    a.set_title("Profiles used for the symmetric models")
    a.legend(fontsize=8, loc="lower center")

    a = axes[1]
    sub = gen_pts[RNG.choice(len(gen_pts), 40000, replace=False)]
    dsub = nn_dist(sub, tree_gt)
    sc = a.scatter(sub[:, 1] - axis_gen[1], sub[:, 2], s=0.5, c=dsub, cmap="viridis", vmin=0, vmax=40)
    a.axvline(0, color="w", lw=0.8)
    a.text(-120, 395, f"{'front' if front < 0 else 'back'}", color="k")
    a.text(90, 395, f"{'front' if front > 0 else 'back'}", color="k")
    a.set_aspect("equal")
    a.set_xlabel("y [mm]  (camera looks along y)")
    a.set_ylabel("z [mm]")
    a.set_title(f"TripoSR error, side view: front mean {halves[fkey].mean():.1f} mm vs back {halves[bkey].mean():.1f} mm")
    fig.colorbar(sc, ax=a, shrink=0.7, label="distance to GT [mm]")

    a = axes[2]
    names = ["A TripoSR\nraw", "B symmetric\nsolid", f"C symmetric\nshell {args.wall:.0f}mm", "D GT\nsymmetrised"]
    ch = [mA["chamfer_L1_mm"], mB["chamfer_L1_mm"], mC["chamfer_L1_mm"], mD["chamfer_L1_mm"]]
    f2 = [mA["fscore_2pct"], mB["fscore_2pct"], mC["fscore_2pct"], mD["fscore_2pct"]]
    x = np.arange(4)
    b1 = a.bar(x - 0.2, ch, 0.4, color="#c33", label="Chamfer L1 [mm] (lower is better)")
    a.set_ylabel("Chamfer [mm]")
    a2 = a.twinx()
    b2 = a2.bar(x + 0.2, f2, 0.4, color="#36c", label="F-score @ 2% (higher is better)")
    a2.set_ylim(0, 1.05)
    a2.set_ylabel("F-score @ 10.7 mm")
    a.set_xticks(x)
    a.set_xticklabels(names, fontsize=9)
    for r_, v in zip(b1, ch):
        a.text(r_.get_x() + r_.get_width() / 2, v + 0.2, f"{v:.1f}", ha="center", fontsize=9)
    for r_, v in zip(b2, f2):
        a2.text(r_.get_x() + r_.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color="#36c")
    a.set_title("Does a symmetry rule help?")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "symmetry_test.png"), dpi=110)
    with open(os.path.join(args.out, "symmetry_test.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
