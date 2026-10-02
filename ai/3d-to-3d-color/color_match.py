"""Match a generated 3D model's base-colour texture to the colours of its 2D input photo.

Pipeline step: 2D -> 3D(파손) -> **3D(사진과 색 맞춤)** -> 3D(복원) -> 3D(제작 당시 색).

What it does
  1. photo  : remove the background (rembg u2net), keep the artefact pixels only.
  2. model  : sample the mesh surface (area-weighted) and read the texels those points use,
              so statistics reflect what is actually visible, not the whole atlas.
  3. transfer in CIE Lab:
       reinhard  - per-channel mean/std match (Reinhard et al. 2001), default
       hist      - per-channel quantile (histogram) match, stronger
       chroma    - match a/b fully, shift only the mean of L (keeps the texture's own shading)
       hue       - shift only the a/b means, leave L untouched. For metals (metallic=1) the base colour
                   is a reflectance, and the photo's lightness is mostly lighting/reflection, so L must not be copied
  4. write a new GLB with the recoloured texture (geometry, UVs, other textures unchanged),
     a before/after/photo comparison figure, and a JSON record of the parameters (FR-AI-005).

Usage
  python color_match.py --photo photo.jpg --model model.glb --out out/ [--method reinhard|hist|chroma] [--strength 1.0]
"""
import argparse
import json
import os
import time

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import binary_erosion
from skimage import color as skcolor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------- photo
def photo_foreground(path, erode_px=6, bg_reject=12.0):
    """Return (rgb uint8 HxWx3, mask bool HxW) for the artefact pixels of the photo.

    bg_reject: drop foreground pixels whose colour is within this Lab distance of the photo's
    background colour (median of the non-object area). Openwork objects (crowns, handles, cracks)
    let the backdrop show through holes that the segmenter keeps; without this the backdrop colour
    (often black or grey) pollutes the target statistics."""
    from rembg import remove, new_session
    im = Image.open(path).convert("RGB")
    rgb = np.asarray(im)
    mask_im = remove(im, session=new_session("u2net"), only_mask=True)
    mask = np.asarray(mask_im) > 128
    if erode_px > 0:
        mask = binary_erosion(mask, iterations=erode_px)
    if bg_reject > 0 and (~mask).sum() > 1000:
        lab = skcolor.rgb2lab(rgb / 255.0)
        bg = np.median(lab[~mask].reshape(-1, 3)[::50], axis=0)
        near_bg = np.linalg.norm(lab - bg, axis=-1) < bg_reject
        dropped = int((mask & near_bg).sum())
        mask = mask & ~near_bg
        print(f"photo: background Lab ~ ({bg[0]:.0f}, {bg[1]:.0f}, {bg[2]:.0f}); dropped {dropped:,} foreground px within dE<{bg_reject} of it")
    return rgb, mask


# ----------------------------------------------------------------------------- model
def load_model(path):
    scene = trimesh.load(path, force="scene")
    meshes = [(name, g) for name, g in scene.geometry.items() if g.visual.kind == "texture" and g.visual.material.baseColorTexture is not None]
    if not meshes:
        raise SystemExit("no textured geometry with a baseColorTexture found in " + path)
    return scene, meshes


def surface_texel_colours(mesh, tex_rgb, n=2_000_000):
    """Colours of area-weighted surface samples, read from the base-colour texture via UVs."""
    pts, fidx = trimesh.sample.sample_surface(mesh, n, seed=0)
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[fidx], np.asarray(pts))
    uv = (bary[:, :, None] * np.asarray(mesh.visual.uv)[mesh.faces[fidx]]).sum(1)
    h, w = tex_rgb.shape[:2]
    x = np.clip((uv[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(int)
    y = np.clip(((1.0 - uv[:, 1]) % 1.0) * (h - 1), 0, h - 1).astype(int)
    return tex_rgb[y, x]


# ----------------------------------------------------------------------------- colour transfer
def robust_stats(lab, lo=1, hi=99):
    out = []
    for c in range(3):
        v = lab[:, c]
        a, b = np.percentile(v, [lo, hi])
        v = v[(v >= a) & (v <= b)]
        out.append((float(v.mean()), float(v.std() + 1e-6)))
    return out  # [(mean, std) for L, a, b]


def to_lab(rgb_u8):
    return skcolor.rgb2lab(rgb_u8.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)


def from_lab(lab):
    rgb = skcolor.lab2rgb(lab.reshape(-1, 1, 3)).reshape(-1, 3)
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def transfer_reinhard(lab, src_stats, tgt_stats, channels=(0, 1, 2), l_std=True):
    out = lab.copy()
    for c in channels:
        ms, ss = src_stats[c]
        mt, st = tgt_stats[c]
        if c == 0 and not l_std:
            out[:, c] = lab[:, c] - ms + mt
        else:
            out[:, c] = (lab[:, c] - ms) / ss * st + mt
    return out


def transfer_hist(lab, src_lab, tgt_lab, channels=(0, 1, 2), nq=1001):
    out = lab.copy()
    q = np.linspace(0, 1, nq)
    for c in channels:
        sq = np.quantile(src_lab[:, c], q)
        tq = np.quantile(tgt_lab[:, c], q)
        out[:, c] = np.interp(lab[:, c], sq, tq)
    return out


# ----------------------------------------------------------------------------- render (point splat, no GPU)
def rot(axis, deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


LIGHT = np.array([-0.3, -0.7, 0.65]) / np.linalg.norm([-0.3, -0.7, 0.65])


def render_points(ax, pts, nrm, col, R, title, lim_c, lim, s=0.9):
    """Orthographic view: camera on -y looking +y, z up, x to the right (proper, not mirrored)."""
    p = pts @ R.T
    nn = nrm @ R.T
    order = np.argsort(-p[:, 1])            # far first, near last
    shade = np.clip(nn @ LIGHT, 0, 1) * 0.55 + 0.45
    c = np.clip(col[order] * shade[order][:, None], 0, 1)
    ax.scatter(p[order, 0], p[order, 2], s=s, c=c, linewidths=0)
    ax.set_aspect("equal")
    cc = lim_c @ R.T
    ax.set_xlim(cc[0] - lim, cc[0] + lim)
    ax.set_ylim(cc[2] - lim, cc[2] + lim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def sample_for_render(mesh, tex_rgb, n=160_000):
    pts, fidx = trimesh.sample.sample_surface(mesh, n, seed=1)
    pts = np.asarray(pts)
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[fidx], pts)
    uv = (bary[:, :, None] * np.asarray(mesh.visual.uv)[mesh.faces[fidx]]).sum(1)
    h, w = tex_rgb.shape[:2]
    x = np.clip((uv[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(int)
    y = np.clip(((1.0 - uv[:, 1]) % 1.0) * (h - 1), 0, h - 1).astype(int)
    return pts, mesh.face_normals[fidx], tex_rgb[y, x] / 255.0


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="reinhard", choices=["reinhard", "hist", "chroma", "hue"])
    ap.add_argument("--strength", type=float, default=1.0, help="0..1 blend between original and matched colours")
    ap.add_argument("--erode", type=int, default=6, help="erode photo mask by N px to avoid background bleed")
    ap.add_argument("--bg-reject", type=float, default=12.0, help="drop foreground px within this Lab dE of the backdrop colour (holes); 0 = off")
    ap.add_argument("--tag", default=None, help="suffix for output files (default: method name)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or args.method
    t0 = time.time()

    # photo
    photo_rgb, mask = photo_foreground(args.photo, args.erode, args.bg_reject)
    tgt_rgb = photo_rgb[mask]
    tgt_lab = to_lab(tgt_rgb)
    tgt_stats = robust_stats(tgt_lab)
    print(f"photo: {photo_rgb.shape[1]}x{photo_rgb.shape[0]}, foreground px {mask.sum():,} ({mask.mean()*100:.1f}%)")

    # model
    scene, meshes = load_model(args.model)
    record = {"photo": args.photo, "model": args.model, "method": args.method, "strength": args.strength, "erode_px": args.erode, "bg_reject_dE": args.bg_reject,
              "photo_foreground_px": int(mask.sum()), "target_lab_mean_std": tgt_stats, "geometries": {}}
    for name, mesh in meshes:
        mat = mesh.visual.material
        tex = np.asarray(mat.baseColorTexture.convert("RGBA"))
        tex_rgb = tex[..., :3]
        src_rgb = surface_texel_colours(mesh, tex_rgb)
        src_lab = to_lab(src_rgb)
        src_stats = robust_stats(src_lab)

        flat = to_lab(tex_rgb.reshape(-1, 3))
        if args.method == "reinhard":
            new = transfer_reinhard(flat, src_stats, tgt_stats)
        elif args.method == "hist":
            new = transfer_hist(flat, src_lab, tgt_lab)
        elif args.method == "chroma":  # a/b fully, L mean only
            new = transfer_reinhard(flat, src_stats, tgt_stats, channels=(0, 1, 2), l_std=False)
        else:  # hue: a/b mean shift only, L untouched
            new = flat.copy()
            for c in (1, 2):
                new[:, c] = flat[:, c] - src_stats[c][0] + tgt_stats[c][0]
        if args.strength < 1.0:
            new = flat + args.strength * (new - flat)
        new_rgb = from_lab(new).reshape(tex_rgb.shape)
        new_tex = np.concatenate([new_rgb, tex[..., 3:4]], axis=-1)
        new_img = Image.fromarray(new_tex, "RGBA")
        mat.baseColorTexture = new_img

        after_rgb = surface_texel_colours(mesh, new_rgb, n=500_000)
        after_stats = robust_stats(to_lab(after_rgb))
        de_before = float(np.linalg.norm(np.array([s[0] for s in src_stats]) - np.array([s[0] for s in tgt_stats])))
        de_after = float(np.linalg.norm(np.array([s[0] for s in after_stats]) - np.array([s[0] for s in tgt_stats])))
        record["geometries"][name] = {
            "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)), "texture_size": list(mat.baseColorTexture.size),
            "source_lab_mean_std": src_stats, "after_lab_mean_std": after_stats,
            "mean_colour_deltaE_before": de_before, "mean_colour_deltaE_after": de_after,
        }
        print(f"{name}: Lab mean  photo L{tgt_stats[0][0]:.1f} a{tgt_stats[1][0]:.1f} b{tgt_stats[2][0]:.1f} | "
              f"model before L{src_stats[0][0]:.1f} a{src_stats[1][0]:.1f} b{src_stats[2][0]:.1f} | after L{after_stats[0][0]:.1f} a{after_stats[1][0]:.1f} b{after_stats[2][0]:.1f}")
        print(f"   deltaE of mean colour vs photo: before {de_before:.1f} -> after {de_after:.1f}")
        Image.fromarray(tex_rgb).save(os.path.join(args.out, f"basecolor_before.png"))
        new_img.convert("RGB").save(os.path.join(args.out, f"basecolor_after_{tag}.png"))

    stem = os.path.splitext(os.path.basename(args.model))[0]
    out_glb = os.path.join(args.out, f"{stem}_colormatched_{tag}.glb")
    scene.export(out_glb)
    record["output_glb"] = out_glb
    print("wrote", out_glb, f"{os.path.getsize(out_glb)/1e6:.1f} MB")

    # ---- comparison figure: photo | before (4 views) | after (4 views) ----
    name, mesh = meshes[0]
    before_tex = np.asarray(Image.open(os.path.join(args.out, "basecolor_before.png")))
    after_tex = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"))
    Rup = rot("x", 90)   # glTF Y-up -> Z-up, glTF +Z (front) -> -y (toward the camera)
    views = [("front", Rup), ("left", rot("z", -90) @ Rup), ("back", rot("z", 180) @ Rup), ("right", rot("z", 90) @ Rup)]
    pts_b, nrm_b, col_b = sample_for_render(mesh, before_tex)
    pts_a, nrm_a, col_a = sample_for_render(mesh, after_tex)
    ext = np.ptp(pts_b, axis=0)
    centre = (pts_b.min(0) + pts_b.max(0)) / 2
    lim = 0.56 * max(ext)

    fig = plt.figure(figsize=(21, 9))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.3, 1, 1, 1, 1])
    ax = fig.add_subplot(gs[:, 0])
    ys, xs = np.where(mask)
    pad = 40
    y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad, mask.shape[0]), max(xs.min() - pad, 0), min(xs.max() + pad, mask.shape[1])
    crop = photo_rgb[y0:y1, x0:x1].copy()
    crop[~mask[y0:y1, x0:x1]] = 255
    ax.imshow(crop)
    ax.set_title("input photo (background removed)", fontsize=10)
    ax.axis("off")
    for j, (vname, R) in enumerate(views):
        render_points(fig.add_subplot(gs[0, j + 1]), pts_b, nrm_b, col_b, R, f"model BEFORE - {vname}", centre, lim)
        render_points(fig.add_subplot(gs[1, j + 1]), pts_a, nrm_a, col_a, R, f"model AFTER ({args.method}) - {vname}", centre, lim)
    fig.suptitle(f"Colour match to photo - method={args.method}, strength={args.strength}", fontsize=12)
    fig.tight_layout()
    fig_path = os.path.join(args.out, f"color_match_compare_{tag}.png")
    fig.savefig(fig_path, dpi=90)
    plt.close(fig)

    # ---- Lab histograms ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    before_lab = to_lab(surface_texel_colours(mesh, before_tex, n=300_000))
    after_lab = to_lab(surface_texel_colours(mesh, after_tex, n=300_000))
    for c, (a, lab) in enumerate(zip(axes, ("L (lightness)", "a (green - red)", "b (blue - yellow)"))):
        rng = (min(tgt_lab[:, c].min(), before_lab[:, c].min()), max(tgt_lab[:, c].max(), before_lab[:, c].max()))
        a.hist(tgt_lab[:, c], bins=80, range=rng, density=True, alpha=0.5, label="photo", color="#333")
        a.hist(before_lab[:, c], bins=80, range=rng, density=True, alpha=0.5, label="model before", color="#c33")
        a.hist(after_lab[:, c], bins=80, range=rng, density=True, alpha=0.5, label="model after", color="#27a")
        a.set_title(lab)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"color_match_hist_{tag}.png"), dpi=100)
    plt.close(fig)

    record["elapsed_s"] = time.time() - t0
    with open(os.path.join(args.out, f"color_match_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print("figure", fig_path, f"| elapsed {record['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
