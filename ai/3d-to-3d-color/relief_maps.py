"""Derive a height map (master) and a tangent-space normal map from a GLB's base-colour texture,
then write a new GLB with the normal map attached. Mesh, UVs and other textures are untouched.

Assumption (recorded as ai_inferred): darker texels are lower (cracks, pores, grain), after removing
large-scale colour/shading variation with a band-pass filter.

  base colour ──luminance──▶ band-pass (sigma_small .. sigma_large) ──invert──▶ height map (16-bit PNG, master)
                                                                                  └──gradient──▶ normal map (RGB, glTF/OpenGL convention) ──▶ GLB normalTexture

Usage
  python relief_maps.py --model in.glb --out out/ [--sigma-small 1.0] [--sigma-large 24] [--slope-deg 25] [--tag reinhard]
"""
import argparse
import json
import os
import time

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, binary_dilation, distance_transform_edt
from skimage import color as skcolor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def used_texel_mask(mesh, W, H, grow=2):
    """Texels actually referenced by the mesh UVs (rasterised triangles), grown by a few px for bilinear bleed."""
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    uv = np.asarray(mesh.visual.uv, dtype=np.float64)
    px = np.stack([uv[:, 0] * (W - 1), (1.0 - uv[:, 1]) * (H - 1)], axis=1)
    for tri in mesh.faces:
        d.polygon([tuple(px[i]) for i in tri], fill=255)
    m = np.asarray(img) > 0
    return binary_dilation(m, iterations=grow) if grow > 0 else m


def masked_blur(x, mask, sigma):
    """Gaussian blur that only averages texels inside the mask (normalised convolution)."""
    if sigma <= 0:
        return x
    num = gaussian_filter(x * mask, sigma)
    den = gaussian_filter(mask.astype(np.float64), sigma)
    return num / np.maximum(den, 1e-6)


def fill_nearest(x, mask):
    """Replace values outside the mask with the nearest inside value (like texture padding)."""
    idx = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return x[tuple(idx)]


def height_from_colour(rgb_u8, mask, sigma_small, sigma_large, lo=0.5, hi=99.5):
    """Band-passed, inverted luminance in [0,1] (0.5 = neutral), computed inside the used-texel mask only,
    so atlas padding and neighbouring UV islands do not leak into the relief."""
    lum = skcolor.rgb2lab(rgb_u8 / 255.0)[..., 0]           # perceptual lightness 0..100
    fine = masked_blur(lum, mask, sigma_small)
    base = masked_blur(lum, mask, sigma_large)
    h = fine - base                                           # brighter than surroundings -> higher, i.e. dark = low
    a, b = np.percentile(h[mask], [lo, hi])
    h = np.clip((h - a) / max(b - a, 1e-6), 0, 1)
    h = fill_nearest(h, mask)                                 # pad outside islands with edge values -> no seam gradients
    return h, float(a), float(b)


def normal_from_height(h, k):
    """glTF (OpenGL) tangent-space normal map: R=+u right, G=+v up, B=out. Image rows grow downward (= -v)."""
    d_col = np.gradient(h, axis=1)      # dH/du
    d_row = np.gradient(h, axis=0)      # dH/d(row) = -dH/dv
    nx = -k * d_col
    ny = +k * d_row                     # = -k * dH/dv
    nz = np.ones_like(h)
    n = np.stack([nx, ny, nz], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return ((n * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8), d_col, d_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma-small", type=float, default=1.0, help="px; suppress noise finer than this")
    ap.add_argument("--sigma-large", type=float, default=16.0, help="px; remove colour/shading variation coarser than this")
    ap.add_argument("--slope-deg", type=float, default=25.0, help="normal-map strength: 95th-percentile slope maps to this angle")
    ap.add_argument("--tag", default="relief")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    scene = trimesh.load(args.model, force="scene")
    name, mesh = next((n, g) for n, g in scene.geometry.items() if g.visual.kind == "texture" and g.visual.material.baseColorTexture is not None)
    mat = mesh.visual.material
    rgb = np.asarray(mat.baseColorTexture.convert("RGB"))
    H, W = rgb.shape[:2]

    t_mask = time.time()
    mask = used_texel_mask(mesh, W, H)
    print(f"used texels: {mask.mean()*100:.1f}% of atlas (rasterised {len(mesh.faces):,} triangles in {time.time()-t_mask:.0f}s)")
    h, a, b = height_from_colour(rgb, mask, args.sigma_small, args.sigma_large)
    # strength: choose k so the 95th percentile gradient magnitude (inside islands) becomes tan(slope_deg)
    g = np.hypot(np.gradient(h, axis=1), np.gradient(h, axis=0))
    p95 = float(np.percentile(g[mask], 95))
    k = float(np.tan(np.deg2rad(args.slope_deg)) / max(p95, 1e-9))
    nrm_u8, _, _ = normal_from_height(h, k)
    nrm_u8 = fill_nearest(nrm_u8, mask)
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(os.path.join(args.out, f"uvmask_{args.tag}.png"))

    # ---- files ----
    height16 = Image.fromarray((h * 65535 + 0.5).astype(np.uint16), mode="I;16")
    height16.save(os.path.join(args.out, f"height_{args.tag}_16bit.png"))
    Image.fromarray((h * 255 + 0.5).astype(np.uint8), mode="L").save(os.path.join(args.out, f"height_{args.tag}.png"))
    normal_img = Image.fromarray(nrm_u8, mode="RGB")
    normal_img.save(os.path.join(args.out, f"normal_{args.tag}.png"))

    mat.normalTexture = normal_img
    stem = os.path.splitext(os.path.basename(args.model))[0]
    out_glb = os.path.join(args.out, f"{stem}_normal.glb")
    scene.export(out_glb)

    # ---- record ----
    rec = {"model": args.model, "output_glb": out_glb, "geometry": name, "texture_size": [W, H],
           "method": "height = bandpass luminance inside used-texel mask (normalised convolution): dark = low; padding filled by nearest texel; normal = gradient of height (glTF/OpenGL tangent space)",
           "used_texel_fraction": float(mask.mean()),
           "sigma_small_px": args.sigma_small, "sigma_large_px": args.sigma_large,
           "height_norm_percentiles": [0.5, 99.5], "height_norm_range_L": [a, b],
           "slope_deg_at_p95": args.slope_deg, "gradient_p95": p95, "normal_strength_k": k,
           "provenance": "ai_inferred - relief is estimated from colour, not measured", "elapsed_s": time.time() - t0}
    with open(os.path.join(args.out, f"relief_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)

    # ---- figure: a detail crop of colour / height / normal ----
    cy, cx, s = H // 2, W // 2, 384
    crop = (slice(cy - s // 2, cy + s // 2), slice(cx - s // 2, cx + s // 2))
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
    axes[0].imshow(rgb[crop]); axes[0].set_title("base colour (crop 384 px)")
    axes[1].imshow(h[crop], cmap="gray", vmin=0, vmax=1); axes[1].set_title(f"height map  (band {args.sigma_small}-{args.sigma_large} px, dark = low)")
    axes[2].imshow(nrm_u8[crop]); axes[2].set_title(f"normal map  (k = {k:.1f}, p95 slope {args.slope_deg:.0f} deg)")
    # shaded preview of the height field (light from top-left)
    light = np.array([-0.5, -0.5, 0.7]); light /= np.linalg.norm(light)
    n = nrm_u8[crop].astype(float) / 127.5 - 1
    n[..., 1] *= -1  # image rows downward
    shade = np.clip((n @ light), 0, 1)
    axes[3].imshow(shade, cmap="gray", vmin=0, vmax=1); axes[3].set_title("relief preview (lit from top-left)")
    for a_ in axes:
        a_.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"relief_{args.tag}_preview.png"), dpi=90)
    plt.close(fig)

    print(f"height/normal {W}x{H}  band {args.sigma_small}-{args.sigma_large} px  k={k:.2f} (p95 gradient {p95:.4f})")
    print("wrote", out_glb, f"{os.path.getsize(out_glb)/1e6:.1f} MB", "| elapsed", f"{rec['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
