"""Semi-automatic protection / seam masks with SAM 2 (design section 8-A, first step: click prompts, no CLIP yet).

The rule-based v1.5 cannot isolate two things from colour + curvature alone: fragment join seams inside a UV chart on
coarse-textured earthenware, and glaze worn through to the body on celadon rims. SAM 2 segments *shapes*, so a few point
prompts on the texture atlas give masks for both. This script runs in texture space (the UV atlas downscaled to
--work-size) and writes full-resolution masks that restore_original_colour.py takes via --crack-mask / --wear-mask.

Prompts come from a JSON file (a person clicks; coordinates in *work* pixels of the atlas) or from --auto, which turns the
rule-based weak detections into prompt points so the same code path can be tested without a UI:

  prompts.json = {"seam": [[x, y] or [x0, y0, x1, y1], ...], "wear": [...], "negative": [[x, y], ...]}  (clicks -> 96 px boxes)

  python segment_masks.py --texture <8K jpg> --model <obj|glb> --out <dir> --prompts prompts.json
  python segment_masks.py --texture <8K jpg> --model <obj|glb> --out <dir> --auto seam --material earthenware
  python segment_masks.py --texture <8K jpg> --model <obj|glb> --out <dir> --auto wear --material celadon

Outputs: mask_sam_seam.png / mask_sam_wear.png (texture resolution, white = mask), prompts_used.json,
sam_overlay.jpg (work-size overlay for inspection). Runs on the RTX 4070 (8 GB) with sam2.1_b in ~1-2 s per prompt.
"""
import argparse, json, os, time
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage import color as skcolor

Image.MAX_IMAGE_PIXELS = None


def load_texture(path, work):
    img = Image.open(path).convert("RGB")
    W, H = img.size
    small = img.resize((work, work), Image.BOX) if max(W, H) > work else img
    return img, small, (W, H)


def used_mask(model_path, W, H):
    from relief_maps import used_texel_mask
    m = trimesh.load(model_path, force="mesh", process=False)
    return used_texel_mask(m, W, H, grow=0), m


def boxes_along_component(lbl, i, sl, step=96, size=96):
    """Small square boxes along a long thin component (so SAM sees the line, not the chart)."""
    ys, xs = np.nonzero(lbl[sl] == i)
    ys = ys + sl[0].start
    xs = xs + sl[1].start
    order = np.argsort(xs if (sl[1].stop - sl[1].start) >= (sl[0].stop - sl[0].start) else ys)
    xs, ys = xs[order], ys[order]
    out = []
    for j in range(0, len(xs), max(1, len(xs) // max(1, int(np.hypot(sl[1].stop - sl[1].start, sl[0].stop - sl[0].start) // step)))):
        cx, cy = int(xs[j]), int(ys[j])
        out.append([cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2])
    return out


def auto_prompts_seam(lab_small, valid_small, n_boxes=48, min_len=60):
    """Weak seam evidence -> prompt BOXES along long thin bright/dark ridges (wide base removed)."""
    L = lab_small[..., 0]
    base = ndi.gaussian_filter(np.where(valid_small, L, np.nan_to_num(L)), 7)
    d = L - base
    cand = valid_small & ((d > 2.0) | (d < -2.5))
    cand = ndi.binary_opening(cand, iterations=1)
    lbl, n = ndi.label(cand, structure=np.ones((3, 3)))
    pts = []
    if n:
        objs = ndi.find_objects(lbl)
        areas = ndi.sum(cand, lbl, index=np.arange(1, n + 1))
        for i, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            h_, w_ = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            diag = float(np.hypot(w_, h_))
            if diag >= min_len and areas[i - 1] / diag <= 6:
                pts += boxes_along_component(lbl, i, sl)
    return pts[:n_boxes]


def auto_prompts_wear(lab_small, valid_small, convex_small, ref_ab, n_boxes=32):
    """Brown texels on the most convex part of the shape -> prompt BOXES (blob bbox grown 30 %)."""
    a, b = lab_small[..., 1], lab_small[..., 2]
    de = np.hypot(a - ref_ab[0], b - ref_ab[1])
    warm = (a > 2) & (b > 8)
    thr = np.quantile(convex_small[valid_small], 0.85)
    cand = valid_small & warm & (de > 12) & (convex_small > thr)
    cand = ndi.binary_opening(cand, iterations=1)
    lbl, n = ndi.label(cand)
    pts = []
    if n:
        areas = ndi.sum(cand, lbl, index=np.arange(1, n + 1))
        order = np.argsort(-areas)
        objs = ndi.find_objects(lbl)
        for i in order:
            if areas[i] < 15:
                break
            sl = objs[i]
            h_, w_ = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            g = int(max(8, 0.3 * max(h_, w_)))
            pts.append([sl[1].start - g, sl[0].start - g, sl[1].stop + g, sl[0].stop + g])
            if len(pts) >= n_boxes:
                break
    return pts


def run_sam_boxes(small_rgb, boxes, weights="sam2.1_b.pt", thin=None, max_box_ratio=3.0):
    """One SAM 2 call per box prompt; keep the mask only if it stays near the box (area <= max_box_ratio x box area) and,
    for seams, is thin (area / bbox diagonal <= thin). Point prompts on a texture atlas select the whole chart; a small
    box makes the line or blob inside it the salient object."""
    from ultralytics import SAM
    model = SAM(weights)
    Hs, Ws = small_rgb.shape[:2]
    union = np.zeros((Hs, Ws), bool)
    kept, tried = 0, 0
    for b in boxes:
        x0, y0, x1, y1 = [int(v) for v in b]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(Ws - 1, x1), min(Hs - 1, y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        tried += 1
        res = model(small_rgb, bboxes=[[x0, y0, x1, y1]], verbose=False)
        if not res or res[0].masks is None:
            continue
        m = res[0].masks.data.cpu().numpy().astype(bool).reshape(-1, Hs, Ws) if res[0].masks.data.shape[-2:] == (Hs, Ws) else None
        if m is None:
            mm0 = res[0].masks.data.cpu().numpy().astype(bool)
            mm0 = mm0.reshape(-1, *mm0.shape[-2:])
            m = np.stack([np.asarray(Image.fromarray(x.astype(np.uint8) * 255).resize((Ws, Hs), Image.NEAREST)) > 0 for x in mm0])
        mm = np.any(m, axis=0)
        grow = int(0.5 * max(x1 - x0, y1 - y0))
        window = np.zeros_like(mm)
        window[max(0, y0 - grow):y1 + grow, max(0, x0 - grow):x1 + grow] = True
        mm &= window
        area = mm.sum()
        if area == 0 or area > max_box_ratio * (x1 - x0) * (y1 - y0):
            continue
        if thin is not None:
            ys, xs = np.nonzero(mm)
            diag = float(np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
            if area / diag > thin:
                continue
        union |= mm
        kept += 1
    return union, {"boxes_tried": tried, "boxes_kept": kept}


def run_sam(small_rgb, points, labels, weights="sam2.1_b.pt"):
    """One SAM 2 call per positive point (with all negatives), union of masks. ultralytics wrapper, GPU if available."""
    from ultralytics import SAM
    model = SAM(weights)
    union = np.zeros(small_rgb.shape[:2], bool)
    per_point = []
    neg = [p for p, l in zip(points, labels) if l == 0]
    for p, l in zip(points, labels):
        if l != 1:
            continue
        pts = [p] + neg
        lbs = [1] + [0] * len(neg)
        res = model(small_rgb, points=[pts], labels=[lbs], verbose=False)
        if not res or res[0].masks is None:
            continue
        m = res[0].masks.data.cpu().numpy().astype(bool)
        m = m.reshape(-1, *m.shape[-2:])
        # ultralytics returns one mask per prompt set; take the union of returned masks for this point
        mm = np.any(m, axis=0)
        if mm.shape != union.shape:
            mm = np.asarray(Image.fromarray(mm.astype(np.uint8) * 255).resize(union.shape[::-1], Image.NEAREST)) > 0
        per_point.append(float(mm.mean()))
        union |= mm
    return union, per_point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texture", required=True)
    ap.add_argument("--model", required=True, help="OBJ/GLB for the UV-used texel mask (and curvature for --auto wear)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=None, help="JSON with seam / wear / negative point lists in work pixels")
    ap.add_argument("--auto", choices=["seam", "wear"], default=None, help="derive prompt points from the rule-based weak detections")
    ap.add_argument("--material", default="celadon")
    ap.add_argument("--work-size", type=int, default=2048)
    ap.add_argument("--weights", default="sam2.1_b.pt")
    ap.add_argument("--thin", type=float, default=10.0, help="seam masks: max area / bbox-diagonal (px at work size) to count as a line")
    ap.add_argument("--max-frac", type=float, default=0.15, help="reject a mask larger than this fraction of the valid atlas (SAM grabbed a whole chart)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    full, small, (W, H) = load_texture(a.texture, a.work_size)
    ws = small.size[0]
    valid_full, mesh = used_mask(a.model, W, H)
    valid_small = np.asarray(Image.fromarray(valid_full.astype(np.uint8) * 255).resize((ws, ws), Image.NEAREST)) > 0
    small_rgb = np.asarray(small)
    lab_small = skcolor.rgb2lab(small_rgb / 255.0)

    prompts = {"seam": [], "wear": [], "negative": []}
    if a.prompts:
        prompts.update(json.load(open(a.prompts, encoding="utf-8")))
    if a.auto == "seam":
        prompts["seam"] += auto_prompts_seam(lab_small, valid_small)
    if a.auto == "wear":
        from restore_original_colour import curvature_texture, PRESETS, dominant_ab
        convex_full = curvature_texture(mesh, W, H)
        convex_small = np.asarray(Image.fromarray((convex_full * 255).astype(np.uint8)).resize((ws, ws), Image.BOX)) / 255.0
        ref = dominant_ab(lab_small, valid_small)
        prompts["wear"] += auto_prompts_wear(lab_small, valid_small, convex_small, ref)
    print(f"prompts: seam {len(prompts['seam'])}, wear {len(prompts['wear'])}, negative {len(prompts['negative'])}")

    overlay = small.copy()
    drw = ImageDraw.Draw(overlay)
    rec = {"texture": a.texture, "work_size": ws, "weights": a.weights, "prompts": prompts, "masks": {}}
    for kind in ("seam", "wear"):
        pts = prompts[kind]
        if not pts:
            continue
        boxes = [q if len(q) == 4 else [q[0] - 48, q[1] - 48, q[0] + 48, q[1] + 48] for q in pts]   # clicks -> 96 px boxes
        union, per = run_sam_boxes(small_rgb, boxes, a.weights, thin=(a.thin if kind == "seam" else None))
        union &= valid_small
        # drop masks that swallowed a whole chart
        lbl, n = ndi.label(union)
        if n:
            areas = ndi.sum(union, lbl, index=np.arange(1, n + 1)) / max(valid_small.sum(), 1)
            keep = np.zeros(n + 1, bool)
            keep[1:] = areas <= a.max_frac
            union = keep[lbl]
        full_mask = np.asarray(Image.fromarray(union.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 0
        full_mask &= valid_full
        Image.fromarray(full_mask.astype(np.uint8) * 255).save(os.path.join(a.out, f"mask_sam_{kind}.png"))
        rec["masks"][kind] = {"fraction_of_valid": float(full_mask.sum() / max(valid_full.sum(), 1)), "boxes": per,
                              "components": int(ndi.label(union)[1])}
        colour = (255, 40, 40) if kind == "seam" else (40, 120, 255)
        ov = np.asarray(overlay).copy()
        ov[union] = (0.45 * ov[union] + 0.55 * np.array(colour)).astype(np.uint8)
        overlay = Image.fromarray(ov)
        drw = ImageDraw.Draw(overlay)
        for bx in boxes:
            drw.rectangle([bx[0], bx[1], bx[2], bx[3]], outline=colour, width=2)
        print(f"{kind}: mask {rec['masks'][kind]['fraction_of_valid']*100:.2f}% of valid, {rec['masks'][kind]['components']} components")
    for x, y in prompts["negative"]:
        drw.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(0, 0, 0), width=3)
    overlay.save(os.path.join(a.out, "sam_overlay.jpg"), quality=88)
    rec["elapsed_s"] = time.time() - t0
    with open(os.path.join(a.out, "prompts_used.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    print("done", f"{rec['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
