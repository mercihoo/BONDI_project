"""Viewer copies of the PoC regression results: original OBJ and every restored variant re-exported as GLB with the
base-colour texture downscaled to 2048 px so the three.js viewer (out_restore/viewer.html) loads them quickly.
Meshes are untouched (regression ran with --no-smooth, so original and restored share the geometry); vertex normals are
recomputed per position so scan meshes shade smoothly under a metallic material.

  python make_viewer_copies.py [--size 2048] [--out out_restore/viewer] [--only key ...]
"""
import argparse, glob, json, os, time
import numpy as np
import trimesh
from PIL import Image
from relief_maps import used_texel_mask, fill_nearest

POC = r"C:\ai\poc_inputs"
ITEMS = [  # key, input folder, restored folders (every *_restored_*.glb in them becomes a viewer variant)
    ("jan_v0", "jan_v0", ["poc_jan_v0", "poc_jan_v0_g04", "poc_jan_v0_g02", "poc_jan_v0_blue1", "poc_jan_v0_blue2"]),
    ("bari_v0", "bari_v0", ["poc_bari_v0_A_normal", "poc_bari_v0_B_flat", "poc_bari_v0_C_b050", "poc_bari_v0_D_dark_normal", "poc_bari_v0_E_dark_flat"]),
    ("gubdari_v2", "gubdari_v2", ["poc_gubdari_v2", "poc_gubdari_v2_flat48", "poc_gubdari_v2_nrm6", "poc_gubdari_v2_white1", "poc_gubdari_v2_white2"]),
    ("gubdari_v0", "gubdari_v0", ["poc_gubdari_v0"]),
    ("bon009435", "bon009435", ["poc_bon009435", "poc_bon009435_destain0", "poc_bon009435_keep8", "poc_bon009435_keep8s", "poc_bon009435_keep8r", "poc_bon009435_keep24", "poc_bon009435_geom1"]),   # v2.1 destain experiments
    ("duk006294", "duk006294", ["poc_duk006294", "poc_duk006294_v19"]),
    ("bon002789", "bon002789", ["poc_bon002789"]),
    ("bon004740", "bon004740", ["poc_bon004740"]),
    ("lkh000002", "lkh000002", ["poc_lkh000002"]),
    ("duk003312", "duk003312", ["poc_duk003312", "poc_duk003312_nodark", "poc_duk003312_destain", "poc_duk003312_tone",
                                "poc_duk003312_v12", "poc_duk003312_t03", "poc_duk003312_v13"]),
    ("don000498_001", "don000498_001", ["poc_don000498_001", "poc_don000498_001_sam", "poc_don000498_001_v19"]),
    ("don000498_002", "don000498_002", ["poc_don000498_002"]),
    ("gae000001_001", "gae000001_001", ["poc_gae000001_001"]),
    ("gae000001_002", "gae000001_002", ["poc_gae000001_002"]),
    ("ssu022891", "ssu022891", ["poc_ssu022891", "poc_ssu022891_sam", "poc_ssu022891_v16", "poc_ssu022891_v17", "poc_ssu022891_brush", "poc_ssu022891_ideal"]),
]
_B = r"C:\ai\poc_results\duk003312\duk003312_pensive-bodhisattva-nt83\digital_obj"
EXTERNAL = {  # key -> (label, obj, diffuse jpg, normal jpg or None, (metallic, roughness)) from other pipelines
    "duk003312": [
        ("poc0827_conserved", _B + r"\duk003312_restored_model.obj", _B + r"\duk003312_restored_diffuse.jpg", _B + r"\duk003312_restored_normal.jpg", (0.6, 0.55)),
        ("poc0827_pristine", _B + r"\duk003312_restored_model.obj", _B + r"\duk003312_restored_diffuse-pristine.jpg", _B + r"\duk003312_restored_normal.jpg", (0.6, 0.55)),
    ],
}


def smooth_position_normals(m):
    """Area-weighted vertex normals shared by every vertex at the same position (UV-seam duplicates included).
    Scan OBJs come out faceted after quad triangulation and vertex splitting; under a metallic material every facet shows."""
    key = np.round(m.vertices, 6)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    v = m.vertices
    fn = np.cross(v[m.faces[:, 1]] - v[m.faces[:, 0]], v[m.faces[:, 2]] - v[m.faces[:, 0]])   # area-weighted face normals
    acc = np.zeros((inv.max() + 1, 3))
    for k in range(3):
        np.add.at(acc, inv[m.faces[:, k]], fn)
    n = acc[inv]
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    m.vertex_normals = n
    return m


def as_pbr_mesh(path):
    m = trimesh.load(path, force="mesh", process=False)
    smooth_position_normals(m)
    mat = m.visual.material
    if not isinstance(mat, trimesh.visual.material.PBRMaterial):
        mat = mat.to_pbr()
    return m, mat


def shrink(img, size, mesh=None):
    """Downscale for the viewer. With a mesh, texels outside the UV islands are first filled with the nearest island
    colour, so LANCZOS filtering and GPU mipmaps do not pull the atlas background into island borders (seen as dark
    seam lines at a distance on gae000001)."""
    img = img.convert("RGB")
    if mesh is not None:
        W, H = img.size
        used = used_texel_mask(mesh, W, H, grow=0)
        rgb = np.asarray(img)
        img = Image.fromarray(fill_nearest(rgb, used).astype(np.uint8))
    if max(img.size) > size:
        img = img.resize((size, size), Image.LANCZOS)
    return img


def export(m, mat, tex, out, pbr=None, normal=None):
    mat.baseColorTexture = tex
    if pbr is not None:               # material preset (metallic, roughness) so the lit view behaves like the material
        mat.metallicFactor, mat.roughnessFactor = float(pbr[0]), float(pbr[1])
    if normal is not None:
        mat.normalTexture = normal
    m.visual.material = mat
    try:
        m.export(out, include_normals=True)
    except TypeError:
        m.export(out)
    return os.path.getsize(out) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--out", default="out_restore/viewer")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--manifest-only", action="store_true", help="GLB는 그대로 두고 manifest.json만 다시 쓴다 (라벨·설명만 바뀌었을 때)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mpath = os.path.join(a.out, "manifest.json")
    manifest = json.load(open(mpath, encoding="utf-8")) if ((a.only or a.manifest_only) and os.path.exists(mpath)) else {}   # --only / --manifest-only merge
    for key, inp, folders in ITEMS:
        if a.only and key not in a.only:
            continue
        t0 = time.time()
        obj = glob.glob(os.path.join(POC, inp, "*.obj"))[0]
        stem = os.path.splitext(os.path.basename(obj))[0]
        if a.manifest_only:
            n_faces, s0 = (manifest.get(key) or {}).get("faces", 0), 0.0
        else:
            m0, mat0 = as_pbr_mesh(obj)
            n_faces = int(len(m0.faces))
            s0 = export(m0, mat0, shrink(mat0.baseColorTexture, a.size, m0), os.path.join(a.out, f"{key}_original.glb"))
        variants = []
        for folder in folders:
            for glb in sorted(glob.glob(os.path.join("out_restore", folder, f"{stem}_restored_*.glb"))):
                suffix = os.path.basename(glb)[len(stem) + len("_restored_"):-4]      # e.g. conserved_gilt_bronze_v13
                mode = suffix.split("_")[0]
                tag = suffix[len(mode) + 1:]
                js = glob.glob(os.path.join("out_restore", folder, f"restore_{tag}.json"))
                rec = json.load(open(js[0], encoding="utf-8")) if js else {}
                # The experiment folders re-run the SAME stem, so their GLBs carry the same suffix as the main run
                # (bon009435-000-30000_restored_conserved_gilt_bronze.glb). Without the folder's tag in the published
                # name, every variant overwrites the previous one and the viewer shows the last one written in all of
                # them. The older folders (_v13, _tone) escaped this only because their filenames already had a tag.
                etag = folder[len(f"poc_{key}_"):] if folder.startswith(f"poc_{key}_") else None
                name = f"{key}_restored_{suffix}" + (f"_{etag}" if etag and not suffix.endswith(etag) else "")
                out = os.path.join(a.out, f"{name}.glb")
                if a.manifest_only:
                    nrm1 = rec.get("normal_map")
                else:
                    m1, mat1 = as_pbr_mesh(glb)
                    nrm1 = shrink(mat1.normalTexture, a.size) if getattr(mat1, "normalTexture", None) is not None else None
                    export(m1, mat1, shrink(mat1.baseColorTexture, a.size, m1), out, pbr=rec.get("preset", {}).get("pbr"), normal=nrm1)
                met = rec.get("metrics", {})
                # Which run produced this? The frozen experiment folders (poc_<key>_v13, _tone, ...) are NOT rerun by the
                # regression, so a viewer that only shows tone/chroma/destain cannot tell today's result from a 9/05 one.
                variants.append({"file": os.path.basename(out), "mode": mode, "tag": tag, "protect_dark": rec.get("protect_dark"),
                                 "material": rec.get("material"),
                                 "folder": folder, "current": folder == f"poc_{key}",
                                 "exp_tag": None if folder == f"poc_{key}" else folder[len(f"poc_{key}_"):] if folder.startswith(f"poc_{key}_") else folder,
                                 "destain_keep": (rec.get("destain") or {}).get("keep_sigma"), "destain_geom": (rec.get("destain") or {}).get("geom"),
                                 "stamp": int(os.path.getmtime(out)) if os.path.exists(out) else 0,
                                 "run_date": time.strftime("%m/%d", time.localtime(os.path.getmtime(js[0]))) if js else None,
                                 "passthrough": bool(((rec.get("metrics") or {}).get("passthrough") or {})),
                                 "target_source": rec.get("target_source", ""),
                                 "damage_dE_before": met.get("damage_dE_ab_to_target_before"), "damage_dE_after": met.get("damage_dE_ab_to_target_after"),
                                 "protect_fraction": met.get("protect_fraction"), "L_ssim": met.get("L_ssim_valid"),
                                 "tone": rec.get("tone", {}).get("strength"), "destain": rec.get("destain", {}).get("strength"),
                                 "chroma": rec.get("chroma_strength"), "pbr": rec.get("preset", {}).get("pbr"), "normal_map": nrm1 is not None,
                                 "masks": [k for k, v in (("wear", rec.get("wear_mask")), ("crack", rec.get("crack_mask"))) if v],
                                 "mask_src": ("붓" if "mask_paint" in (rec.get("crack_mask") or "") else "SAM") if rec.get("crack_mask") else None,
                                 "crack_mask": rec.get("crack_mask"), "crack_params": rec.get("crack_params"),
                                 "band_match": ((rec.get("metrics") or {}).get("band_match") or {}).get("reference"),
                                 "groove": (rec.get("crack_geometry") or {}).get("method"), "method": rec.get("method"), "mirror": (rec.get("pattern_mirror") or {}).get("band_mm"),
                                 "reemboss": ((rec.get("crack_geometry") or {}).get("reemboss") or {}).get("mean_abs_mm")})
        for label, obj_e, diff_e, nrm_e, pbr_e in EXTERNAL.get(key, []):
            out = os.path.join(a.out, f"{key}_{label}.glb")
            if not a.manifest_only:
                m2, mat2 = as_pbr_mesh(obj_e)
                nrm = shrink(Image.open(nrm_e), a.size) if nrm_e else None
                export(m2, mat2, shrink(Image.open(diff_e), a.size, m2), out, pbr=pbr_e, normal=nrm)
            variants.append({"file": os.path.basename(out), "mode": label, "tag": "external", "protect_dark": None,
                             "target_source": "8/27 PoC pipeline.py output (external)", "damage_dE_before": None, "damage_dE_after": None,
                             "protect_fraction": None, "L_ssim": None, "tone": None, "destain": None, "chroma": None, "pbr": list(pbr_e), "external": True})
        orig = os.path.join(a.out, f"{key}_original.glb")
        manifest[key] = {"material": variants[0]["tag"].split("_")[0] if variants else "", "faces": n_faces,
                         "stamp": int(os.path.getmtime(orig)) if os.path.exists(orig) else 0, "variants": variants}
        print(f"{key:15s} faces {n_faces:>7,}  original {s0:5.1f} MB  variants {len(variants)}  {time.time()-t0:.0f}s", flush=True)
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
