"""
synth_corrosion.py — 합성 부식으로 정답 만들기 (설계 8절 F)

복원이 좋아졌는지 잴 방법이 없다는 것이 지금까지 가장 큰 공백이었다. 손상부 ΔE는 "평평한 프리셋 색에 얼마나 가까운가"라
결을 살릴수록 나빠지고, 육안 검수는 사람 시간을 먹는다. 정답이 있으려면 **원래 색을 아는 표면**이 필요하다.

방법: 같은 유물의 **살아남은 건전 표면**을 정답으로 쓴다. 그 일부에 부식을 합성해 입히고, 복원을 돌린 뒤, 가려 두었던
원래 색과 비교한다. 건전부 전체를 덮으면 복원이 기준을 잃으므로 절반만 덮는다(`--cover`).

  python synth_corrosion.py --key don000498_001 --material celadon [--cover 0.5] [--severity 1.0] [--seed 0]
  → C:/ai/poc_synth/<key>/<stem>.{obj,mtl,jpg}  (부식된 텍스처)
    out_restore/synth_<key>/mask_truth.png       (정답 구간 = 합성 부식을 입힌 건전 텍셀)
    out_restore/synth_<key>/synth.json           (설정과 합성 전후 통계)

그다음 복원을 그 입력으로 돌리고 `eval_synth.py`로 정답 구간의 ΔE를 잰다.

부식 모델은 실제 손상부의 통계에 맞춘다: 같은 유물에서 측정한 (손상부 − 건전부)의 평균 L 차, 평균 a/b 차, 대역별
표준편차 비를 목표로 삼아 강도를 자동으로 맞춘다(`--severity`는 그 위의 배율).
"""
import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color as skcolor

Image.MAX_IMAGE_PIXELS = None
POC_IN = r"C:\ai\poc_inputs"
POC_SYNTH = r"C:\ai\poc_synth"


def smooth_noise(shape, sigma, rng):
    """Zero-mean smooth random field, unit standard deviation."""
    n = rng.normal(0.0, 1.0, shape).astype(np.float32)
    n = ndi.gaussian_filter(n, sigma)
    s = float(n.std())
    return n / max(s, 1e-6)


def blotch_field(shape, rng, scales=(120.0, 40.0, 12.0), weights=(1.0, 0.6, 0.3)):
    """Multi-scale blotches in 0..1. Real corrosion is patchy at several scales at once, so a single-scale field looks
    obviously fake (and would be trivially removed by the destain step, which works at one scale)."""
    f = np.zeros(shape, np.float32)
    for s, w in zip(scales, weights):
        f += w * smooth_noise(shape, s, rng)
    f -= f.min()
    return f / max(float(f.max()), 1e-6)


def corrode(lab, valid, target_mask, patina_ab, rng, cover=0.5, severity=1.0, dL=12.0, detail_loss=0.6,
            speckle=1.5, scales=(120.0, 40.0, 12.0)):
    """Put synthetic corrosion on `target_mask`: darken toward a patina colour in multi-scale patches, flatten the
    surface detail under them, and add fine speckle. Returns the corroded Lab and the mask actually affected."""
    H, W = valid.shape
    f = blotch_field((H, W), rng, scales)
    thr = float(np.quantile(f[target_mask], 1.0 - cover)) if target_mask.any() else 1.0
    s = np.clip((f - thr) / max(1.0 - thr, 1e-6), 0, 1) * target_mask
    s = (s * severity).clip(0, 1).astype(np.float32)
    hit = target_mask & (s > 0.05)
    out = lab.astype(np.float32).copy()
    L = out[..., 0]
    lo = ndi.gaussian_filter(L, 6.0)
    out[..., 0] = np.where(hit, lo + (L - lo) * (1.0 - detail_loss * s) - dL * s, L)          # darken + flatten detail
    for c, p in ((1, patina_ab[0]), (2, patina_ab[1])):
        out[..., c] = np.where(hit, out[..., c] + (p - out[..., c]) * s, out[..., c])         # drift toward patina
    out[..., 0] += np.where(hit, smooth_noise((H, W), 1.2, rng) * speckle * s, 0)             # pitting speckle
    out[..., 0] = np.clip(out[..., 0], 0, 100)
    return out, hit, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, help="유물 키 (C:/ai/poc_inputs/<key>/ 와 out_restore/poc_<key>/ 를 쓴다)")
    ap.add_argument("--material", required=True)
    ap.add_argument("--cover", type=float, default=0.5, help="건전 표면 중 부식을 입힐 비율 (나머지는 복원의 기준으로 남긴다)")
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--detail-loss", type=float, default=None, help="얼룩 밑에서 지울 결의 비율. 기본: 유물 자체의 실제 손상부/건전부 중간 대역(6~24 px) 세기 비에서 측정 (1 - 비). 9/13 판은 0.6 고정이었고, 금동 bon009435의 실제 부식은 결을 거의 안 지운다(비 0.97~1.01)")
    ap.add_argument("--out-root", default=POC_SYNTH)
    ap.add_argument("--tag", default=None, help="같은 유물의 다른 합성 설정을 나란히 두려면: 입력 <key>_<tag>/, 정답 out_restore/synth_<key>_<tag>/")
    a = ap.parse_args()

    src = os.path.join(POC_IN, a.key)
    stem = [f[:-4] for f in os.listdir(src) if f.endswith(".obj")][0]
    base = os.path.join("out_restore", f"poc_{a.key}")
    rec = json.load(open([os.path.join(base, f) for f in os.listdir(base) if f.startswith("restore_") and f.endswith(".json")][0], encoding="utf-8"))
    valid = np.asarray(Image.open(os.path.join(base, f"mask_valid_{a.material}.png")).convert("L")) > 127
    healthy = np.asarray(Image.open(os.path.join(base, f"mask_healthy_{a.material}.png")).convert("L")) > 127
    dmg = np.asarray(Image.open(os.path.join(base, f"mask_damage_{a.material}.png")).convert("L")) >= 51
    tex_path = [os.path.join(src, f) for f in os.listdir(src) if f.lower().endswith((".jpg", ".png"))][0]
    rgb = np.asarray(Image.open(tex_path).convert("RGB"))
    lab = skcolor.rgb2lab(rgb / 255.0)

    real = dmg & valid
    if real.sum() > 1000 and healthy.sum() > 1000:                      # aim at this artefact's own damage statistics
        dL = float(lab[..., 0][healthy].mean() - lab[..., 0][real].mean())
        patina = (float(lab[..., 1][real].mean()), float(lab[..., 2][real].mean()))
    else:
        dL, patina = 12.0, (6.0, 14.0)
    dL = float(np.clip(dL, 4.0, 25.0))
    # How much surface detail does THIS artefact's real corrosion destroy? Measured as the mid-band (6-24 px) energy of
    # L in the real damaged area over the healthy area, with a masked blur so the atlas background cannot leak in.
    detail_loss = a.detail_loss
    band_ratio = None
    if detail_loss is None and real.sum() > 1000 and healthy.sum() > 1000:
        m = valid.astype(np.float32)
        L0 = lab[..., 0].astype(np.float32) * m
        lo6 = ndi.gaussian_filter(L0, 6.0) / np.maximum(ndi.gaussian_filter(m, 6.0), 1e-6)
        lo24 = ndi.gaussian_filter(L0, 24.0) / np.maximum(ndi.gaussian_filter(m, 24.0), 1e-6)
        mid = lo6 - lo24
        band_ratio = float(mid[real].std() / max(mid[healthy].std(), 1e-6))
        detail_loss = float(np.clip(1.0 - band_ratio, 0.0, 0.9))
    if detail_loss is None:
        detail_loss = 0.6
    rng = np.random.default_rng(a.seed)
    out_lab, hit, sev = corrode(lab, valid, healthy, patina, rng, a.cover, a.severity, dL, detail_loss=detail_loss)
    out_rgb = (np.clip(skcolor.lab2rgb(out_lab), 0, 1) * 255 + 0.5).astype(np.uint8)
    out_rgb = np.where(valid[..., None], out_rgb, rgb)                  # atlas background untouched

    name = a.key if not a.tag else f"{a.key}_{a.tag}"
    dst = os.path.join(a.out_root, name)
    os.makedirs(dst, exist_ok=True)
    for ext in (".obj", ".mtl"):
        shutil.copy2(os.path.join(src, stem + ext), os.path.join(dst, stem + ext))
    Image.fromarray(out_rgb).save(os.path.join(dst, os.path.basename(tex_path)), quality=97)
    od = os.path.join("out_restore", f"synth_{name}")
    os.makedirs(od, exist_ok=True)
    Image.fromarray((hit * 255).astype(np.uint8)).save(os.path.join(od, "mask_truth.png"))
    Image.fromarray((sev * 255).astype(np.uint8)).save(os.path.join(od, "mask_severity.png"))

    de = np.hypot(out_lab[..., 1] - lab[..., 1], out_lab[..., 2] - lab[..., 2])
    info = {"key": a.key, "material": a.material, "cover": a.cover, "severity": a.severity, "seed": a.seed,
            "source_texture": tex_path, "input_written": dst,
            "patina_ab_from_real_damage": patina, "dL_from_real_damage": dL,
            "detail_loss": detail_loss, "mid_band_ratio_real_damage_over_healthy": band_ratio,
            "healthy_fraction_of_valid": float(healthy.sum() / valid.sum()),
            "truth_fraction_of_valid": float(hit.sum() / valid.sum()),
            "healthy_left_fraction_of_valid": float((healthy & ~hit).sum() / valid.sum()),
            "applied": {"mean_dL": float((lab[..., 0] - out_lab[..., 0])[hit].mean()),
                        "mean_dE_ab": float(de[hit].mean()),
                        "L_std_before": float(lab[..., 0][hit].std()), "L_std_after": float(out_lab[..., 0][hit].std())},
            "real_damage_reference": {"mean_L_healthy": float(lab[..., 0][healthy].mean()),
                                      "mean_L_damage": float(lab[..., 0][real].mean()) if real.any() else None}}
    json.dump(info, open(os.path.join(od, "synth.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{a.key}: 건전부 {info['healthy_fraction_of_valid']:.1%} 중 {info['truth_fraction_of_valid']:.1%}에 부식 합성 "
          f"(남은 건전부 {info['healthy_left_fraction_of_valid']:.1%})")
    print(f"   입힌 손상: ΔL {info['applied']['mean_dL']:.1f}, ΔE_ab {info['applied']['mean_dE_ab']:.1f} "
          f"(실제 손상부 기준 ΔL {dL:.1f}, 파티나 ab {patina[0]:.1f}/{patina[1]:.1f}, 결 손실 {detail_loss:.2f}"
          + (f" ← 실제 중간 대역 비 {band_ratio:.2f})" if band_ratio is not None else ")"))
    print(f"   입력: {dst}")


if __name__ == "__main__":
    main()
