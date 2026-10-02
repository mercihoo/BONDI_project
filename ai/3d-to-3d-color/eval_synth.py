"""
eval_synth.py — 합성 부식 정답으로 복원 품질 측정 (설계 8절 F, 7절 정량 검증)

`synth_corrosion.py`가 가려 둔 구간의 **원래 색을 알고 있으므로**, 복원본을 그 원래 색과 직접 비교한다. 이것이
"손상부 ΔE(프리셋까지)"와 다른 점: 프리셋까지의 거리는 평평할수록 좋아지지만, 여기서는 **원래 표면과 같아야** 좋다.

  python eval_synth.py --key don000498_001 --material celadon --restored out_restore/synth_don000498_001_v19
  python eval_synth.py --key ... --restored A --restored B     # 두 버전을 같은 정답으로 비교

측정
  ΔE_ab, ΔL, ΔE_00      정답 구간 평균·중앙값·90 백분위
  대역별 표준편차 비      복원본 / 원본 (1.0이면 결의 세기가 같다)
  건전부 불변            부식을 입히지 않은 건전 텍셀의 ΔE (건드리지 않았는지)
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color as skcolor

Image.MAX_IMAGE_PIXELS = None
POC_IN = r"C:\ai\poc_inputs"


def band_std(x, mask, valid, sigmas=(3.0, 12.0, 48.0)):
    """Masked band decomposition (normalised convolution, so the atlas background cannot leak in)."""
    out, prev = [], x.astype(np.float32)
    m = valid.astype(np.float32)
    for s in sigmas:
        num = ndi.gaussian_filter(x.astype(np.float32) * m, s)
        den = ndi.gaussian_filter(m, s)
        lo = num / np.maximum(den, 1e-6)
        out.append(float((prev - lo)[mask].std()))
        prev = lo
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--material", required=True)
    ap.add_argument("--restored", action="append", required=True, help="복원 결과 폴더 (여러 번 주면 나란히 비교)")
    ap.add_argument("--synth-dir", default=None)
    a = ap.parse_args()

    od = a.synth_dir or os.path.join("out_restore", f"synth_{a.key}")
    truth = np.asarray(Image.open(os.path.join(od, "mask_truth.png")).convert("L")) > 127
    info = json.load(open(os.path.join(od, "synth.json"), encoding="utf-8"))
    src = np.asarray(Image.open(info["source_texture"]).convert("RGB"))
    lab_true = skcolor.rgb2lab(src / 255.0)
    base = os.path.join("out_restore", f"poc_{a.key}")
    valid = np.asarray(Image.open(os.path.join(base, f"mask_valid_{a.material}.png")).convert("L")) > 127
    healthy = np.asarray(Image.open(os.path.join(base, f"mask_healthy_{a.material}.png")).convert("L")) > 127
    kept = healthy & ~truth                                             # 부식을 입히지 않은 건전부

    corroded = np.asarray(Image.open(glob.glob(os.path.join(info["input_written"], "*.jpg"))[0]).convert("RGB"))
    lab_cor = skcolor.rgb2lab(corroded / 255.0)
    rows = [("부식된 입력", lab_cor)]
    for r in a.restored:
        f = glob.glob(os.path.join(r, "texture_restored_conserved_*.png"))
        if not f:
            print(f"   ! {r}: 복원 텍스처 없음")
            continue
        rows.append((os.path.basename(r), skcolor.rgb2lab(np.asarray(Image.open(f[0]).convert("RGB")) / 255.0)))

    tb = band_std(lab_true[..., 0], truth, valid)
    print(f"{a.key} ({a.material})  정답 구간 {truth.sum() / valid.sum():.1%} of valid, 남긴 건전부 {kept.sum() / valid.sum():.1%}")
    print(f"{'대상':22}{'ΔE_ab':>8}{'중앙':>7}{'p90':>7}{'ΔL':>8}{'ΔE00':>8}   {'대역 세기 비(3/12/48)':>22}   건전부 ΔE")
    for name, lab in rows:
        d_ab = np.hypot(lab[..., 1] - lab_true[..., 1], lab[..., 2] - lab_true[..., 2])[truth]
        dL = (lab[..., 0] - lab_true[..., 0])[truth]
        d00 = skcolor.deltaE_ciede2000(lab_true[truth].reshape(-1, 1, 3), lab[truth].reshape(-1, 1, 3)).ravel()
        bs = band_std(lab[..., 0], truth, valid)
        ratio = "/".join(f"{b / max(t, 1e-6):.2f}" for b, t in zip(bs, tb))
        k_ab = np.hypot(lab[..., 1] - lab_true[..., 1], lab[..., 2] - lab_true[..., 2])[kept].mean() if kept.any() else float("nan")
        print(f"{name:22}{d_ab.mean():8.2f}{np.median(d_ab):7.2f}{np.quantile(d_ab, 0.9):7.2f}{dL.mean():+8.2f}{d00.mean():8.2f}   {ratio:>22}   {k_ab:8.2f}")


if __name__ == "__main__":
    main()
