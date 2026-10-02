# -*- coding: utf-8 -*-
"""EA_028(토기 항아리) PoinTr 완성 실험.
- 8/26 restore_standalone.py의 셰임/로더 재사용
- ShapeNet-55는 정준 자세(Y-up) 학습이므로 원자세(Z-up)와 Y-up 회전 두 변형을 모두 추론해 비교
- 산출: 변형별 raw 출력 LAS + (원본+보충점 class12) LAS + 3면 전후 비교 PNG
"""
import numpy as np
import open3d as o3d
import laspy
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import restore_standalone as rs  # C++/CUDA 확장 대체 셰임 등록 + load_las 재사용

STEM = "RR_07_01_EA_028"
HOLE_THR = 0.012  # 12mm: 이 거리 이상 떨어진 완성점만 '결손 보충'으로 인정


def build_model(device):
    from models.PoinTr import PoinTr
    from easydict import EasyDict
    import yaml
    with open("cfgs/ShapeNet55_models/PoinTr.yaml") as f:
        config = EasyDict(yaml.safe_load(f))
    model = PoinTr(config.model).to(device)
    ckpt = torch.load("pretrained/pointr_shapenet.pth", map_location=device, weights_only=False)
    raw = ckpt.get("base_model", ckpt.get("model", ckpt))
    model.load_state_dict({k.replace("module.", "", 1): v for k, v in raw.items()}, strict=False)
    model.eval()
    return model


def complete(model, device, pts_norm):
    tmp = o3d.geometry.PointCloud()
    tmp.points = o3d.utility.Vector3dVector(pts_norm)
    inp = np.asarray(tmp.farthest_point_down_sample(2048).points)
    with torch.no_grad():
        _, fine = model(torch.from_numpy(inp).float().unsqueeze(0).to(device))
    return fine.squeeze(0).cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    pcd, header = rs.load_las(f"{STEM}.las")
    orig = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    centroid = orig.mean(axis=0)
    norm = orig - centroid
    scale = np.max(np.linalg.norm(norm, axis=1))
    norm = norm / scale
    print(f"원본 {len(orig)}점, 스케일 {scale:.3f}m")

    model = build_model(device)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # 자세 변형: raw(Z-up 그대로) / yup(ShapeNet 정준: X축 -90° 회전, (x,y,z)->(x,z,-y))
    variants = {
        "raw": (norm, lambda p: p),
        "yup": (norm[:, [0, 2, 1]] * np.array([1, 1, -1]),
                lambda p: (p * np.array([1, 1, -1]))[:, [0, 2, 1]]),
    }
    results = {}
    for name, (pts_in, invert) in variants.items():
        fine = complete(model, device, pts_in)
        fine_world = invert(fine) * scale + centroid
        # 결손 보충점 = 원본 표면에서 HOLE_THR 이상 떨어진 완성점
        infill, infill_c = [], []
        for pt in fine_world:
            _, idx, d = kdtree.search_knn_vector_3d(pt, 1)
            if np.sqrt(d[0]) > HOLE_THR:
                infill.append(pt)
                infill_c.append(cols[idx[0]])
        infill = np.array(infill).reshape(-1, 3)
        results[name] = (fine_world, infill, np.array(infill_c).reshape(-1, 3))
        print(f"[{name}] 완성 출력 {len(fine_world)}점 -> 보충점 {len(infill)}점")

        # raw 출력 저장
        h = laspy.LasHeader(point_format=header.point_format, version=header.version)
        h.offsets, h.scales = header.offsets, header.scales
        raw_las = laspy.LasData(h)
        raw_las.x, raw_las.y, raw_las.z = fine_world.T
        raw_las.write(f"{STEM}_pointr_raw_{name}.las")

    # 보충점이 많은(=결손을 더 채운) 변형을 채택해 병합 저장
    best = max(results, key=lambda k: len(results[k][1]))
    fine_world, infill, infill_c = results[best]
    print("채택 변형:", best)
    if len(infill):
        pts_all = np.vstack([orig, infill])
        col_all = np.vstack([cols, infill_c])
        cls = np.hstack([np.full(len(orig), 1, np.uint8), np.full(len(infill), 12, np.uint8)])
    else:
        pts_all, col_all = orig, cols
        cls = np.full(len(orig), 1, np.uint8)
    h = laspy.LasHeader(point_format=header.point_format, version=header.version)
    h.offsets, h.scales = header.offsets, header.scales
    out = laspy.LasData(h)
    out.x, out.y, out.z = pts_all.T
    out.red, out.green, out.blue = (col_all.T * 65535).astype(np.uint16)
    out.classification = cls
    out.write(f"{STEM}_pointr_restored.las")
    print(f"저장: {STEM}_pointr_restored.las ({len(pts_all)}점, 보충 {int((cls==12).sum())}점)")

    # 3면 전후 비교: 원본(회색) + 변형별 보충점(빨강) / raw 출력 전체(파랑)
    fig, axes = plt.subplots(3, 3, figsize=(15, 13), facecolor="0.13")
    views = [(0, 2, "XZ"), (1, 2, "YZ"), (0, 1, "XY")]
    sets = [("original", orig, None), *[(f"{n}: raw 8k output", results[n][0], results[n][1]) for n in variants]]
    for r, (label, cloud, infill_pts) in enumerate(sets):
        for c, (a, b, vn) in enumerate(views):
            ax = axes[r][c]
            ax.set_facecolor("0.13")
            ax.scatter(orig[::20, a], orig[::20, b], s=0.4, c="0.6", linewidths=0)
            if r > 0:
                ax.scatter(cloud[:, a], cloud[:, b], s=2.2, c="#3b78d8", linewidths=0)
                if infill_pts is not None and len(infill_pts):
                    ax.scatter(infill_pts[:, a], infill_pts[:, b], s=3.0, c="#e04338", linewidths=0)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{label} ({vn})", color="w", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"diag_{STEM}_pointr.png", dpi=110)
    print(f"진단 이미지: diag_{STEM}_pointr.png")


if __name__ == "__main__":
    main()
