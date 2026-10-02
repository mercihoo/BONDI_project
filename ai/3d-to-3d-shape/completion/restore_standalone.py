# -*- coding: utf-8 -*-
import sys, types
import numpy as np
import open3d as o3d
import laspy
import torch
import torch.nn as nn
import copy

# ---------------------------------------------------------------------------
# C++/CUDA 확장 모듈(pointnet2_ops, chamfer 등)을 순수 PyTorch 구현으로 대체
# models 패키지 import 전에 sys.modules에 등록해야 함
# ---------------------------------------------------------------------------
def _furthest_point_sample(xyz, npoint):
    # xyz: (B, N, 3) float -> (B, npoint) int32
    B, N, _ = xyz.shape
    idx = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), 1e10, device=xyz.device)
    farthest = torch.zeros(B, dtype=torch.long, device=xyz.device)
    batch = torch.arange(B, device=xyz.device)
    for i in range(npoint):
        idx[:, i] = farthest
        centroid = xyz[batch, farthest, :].unsqueeze(1)
        d = torch.sum((xyz - centroid) ** 2, -1)
        dist = torch.minimum(dist, d)
        farthest = torch.max(dist, -1)[1]
    return idx.int()

def _gather_operation(features, idx):
    # features: (B, C, N), idx: (B, M) -> (B, C, M)
    idx = idx.long().unsqueeze(1).expand(-1, features.shape[1], -1)
    return torch.gather(features, 2, idx)

def _three_nn(unknown, known):
    # unknown: (B, n, 3), known: (B, m, 3) -> dist (B, n, 3), idx (B, n, 3)
    d = torch.cdist(unknown, known)
    dist, idx = torch.topk(d, 3, dim=-1, largest=False)
    return dist, idx.int()

def _three_interpolate(feats, idx, weight):
    # feats: (B, C, m), idx: (B, n, 3), weight: (B, n, 3) -> (B, C, n)
    B, C, m = feats.shape
    n = idx.shape[1]
    idx = idx.long().reshape(B, 1, n * 3).expand(-1, C, -1)
    g = torch.gather(feats, 2, idx).reshape(B, C, n, 3)
    return (g * weight.unsqueeze(1)).sum(-1)

def _grouping_operation(features, idx):
    # features: (B, C, N), idx: (B, npoint, k) -> (B, C, npoint, k)
    B, C, N = features.shape
    npoint, k = idx.shape[1], idx.shape[2]
    idx = idx.long().reshape(B, 1, npoint * k).expand(-1, C, -1)
    return torch.gather(features, 2, idx).reshape(B, C, npoint, k)

def _ball_query(radius, nsample, xyz, new_xyz):
    d = torch.cdist(new_xyz, xyz)
    idx = torch.topk(d, min(nsample, d.shape[-1]), dim=-1, largest=False)[1]
    first = idx[..., :1].expand_as(idx)
    idx = torch.where(torch.gather(d, 2, idx) <= radius, idx, first)
    return idx.int()

_pn2 = types.ModuleType('pointnet2_ops')
_pn2_utils = types.ModuleType('pointnet2_ops.pointnet2_utils')
_pn2_utils.furthest_point_sample = _furthest_point_sample
_pn2_utils.gather_operation = _gather_operation
_pn2_utils.three_nn = _three_nn
_pn2_utils.three_interpolate = _three_interpolate
_pn2_utils.grouping_operation = _grouping_operation
_pn2_utils.ball_query = _ball_query
_pn2.pointnet2_utils = _pn2_utils
sys.modules['pointnet2_ops'] = _pn2
sys.modules['pointnet2_ops.pointnet2_utils'] = _pn2_utils

# 손실 함수용 C 확장은 추론 시 호출되지 않으므로 빈 모듈로 대체
for _name in ('chamfer', 'emd', 'gridding', 'gridding_distance', 'cubic_feature_sampling'):
    sys.modules.setdefault(_name, types.ModuleType(_name))
# ---------------------------------------------------------------------------

def load_las(las_path):
    las = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        colors = np.vstack((las.red, las.green, las.blue)).transpose()
        colors = colors / 65535.0 if colors.max() > 255 else colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([0.7, 0.7, 0.7])
    return pcd, las.header

def run_completion_inference(input_las_path, output_las_path, ckpt_path="pretrained/pointr_shapenet.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")

    # 1. LAS 데이터 로드
    pcd, ref_header = load_las(input_las_path)
    orig_points = np.asarray(pcd.points)
    orig_colors = np.asarray(pcd.colors)
    print(f"원본 포인트 수: {len(orig_points)}")

    # 2. 정규화
    centroid = np.mean(orig_points, axis=0)
    norm_points = orig_points - centroid
    scale = np.max(np.sqrt(np.sum(norm_points ** 2, axis=1)))
    norm_points /= scale

    # 3. 2048개 포인트 FPS 샘플링 (Open3D 내장 함수 사용)
    temp_pcd = o3d.geometry.PointCloud()
    temp_pcd.points = o3d.utility.Vector3dVector(norm_points)
    print("FPS 샘플링 중 (2048 points)...")
    down_pcd = temp_pcd.farthest_point_down_sample(2048)
    input_pts = np.asarray(down_pcd.points)

    input_tensor = torch.from_numpy(input_pts).float().unsqueeze(0).to(device)

    # 4. 모델 로드 및 추론
    from models.PoinTr import PoinTr
    from easydict import EasyDict
    import yaml

    with open("cfgs/ShapeNet55_models/PoinTr.yaml", 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    model = PoinTr(config.model).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_dict = ckpt.get('base_model', ckpt.get('model', ckpt))
    base_dict = {k.replace("module.", "", 1): v for k, v in raw_dict.items()}
    incompat = model.load_state_dict(base_dict, strict=False)
    print(f"체크포인트 로드: missing={len(incompat.missing_keys)}, unexpected={len(incompat.unexpected_keys)}")
    if incompat.missing_keys:
        print("  missing 예시:", incompat.missing_keys[:5])
    model.eval()

    print("추론 중...")
    with torch.no_grad():
        coarse_pts, fine_pts = model(input_tensor)
        completed_norm = fine_pts.squeeze(0).cpu().numpy()
    print(f"복원 결과 포인트 수: {len(completed_norm)}")

    # 5. 역정규화
    completed_points = (completed_norm * scale) + centroid

    # 6. 결손 영역만 필터링 (KNN 거리)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    infill_points = []
    infill_colors = []
    hole_threshold = 0.012

    for pt in completed_points:
        [_, idx, dists] = kdtree.search_knn_vector_3d(pt, 1)
        if np.sqrt(dists[0]) > hole_threshold:
            infill_points.append(pt)
            infill_colors.append(orig_colors[idx[0]])

    print(f"결손 영역 보충 포인트 수: {len(infill_points)}")

    # 7. 원본 + 복원 결합
    if len(infill_points) > 0:
        infill_points = np.array(infill_points)
        infill_colors = np.array(infill_colors)

        restored_points = np.vstack((orig_points, infill_points))
        restored_colors = np.vstack((orig_colors, infill_colors))
        classes = np.hstack((
            np.full(len(orig_points), 1, dtype=np.uint8),
            np.full(len(infill_points), 12, dtype=np.uint8)
        ))
    else:
        restored_points = orig_points
        restored_colors = orig_colors
        classes = np.full(len(orig_points), 1, dtype=np.uint8)

    # 8. LAS 저장
    out_header = laspy.LasHeader(point_format=ref_header.point_format, version=ref_header.version)
    out_header.offsets = ref_header.offsets
    out_header.scales = ref_header.scales

    out_las = laspy.LasData(out_header)
    out_las.x = restored_points[:, 0]
    out_las.y = restored_points[:, 1]
    out_las.z = restored_points[:, 2]

    if len(restored_colors) > 0:
        out_las.red = (restored_colors[:, 0] * 65535).astype(np.uint16)
        out_las.green = (restored_colors[:, 1] * 65535).astype(np.uint16)
        out_las.blue = (restored_colors[:, 2] * 65535).astype(np.uint16)

    out_las.classification = classes
    out_las.write(output_las_path)
    print(f"복원 파일 저장 완료: {output_las_path} (총 {len(restored_points)} 포인트)")

if __name__ == "__main__":
    run_completion_inference("RR_07_01_PA_033.las", "RR_07_01_PA_033_pointr_restored.las")
