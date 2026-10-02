#!/usr/bin/env python3
"""결과를 눈으로 볼 수 있게 PNG 로 그린다 — GPU·디스플레이 없이(matplotlib 점 산포).
입력(손상) · LaS-Comp 결과 · (있으면) 팀 복원본을 같은 시점 4 방향으로 나란히.
투창(뚫린 창)이 살아남았는지 보는 것이 목적이라 점 크기를 작게, 깊이로 명암을 준다.

    python tools/render_views.py results/<라벨> samples/artifacts/gupdari71489/partial.ply [artifacts/gupdari71489/team_restored_model.glb] out.png
"""
import sys
from pathlib import Path
import numpy as np, trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_completion import load_geometry, load_points

N = 120_000

def surface_pts(path: Path):
    m = load_geometry(path)
    if m is not None:
        p, _ = trimesh.sample.sample_surface(m, N); return np.asarray(p)
    p, _ = load_points(path, N)
    if len(p) > N: p = p[np.random.default_rng(0).choice(len(p), N, replace=False)]
    return p

def normalize(p):
    c = (p.min(0) + p.max(0)) / 2; s = (p.max(0) - p.min(0)).max()
    return (p - c) / s

res_dir, partial = Path(sys.argv[1]), Path(sys.argv[2])
team = Path(sys.argv[3]) if len(sys.argv) > 4 else None
out = Path(sys.argv[-1])

cols = [("입력 (손상)", surface_pts(partial))]
mesh_file = res_dir / "output_mesh.glb"
res = surface_pts(mesh_file)
# LaS-Comp 은 내부에서 yz 를 바꿨다가 저장 시 원좌표로 되돌린다(--normalize-partial 도 되돌림).
# 그래도 프레임이 어긋나 보이면 yz 교환본으로 비교한다 — 두 후보 중 입력과 bbox 가 더 맞는 쪽.
def bbox_gap(a, b): return np.abs((a.max(0) - a.min(0)) - (b.max(0) - b.min(0))).sum()
res_sw = res.copy(); res_sw[:, [1, 2]] = res_sw[:, [2, 1]]
if bbox_gap(res_sw, cols[0][1]) < bbox_gap(res, cols[0][1]): res = res_sw
cols.append(("LaS-Comp 결과", res))
if team is not None and team.exists():
    cols.append(("팀 복원본 (기하 방식)", surface_pts(team)))

# 위 축: 입력에서 가장 긴 축을 위로 세운다
up = int(np.argmax(cols[0][1].max(0) - cols[0][1].min(0)))
def to_view(p):
    q = normalize(p)
    axes = [i for i in range(3) if i != up]
    return q[:, axes[0]], q[:, axes[1]], q[:, up]   # (가로1, 가로2, 세로)

views = [("정면", 0), ("측면", 90), ("후면", 180), ("위에서", None)]
fig, ax = plt.subplots(len(views), len(cols), figsize=(4.2 * len(cols), 4.2 * len(views)), squeeze=False)
for j, (title, p) in enumerate(cols):
    a, b, h = to_view(p)
    for i, (vname, deg) in enumerate(views):
        A = ax[i][j]
        if deg is None:
            x, y, depth = a, b, h
        else:
            t = np.deg2rad(deg); x = a * np.cos(t) + b * np.sin(t); depth = -a * np.sin(t) + b * np.cos(t); y = h
        order = np.argsort(depth)
        A.scatter(x[order], y[order], c=depth[order], s=0.15, cmap="gray", linewidths=0)
        A.set_aspect("equal"); A.set_xlim(-0.55, 0.55); A.set_ylim(-0.55, 0.55); A.axis("off")
        if i == 0: A.set_title(title, fontsize=13)
        if j == 0: A.text(-0.53, 0.48, vname, fontsize=11)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
fig.suptitle(res_dir.name, fontsize=12)
fig.tight_layout()
fig.savefig(out, dpi=110)
print("저장:", out, "| 열:", [c[0] for c in cols])
