"""PIPELINE.md 에 넣을 그림을 만든다."""
from __future__ import annotations

import csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Ellipse, FancyArrowPatch

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
H = Path(__file__).resolve().parent
FIG = H / "work" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
META = {r["file"]: r for r in json.loads((H / "meta/artifacts.json").read_text("utf-8"))}


def summ(tag):
    f = H / "work/profiles" / f"step1_summary_{tag}.csv"
    d = {}
    for r in csv.DictReader(f.open(encoding="utf-8-sig")):
        if r["ok"] == "True" and r["cls"] == "plain":
            d[r["num"]] = r
    return d


def maskof(stem, tag):
    p = H / "work/silhouettes" / f"{stem}__{tag}.png"
    return np.asarray(Image.open(p).convert("L")) > 127 if p.exists() else None


# ── 1. 규칙 vs 학습, 같은 사진 위에 겹쳐 ────────────────────────────
A, B = summ("rule"), summ("rembg")
pick = sorted(A, key=lambda k: -float(A[k]["foot_bad"]))[:4]
fig, ax = plt.subplots(1, 4, figsize=(15, 4.6))
for a_, num in zip(ax, pick):
    r = A[num]; stem = Path(r["file"]).stem
    im = Image.open(H / "images" / r["file"]).convert("RGB")
    s = min(1.0, 1400 / max(im.size))
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    a_.imshow(np.asarray(im))
    for tag, c, lw in (("rule", "#ff2d2d", 1.4), ("rembg", "#00d2ff", 1.4)):
        m = maskof(stem, tag)
        if m is not None:
            a_.contour(m, [0.5], colors=c, linewidths=lw)
    a_.set_title(f"{num}\n굽오염 규칙 {float(A[num]['foot_bad']):.0f}%  →  "
                 f"rembg {float(B[num]['foot_bad']):.0f}%", fontsize=10)
    a_.axis("off")
fig.suptitle("빨강 = 손으로 짠 규칙 (그림자를 삼킨다)      파랑 = 학습 모델 rembg",
             fontsize=12, y=0.02)
fig.tight_layout(); fig.savefig(FIG / "fig_rule_vs_learned.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ── 2. 왜 폭이 좁게 나오나 — 원근 ───────────────────────────────
fig = plt.figure(figsize=(14, 4.4))
g = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1])

a0 = fig.add_subplot(g[0]); a0.set_aspect("equal"); a0.axis("off")
a0.set_xlim(-1.9, 2.3); a0.set_ylim(-0.5, 2.5)
D, Hh, t = 1.6, 1.7, np.radians(28)
a0.add_patch(Ellipse((0, Hh), D, D * np.sin(t), fc="#d8d2c6", ec="#6b6255", lw=1.6))
a0.plot([-D/2, -D*0.22, -D*0.34, D*0.34, D*0.22, D/2],
        [Hh, 0.55, 0.12, 0.12, 0.55, Hh], color="#6b6255", lw=1.6)
a0.plot([-D/2, D/2], [Hh, Hh], color="#6b6255", lw=1.0, ls=":")
top = Hh + D * np.sin(t) / 2
a0.add_patch(FancyArrowPatch((-1.35, 0.12), (-1.35, Hh), arrowstyle="<->", mutation_scale=11,
                             color="#2f7d32", lw=1.7))
a0.text(-1.45, (Hh + 0.12)/2, "실제 높이 H", rotation=90, va="center", ha="right",
        color="#2f7d32", fontsize=10)
a0.add_patch(FancyArrowPatch((1.85, 0.12), (1.85, top), arrowstyle="<->", mutation_scale=11,
                             color="#c62828", lw=1.7))
a0.text(1.95, (top + 0.12)/2, "사진에서 보이는 세로 A", rotation=90, va="center",
        color="#c62828", fontsize=10)
a0.plot([-D/2, 1.85], [top, top], color="#c62828", lw=0.8, ls="--")
a0.text(0, 2.34, "A = H·cos t + D·sin t   >   H", ha="center", fontsize=11.5, color="#c62828")
a0.text(0, -0.42, "아가리가 타원으로 보여 뒤쪽 테두리가 위로 솟는다\n"
                  "→ A 를 H 로 나누면 cm/px 가 작아져 폭이 좁게 나온다",
        ha="center", fontsize=9.5, color="#444")
a0.set_title("(a) 왜 실루엣이 좁게 나오나", fontsize=11)

a1 = fig.add_subplot(g[1])
for tag, S, c, mk in (("규칙 v3", A, "#ff2d2d", "o"), ("rembg", B, "#0098c8", "s")):
    x, y = [], []
    for k, r in S.items():
        if r["mouth_cm"] not in ("", "None") and r["wmax_cm"] not in ("", "None"):
            x.append(float(r["mouth_cm"])); y.append(float(r["wmax_cm"]))
    a1.scatter(x, y, s=26, alpha=.75, c=c, marker=mk, label=tag, edgecolors="none")
lim = [5, 22]
a1.plot(lim, lim, "k--", lw=1, label="일치선")
a1.set_xlabel("박물관 기록 입지름 (cm)"); a1.set_ylabel("실루엣 최대폭 (cm)")
a1.set_xlim(lim); a1.set_ylim(lim); a1.legend(fontsize=9); a1.grid(alpha=.25)
a1.set_title("(b) 두 방식 모두 일치선 아래 = 계통 오차", fontsize=11)

a2 = fig.add_subplot(g[2])
for tag, S, c in (("규칙 v3", A, "#ff2d2d"), ("rembg", B, "#0098c8")):
    v = [float(r["elev"]) for r in S.values() if r["elev"] not in ("", "None")]
    a2.hist(v, bins=np.arange(0, 40, 2.5), alpha=.6, color=c, label=f"{tag} (n={len(v)})")
a2.set_xlabel("역산한 촬영 앙각 (도)"); a2.set_ylabel("유물 수")
a2.legend(fontsize=9); a2.grid(alpha=.25)
a2.set_title("(c) 학습 모델 쪽이 훨씬 덜 흩어진다", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "fig_perspective.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ── 3. 뽑아낸 프로파일 ──────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
prof = []
for num, r in B.items():
    stem = Path(r["file"]).stem
    p = H / "work/profiles" / f"{stem}__rembg.npz"
    if not p.exists():
        continue
    d = np.load(p); z, rr = d["z"], d["r"]
    if not np.isfinite(d["cm_per_px"]) or z.max() <= 0:
        continue
    ax[0].plot(rr, z, lw=.9, alpha=.55)
    prof.append(np.interp(np.linspace(0, 1, 200), z / z.max(), rr / z.max()))
ax[0].set_xlabel("반지름 (cm)"); ax[0].set_ylabel("높이 (cm)")
ax[0].set_title(f"(a) 실측 축척 프로파일 {len(prof)}점", fontsize=11); ax[0].grid(alpha=.25)
P = np.array(prof); u = np.linspace(0, 1, 200)
ax[1].fill_betweenx(u, np.percentile(P, 10, 0), np.percentile(P, 90, 0),
                    color="#7fb3d5", alpha=.45, label="10~90 백분위")
ax[1].plot(np.median(P, 0), u, color="#14406b", lw=2.4, label="중앙 형태")
ax[1].set_xlabel("반지름 / 높이"); ax[1].set_ylabel("높이 (정규화)")
ax[1].legend(fontsize=9); ax[1].grid(alpha=.25)
ax[1].set_title("(b) 높이로 정규화 — 갈래의 공통 형태", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "fig_profiles.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ── 4. 아가리 회귀 ─────────────────────────────────────────────
P2 = [r for r in META.values() if r.get("class") == "plain"
      and r.get("height_cm") and r.get("mouth_cm")]
h = np.array([r["height_cm"] for r in P2]); d_ = np.array([r["mouth_cm"] for r in P2])
a, b = np.polyfit(h, d_, 1)
loo = []
for i in range(len(h)):
    m = np.ones(len(h), bool); m[i] = False
    aa, bb = np.polyfit(h[m], d_[m], 1)
    loo.append(abs(aa * h[i] + bb - d_[i]))
loo = float(np.median(loo))
fig, ax = plt.subplots(figsize=(6.4, 4.8))
xs = np.linspace(6, 19, 50)
ax.fill_between(xs, a*xs+b-loo, a*xs+b+loo, color="#b9d6ea", alpha=.55, label=f"LOO 중앙오차 ±{loo:.2f}cm")
ax.plot(xs, a*xs+b, color="#14406b", lw=2, label=f"입지름 = {a:.3f}·높이 + {b:.2f}")
ax.scatter(h, d_, s=34, c="#14406b", zorder=3, label=f"기록 {len(h)}점")
ax.scatter([15.0], [a*15+b], s=150, marker="*", c="#e8a33d", zorder=4,
           edgecolors="#7a5510", label=f"71489 예측 {a*15+b:.2f}cm")
ax.scatter([15.0], [14.81], s=110, marker="X", c="#c62828", zorder=4,
           label="오늘 MCP 복원본 14.81cm")
ax.annotate("", xy=(15.0, a*15+b), xytext=(15.0, 14.81),
            arrowprops=dict(arrowstyle="<->", color="#c62828", lw=1.6))
ax.text(15.35, (a*15+b+14.81)/2, "1.83cm\n(11%)", color="#c62828", fontsize=10, va="center")
ax.set_xlabel("높이 (cm)"); ax.set_ylabel("입지름 (cm)")
ax.set_title(f"사진 없이 기록만으로 아가리 맞히기  (r² = {np.corrcoef(h,d_)[0,1]**2:.3f})", fontsize=11)
ax.legend(fontsize=8.5, loc="upper left"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(FIG / "fig_rim_regression.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("만든 그림:", *(p.name for p in sorted(FIG.glob("fig_*.png"))), sep="\n  ")
