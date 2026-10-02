"""사진 없이, **기록된 치수만으로** 71489 의 입지름을 맞출 수 있나?

이 스크립트를 쓴 이유
--------------------
형상 사전 폴더를 만든 목적은 하나였다 — 굽다리바리 71489 는 아가리가 둘레 전체로
깨져 있어 자기 자신에게서 아가리를 뽑을 수 없다. 박물관도 `입지름` 을 못 적었다.

그런데 사진에서 실루엣을 뽑기 전에 물어봤어야 할 게 있었다.
**그 한 숫자를 위해 정말 사진 27장이 필요한가?**

`meta/artifacts.csv` 에는 높이와 입지름을 둘 다 가진 굽다리바리가 20점 있다.
고배는 한 갈래의 기물이니 둘이 상관될 것이다. 그러면 회귀 한 줄이면 끝난다.

평가는 leave-one-out 으로 한다 — 20점으로 식을 만들어 그 20점을 맞히면 당연히 잘 맞는다.
하나씩 빼고 맞혀야 **실제로 쓸 때의 오차**가 나온다.
"""
import json, pathlib
import numpy as np
R = json.loads(pathlib.Path("meta/artifacts.json").read_text("utf-8"))
P = [r for r in R if r.get("class") == "plain"]
both = [(r["height_cm"], r["mouth_cm"], r["소장품번호"]) for r in P
        if r.get("height_cm") and r.get("mouth_cm")]
H = np.array([b[0] for b in both]); D = np.array([b[1] for b in both])
print(f"높이·입지름 둘 다 있는 굽다리바리 {len(both)}점\n")
print(f"  높이   {H.min():.1f} ~ {H.max():.1f} cm   (중앙 {np.median(H):.1f})")
print(f"  입지름 {D.min():.1f} ~ {D.max():.1f} cm   (중앙 {np.median(D):.1f})")
r = np.corrcoef(H, D)[0, 1]
a, b = np.polyfit(H, D, 1)
pred = a * H + b
res = D - pred
print(f"\n상관계수 r = {r:.3f}   (r^2 = {r*r:.3f})")
print(f"회귀식   입지름 = {a:.3f} x 높이 + {b:.3f}")
print(f"잔차     중앙 {np.median(np.abs(res)):.2f}cm · p90 {np.percentile(np.abs(res),90):.2f}cm"
      f" · 상대 {100*np.median(np.abs(res)/D):.1f}%")

# 하나씩 빼고 맞히기 (leave-one-out) — 실제로 쓸 때의 오차
loo = []
for i in range(len(H)):
    m = np.ones(len(H), bool); m[i] = False
    aa, bb = np.polyfit(H[m], D[m], 1)
    loo.append(abs(aa * H[i] + bb - D[i]))
loo = np.array(loo)
print(f"LOO      중앙 {np.median(loo):.2f}cm · p90 {np.percentile(loo,90):.2f}cm"
      f" · 상대 {100*np.median(loo/D):.1f}%")

# 비율 자체를 쓰면?
ratio = D / H
print(f"\n입지름/높이 비: 중앙 {np.median(ratio):.3f} · 표준편차 {ratio.std():.3f}"
      f" · 범위 {ratio.min():.3f}~{ratio.max():.3f}")
h49 = 15.0
print(f"\n71489 (높이 {h49}cm) 예측 입지름")
print(f"  회귀      {a*h49+b:.2f} cm")
print(f"  비율중앙  {np.median(ratio)*h49:.2f} cm")
print(f"  오늘 MCP 복원본이 만든 아가리: 직경 {2*0.495:.3f} (블렌더 단위)"
      f" = 높이 1.003 대비 {2*0.495/1.003:.3f} 배 -> {2*0.495/1.003*h49:.2f} cm")
