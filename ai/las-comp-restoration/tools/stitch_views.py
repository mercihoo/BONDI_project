#!/usr/bin/env python3
"""캡처된 칸 PNG(<이름>_a.png 원본, <이름>_b.png 결과, 있으면 <이름>_c.png 참조)를 방향별로 나란히 붙이고, 4 방향을 2×2 한 장으로도 만든다.

    python tools/stitch_views.py renders <prefix> front side back top
→ renders/<prefix>_front.png … renders/<prefix>_sheet.png

칸이 셋이면(viewer3.html 의 a/b/c) 머리글도 셋으로 쓴다. 머리글은 환경변수 STITCH_HEADER 로 바꿀 수 있다.
"""
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

d, prefix, names = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
KO = {"front": "정면", "side": "측면", "back": "후면", "top": "위에서", "bottom": "아래에서"}
HEAD2 = "왼쪽 손상 원본(질감) · 오른쪽 LaS-Comp 결과(음영 메시)"
HEAD3 = "왼쪽 손상 원본 · 가운데 LaS-Comp 결과(참조 조건) · 오른쪽 팀 기하 복원본(조건 이미지 출처)"
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 22)
except Exception:
    try: font = ImageFont.truetype("/mnt/c/Windows/Fonts/malgun.ttf", 22)
    except Exception: font = ImageFont.load_default()

GAP = 12
pairs = []
for n in names:
    tiles = [d / f"{prefix}_{n}_{t}.png" for t in ("a", "b", "c")]
    tiles = [p for p in tiles if p.exists()]
    if len(tiles) < 2:
        print("없음:", n); continue
    ims = [Image.open(p).convert("RGB") for p in tiles]
    h = max(i.height for i in ims); w = sum(i.width for i in ims) + GAP * (len(ims) - 1)
    im = Image.new("RGB", (w, h + 40), (28, 28, 32))
    x = 0
    for t in ims:
        im.paste(t, (x, 40)); x += t.width + GAP
    head = os.environ.get("STITCH_HEADER") or (HEAD3 if len(ims) == 3 else HEAD2)
    ImageDraw.Draw(im).text((10, 8), f"{KO.get(n, n)} · {head}", fill=(235, 235, 235), font=font)
    out = d / f"{prefix}_{n}.png"; im.save(out); pairs.append(im); print("저장:", out, im.size)

if len(pairs) >= 2:
    cols = 2 if len(pairs) > 1 else 1
    rows = (len(pairs) + cols - 1) // cols
    w, h = pairs[0].width, pairs[0].height
    sheet = Image.new("RGB", (cols * w + (cols - 1) * 10, rows * h + (rows - 1) * 10), (18, 18, 20))
    for i, im in enumerate(pairs):
        sheet.paste(im.resize((w, h)), ((i % cols) * (w + 10), (i // cols) * (h + 10)))
    out = d / f"{prefix}_sheet.png"; sheet.save(out); print("저장:", out, sheet.size)
