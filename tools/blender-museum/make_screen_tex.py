"""Build the folding-screen textures for Room B.

T_RoomB_ScreenPainting.png  4096x1536 — 8 panels x 512 px. The user's Silla cavalry illustration is
                            scaled to full height and centred; the paper on both sides is extended with the
                            illustration's own background tone so the picture reads as one continuous scroll.
                            The right-most panel carries a vertical Hanja title and a red seal, like the reference.
T_RoomB_Brocade.png         512x512 tileable gold brocade for the mounting border.
"""
import sys, math, random
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops

import os
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "roomB_screen_source.png")   # the user's Silla cavalry illustration
OUT = r"C:\Users\<USER>\AppData\Local\Temp\museum_fbx"
W, H, PANELS = 4096, 1536, 8
random.seed(7)

img = Image.open(SRC).convert("RGB")
w0, h0 = img.size
img = img.crop((0, 0, w0, h0 - 75))          # drop the lens button strip along the bottom edge
img = img.crop((6, 0, img.width - 6, img.height))
h0 = img.height
scale = H / h0
img = img.resize((int(img.width * scale), H), Image.LANCZOS)
iw = img.width
x0 = (W - iw) // 2

canvas = Image.new("RGB", (W, H))
# background tone: row averages of the illustration's edges, blurred vertically so no streaks survive
def tone(strip):
    col = strip.resize((1, H), Image.BOX).resize((64, H), Image.NEAREST).filter(ImageFilter.GaussianBlur(70))
    return col.resize((1, H), Image.BOX)
left, right = tone(img.crop((0, 0, 60, H))), tone(img.crop((iw - 60, 0, iw, H)))
lp, rp = left.load(), right.load()
cp = canvas.load()
for y in range(H):
    for x in range(0, x0):
        cp[x, y] = lp[0, y]
    for x in range(x0 + iw, W):
        cp[x, y] = rp[0, y]
# soft cloud variation so the extended paper is not flat
noise = Image.effect_noise((W // 8, H // 8), 48).resize((W, H), Image.BILINEAR).filter(ImageFilter.GaussianBlur(40))
noise = noise.point(lambda v: 128 + (v - 128) * 0.35)
canvas = ImageChops.add(canvas, Image.merge("RGB", (noise, noise, noise)), scale=1.0, offset=-128)
# the distant army continues onto the side panels: mirrored copy of the crowd band, fading out at its foot
BAND = int(H * 0.36)
flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
fade = Image.new("L", (iw, BAND), 255)
fd = ImageDraw.Draw(fade)
for k in range(140):
    fd.line((0, BAND - 140 + k, iw, BAND - 140 + k), fill=int(255 * (1 - k / 140)))
band = flipped.crop((0, 0, iw, BAND))
canvas.paste(band, (x0 - iw, 0), fade)          # left side (only the part inside the canvas lands)
canvas.paste(band, (x0 + iw, 0), fade)          # right side
canvas.paste(img, (x0, 0))
# blend the seams with a 160 px ramp (stretch the edge columns outward, then alpha-fade them)
RAMP = 160
for side in ("L", "R"):
    if side == "L":
        strip = img.crop((0, 0, 24, H)).resize((RAMP, H), Image.BILINEAR).filter(ImageFilter.GaussianBlur(18))
        px = x0 - RAMP
        alpha = Image.linear_gradient("L").rotate(90, expand=True).resize((RAMP, H))       # 0 -> 255 left to right
    else:
        strip = img.crop((iw - 24, 0, iw, H)).resize((RAMP, H), Image.BILINEAR).filter(ImageFilter.GaussianBlur(18))
        px = x0 + iw
        alpha = Image.linear_gradient("L").rotate(-90, expand=True).resize((RAMP, H))      # 255 -> 0
    base = canvas.crop((px, 0, px + RAMP, H))
    canvas.paste(Image.composite(strip, base, alpha), (px, 0))
# paper grain over everything
grain = Image.effect_noise((W, H), 14).point(lambda v: 128 + (v - 128) * 0.5)
canvas = ImageChops.add(canvas, Image.merge("RGB", (grain, grain, grain)), scale=1.0, offset=-128)

# title + seal on the right-most panel
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype(r"C:\Windows\Fonts\batang.ttc", 100)
    small = ImageFont.truetype(r"C:\Windows\Fonts\batang.ttc", 54)
except Exception:
    font = small = ImageFont.load_default()
title = "新羅騎兵圖"
px, py = W - 512 + 200, 620          # below the army band
for ch in title:
    draw.text((px, py), ch, font=font, fill=(28, 24, 22))
    py += 112
py += 30
for ch in "慶州":
    draw.text((px + 24, py), ch, font=small, fill=(50, 44, 40))
    py += 60
# red seal with a lighter mark inside
sx, sy, ss = px + 2, py + 30, 104
draw.rectangle((sx, sy, sx + ss, sy + ss), fill=(178, 38, 36))
draw.rectangle((sx + 9, sy + 9, sx + ss - 9, sy + ss - 9), outline=(232, 200, 190), width=6)
draw.text((sx + 20, sy + 12), "羅", font=ImageFont.truetype(r"C:\Windows\Fonts\batang.ttc", 84), fill=(236, 210, 200))

canvas.save(OUT + r"\T_RoomB_ScreenPainting.png")
print("painting", canvas.size, "illustration at x", x0, "..", x0 + iw, "-> panels", round(x0 / 512, 2), "..", round((x0 + iw) / 512, 2))

# ---------------------------------------------------------------- brocade (tileable)
S = 512
br = Image.new("RGB", (S, S), (176, 140, 74))
d = ImageDraw.Draw(br)
step = 64
for k in range(-S, 2 * S, step):
    d.line((k, 0, k + S, S), fill=(150, 112, 52), width=5)
    d.line((k, S, k + S, 0), fill=(150, 112, 52), width=5)
for yy in range(step // 2, S, step):
    for xx in range(step // 2, S, step):
        d.ellipse((xx - 9, yy - 9, xx + 9, yy + 9), fill=(214, 182, 104))
        d.ellipse((xx - 4, yy - 4, xx + 4, yy + 4), fill=(150, 112, 52))
g = Image.effect_noise((S, S), 10).point(lambda v: 128 + (v - 128) * 0.6)
br = ImageChops.add(br, Image.merge("RGB", (g, g, g)), scale=1.0, offset=-128)
br.save(OUT + r"\T_RoomB_Brocade.png")
print("brocade", br.size)
