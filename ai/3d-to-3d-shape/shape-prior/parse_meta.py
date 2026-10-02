"""e뮤지엄에서 긁어온 설명문을 표로 만든다 — 형상 모델의 '정답표'가 된다.

왜 이걸 먼저 하나
-----------------
사진만 있으면 프로파일은 뽑아도 **축척을 모른다.** 픽셀 높이 800이 14cm인지 19cm인지
알 수 없으면 여러 유물을 같은 자에 올릴 수 없다. `크기` 항목이 그 자를 준다.

그리고 `설명` 이 뜻밖에 중요하다. 한자로 구조가 그대로 적혀 있다.

    "臺脚은 2段 透窓 형식으로 上下段에 각각 3개의 透窓이 貫通됨"
      -> 대각은 이단투창, 위아래 각 3개

오늘 굽다리바리 71489 에서 기하학적으로 맞춘 값이 '위단 3개 / 아래단 4개' 였다.
이 문장들은 **그 적합을 검증할 독립 근거**다. 사진에서 잰 것과 문헌이 적은 것이 맞는지
대조할 수 있다.

`높이` 와 `현재높이` 를 구분하는 것도 중요하다 — `현재높이` 는 깨져서 줄어든 높이다.
그런 유물을 평균 형태에 넣으면 평균이 낮아진다.

출력: meta/artifacts.csv · meta/artifacts.json
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "meta" / "emuseum_raw.txt"

FIELDS = ["명칭", "다른명칭", "다른 명칭", "국적/시대", "출토지", "분류", "재질",
          "작가", "크기", "소장품번호", "설명"]
ZW = dict.fromkeys(map(ord, "​‌‍﻿\xa0"), None)   # 눈에 안 보이는 글자


def split_records(text: str) -> list[list[str]]:
    recs, cur = [], []
    for ln in text.splitlines():
        s = ln.translate(ZW).strip()
        if not s:
            continue
        if set(s) == {"-"}:                      # 길이가 제각각인 구분선
            if cur:
                recs.append(cur)
            cur = []
        else:
            cur.append(s)
    if cur:
        recs.append(cur)
    return recs


def parse_record(lines: list[str]) -> dict | None:
    if not lines or not lines[0].lower().endswith(".jpg"):
        return None
    rec = {"file": lines[0]}
    for s in lines[1:]:
        for f in FIELDS:
            if s.startswith(f):
                rec.setdefault(f.replace(" ", ""), s[len(f):].strip())
                break
        else:
            # 접두어가 빠진 줄 (실제로 한 건 있다) — 명칭으로 본다
            rec.setdefault("명칭", s)
    return rec


NUM = r"(\d+(?:\.\d+)?)"


def parse_size(s: str) -> dict:
    """'높이 15.0cm, 입지름 16cm, 받침지름 11.2cm' 를 숫자로."""
    out = {}
    for key, pat in (("height_cm", rf"(?<!현재)높이\s*{NUM}\s*cm"),
                     ("height_cm", rf"현재높이\s*{NUM}\s*cm"),
                     ("mouth_cm", rf"입지름\s*{NUM}\s*cm"),
                     ("base_cm", rf"받침지름\s*{NUM}\s*cm"),
                     ("foot_h_cm", rf"굽높이\s*{NUM}\s*cm")):
        m = re.search(pat, s)
        if m and key not in out:
            out[key] = float(m.group(1))
    out["height_is_current"] = bool(re.search(r"현재높이", s))   # 깨져서 줄어든 높이
    return out


HANJA_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "1": 1, "2": 2,
             "3": 3, "4": 4, "5": 5, "6": 6}


def parse_desc(s: str) -> dict:
    """한자 설명에서 구조를 읽는다. 透窓=투창 · 突帶=돌대 · 段=단"""
    out = {"has_tuchang": "透窓" in s, "has_doldae": "突帶" in s}
    m = re.search(r"([1-6一二三四五六])\s*段", s)
    if m:
        out["tiers"] = HANJA_NUM.get(m.group(1))
    elif "上下段" in s:
        out["tiers"] = 2
    # 각 단 몇 개인가
    m = re.search(r"각각\s*([0-9一二三四五六])\s*개의?\s*透窓", s)
    if m:
        out["per_tier"] = HANJA_NUM.get(m.group(1))
    else:
        # '4個의 長方形 1段 透窓' 처럼 사이에 말이 끼는 경우가 있다 — 붙어 있는 것만
        # 찾으면 13건 중 2건을 놓친다
        for pat in (r"透窓이?\s*([0-9一二三四五六])\s*[개個]",
                    r"([0-9一二三四五六])\s*[개個]의?\s*[^,.。]{0,14}?透窓"):
            m = re.search(pat, s)
            if m:
                out["per_tier"] = HANJA_NUM.get(m.group(1))
                break
    out["is_broken"] = bool(re.search(r"缺失|破損", s))
    return out


def klass(name: str) -> str:
    if "손잡이" in name or "把手" in name:
        return "handled"          # 손잡이가 실루엣을 깨뜨린다 — 1차 모델에서 제외
    if "접시" in name:
        return "dish"
    return "plain"


def main():
    recs = []
    for lines in split_records(RAW.read_text(encoding="utf-8", errors="replace")):
        r = parse_record(lines)
        if not r:
            continue
        r.update(parse_size(r.get("크기", "")))
        r.update(parse_desc(r.get("설명", "")))
        r["class"] = klass(r.get("명칭", "") + r.get("file", ""))
        r["has_image"] = (HERE / "images" / r["file"]).exists()
        recs.append(r)

    cols = ["file", "class", "명칭", "국적/시대", "출토지", "소장품번호",
            "height_cm", "height_is_current", "mouth_cm", "base_cm", "foot_h_cm",
            "tiers", "per_tier", "has_tuchang", "has_doldae", "is_broken",
            "has_image", "크기"]
    (HERE / "meta").mkdir(exist_ok=True)
    with (HERE / "meta" / "artifacts.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)
    (HERE / "meta" / "artifacts.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")

    n = len(recs)
    img = sum(r["has_image"] for r in recs)
    print(f"기록 {n}건 · 사진 있는 것 {img}건\n")
    print(f"{'분류':<10}{'건수':>5}{'높이있음':>8}{'입지름있음':>10}{'현재높이':>8}{'투창단수':>9}")
    print("-" * 52)
    for k in ("plain", "handled", "dish"):
        g = [r for r in recs if r["class"] == k]
        if not g:
            continue
        t = [r["tiers"] for r in g if r.get("tiers")]
        print(f"{k:<10}{len(g):>5}{sum('height_cm' in r for r in g):>8}"
              f"{sum('mouth_cm' in r for r in g):>10}"
              f"{sum(r['height_is_current'] for r in g):>8}"
              f"{(str(sorted(set(t))) if t else '-'):>9}")

    print(f"\n설명에서 구조를 읽어낸 것 {sum('tiers' in r for r in recs)}건")
    for r in recs:
        if r.get("tiers"):
            print(f"  {r['소장품번호']:<12} {r['class']:<8} {r['tiers']}단"
                  f"{' · 단당 ' + str(r['per_tier']) + '개' if r.get('per_tier') else ''}"
                  f"   높이 {r.get('height_cm', '?')}cm")

    tgt = [r for r in recs if "71489" in r.get("소장품번호", "")]
    if tgt:
        print(f"\n오늘 복원한 그 유물 — {tgt[0]['소장품번호']}")
        for k in ("height_cm", "mouth_cm", "height_is_current", "tiers", "per_tier", "크기"):
            print(f"  {k:<18} {tgt[0].get(k, '(없음)')}")


if __name__ == "__main__":
    main()
