import json, re, pathlib
H = pathlib.Path(".")
recs = json.loads((H/"meta/artifacts.json").read_text("utf-8"))
have = {p.stem for p in (H/"trellis3d").glob("*.glb")}
seen, missing, used = {}, [], set()
for r in recs:
    num = (r.get("소장품번호") or "").strip()
    base = re.sub(r"[^0-9A-Za-z가-힣]+", "_", num).strip("_")
    f = r["file"]
    if seen.get(base) in (None, f):
        seen[base] = f; s = base
    else:
        k = 2
        while seen.get(f"{base}_{k}") not in (None, f): k += 1
        seen[f"{base}_{k}"] = f; s = f"{base}_{k}"
    used.add(s)
    if s not in have:
        missing.append((num, f))
print(f"기록 {len(recs)}건 · GLB {len(have)}개 · 매칭 {len(used & have)}건")
print(f"빠진 것 {len(missing)}건" + (":" if missing else " — 전부 완료"))
for n, f in missing: print(f"  {n}  {f}")
o = have - used
print("기록에 없는 GLB:", o if o else "없음")
