"""Build Room B (신라 전각 내부) in the running UE editor. Phases: import materials level lights cleanup save."""
import sys, math, json
from uemcp import *

FBX = "C:/Users/<USER>/AppData/Local/Temp/museum_fbx"
MESH_DIR = "/Game/Museum/Meshes"
MOCK_DIR = "/Game/Museum/Artifacts/Mock"
MAT_DIR = "/Game/Museum/Materials"
LEVEL = "/Game/Museum/Levels/L_Room_B"
HALL = "/Game/XRFramework/Levels/L_Museum"

SHELL = {  # asset -> (lightmap res, hull count)
    "SM_RoomB_Floor": (1024, 24), "SM_RoomB_Walls": (1024, 24), "SM_RoomB_Ceiling": (1024, 24),
    "SM_RoomB_Screen": (512, 12), "SM_RoomB_Platform": (512, 8), "SM_RoomB_Stand": (256, 12),
    "SM_RoomB_Table": (256, 12), "SM_RoomB_Candle": (64, 4)}
MOCKS = ["SM_Mock_Silla_Janggyeongho", "SM_Mock_Silla_Gobae", "SM_Mock_Silla_Gidae",
         "SM_Mock_Silla_Dangyeongho", "SM_Mock_Silla_Gobae_Tall"]
OLD_MESHES = ["SM_RoomB_Backdrop", "SM_RoomB_Dais", "SM_RoomB_Altar"]
OLD_MIS = ["MI_RoomB_Wall", "MI_RoomB_Ceiling", "MI_RoomB_Floor", "MI_RoomB_Stone", "MI_RoomB_Ring",
           "MI_RoomB_Timber", "MI_RoomB_Cove", "MI_RoomB_Gold"]

MS = "/Game/Fab/Megascans/Surfaces/"
TEX = {
    "plaster": (MS + "Rough_Cement_Plaster_peugfls0/High/peugfls0_tier_1/Textures/T_peugfls0_4K_B",
                MS + "Rough_Cement_Plaster_peugfls0/High/peugfls0_tier_1/Textures/T_peugfls0_4K_N"),
    "plaster_dmg": (MS + "Damaged_Wall_Plaster_uc2lddzcw/Medium/uc2lddzcw_tier_2/Textures/T_uc2lddzcw_2K_B",
                    MS + "Damaged_Wall_Plaster_uc2lddzcw/Medium/uc2lddzcw_tier_2/Textures/T_uc2lddzcw_2K_N"),
    "walnut": (MS + "Walnut_Veneer_tfdoebqc/Medium/tfdoebqc_tier_2/Textures/T_tfdoebqc_2K_B",
               MS + "Walnut_Veneer_tfdoebqc/Medium/tfdoebqc_tier_2/Textures/T_tfdoebqc_2K_N"),
    "fabric": (MS + "Furniture_Fabric_sjfvcgnc/Medium/sjfvcgnc_tier_2/Textures/T_sjfvcgnc_2K_B",
               MS + "Furniture_Fabric_sjfvcgnc/Medium/sjfvcgnc_tier_2/Textures/T_sjfvcgnc_2K_N"),
}
# scan-based instances: name -> (texture key, TileSize, Roughness, Tint)   (albedo cap 0.75, D-55)
SCAN_MIS = {
    "MI_RoomB_Plaster":     ("plaster_dmg", 320, 0.92, (0.74, 0.73, 0.70)),   # whitewashed panels; big tile hides the brick
    "MI_RoomB_WoodFloor":   ("walnut",  110, 0.50, (0.42, 0.30, 0.18)),   # 마루
    "MI_RoomB_WoodDark":    ("walnut",  140, 0.60, (0.18, 0.11, 0.06)),   # furniture, rafters, screen frame
    "MI_RoomB_ScreenField": ("plaster_dmg", 150, 0.80, (0.60, 0.48, 0.26)),   # gold-cream 병풍 field
    "MI_RoomB_Painting":    ("plaster_dmg",  80, 0.85, (0.74, 0.70, 0.60)),   # painting inset (placeholder)
    "MI_RoomB_MatRed":      ("fabric",   60, 0.95, (0.55, 0.10, 0.08)),
    "MI_RoomB_MatBlue":     ("fabric",   60, 0.95, (0.10, 0.14, 0.45)),
}
# procedural instances duplicated from existing ones: name -> (source, scalars, vectors)
DUP_MIS = {
    "MI_RoomB_Lacquer":  ("MI_Lobby_Brass", {"Roughness": 0.42, "Metallic": 0.0, "NoiseScale": 0.03},
                          {"ColorA": (0.42, 0.08, 0.05), "ColorB": (0.34, 0.06, 0.04)}),   # 주칠
    "MI_RoomB_Green":    ("MI_Lobby_Brass", {"Roughness": 0.55, "Metallic": 0.0, "NoiseScale": 0.03},
                          {"ColorA": (0.10, 0.30, 0.16), "ColorB": (0.08, 0.24, 0.13)}),   # 창살 green
    "MI_RoomB_Daylight": ("MI_RoomA_Cove", {"Intensity": 3.5}, {"EmissiveColor": (1.0, 0.96, 0.88)}),
    "MI_RoomB_Flame":    ("MI_RoomA_Cove", {"Intensity": 10.0}, {"EmissiveColor": (1.0, 0.62, 0.22)}),
    "MI_RoomB_Wax":      ("MI_Artifact_Clay", {"Roughness": 0.6}, {"Color": (0.85, 0.80, 0.68)}),
    "MI_Artifact_Stoneware": ("MI_Artifact_Clay", {"Roughness": 0.72}, {"Color": (0.30, 0.33, 0.36)}),
}
SLOTS = {
    "SM_RoomB_Floor":    {"Planks": "MI_RoomB_WoodFloor", "Border": "MI_RoomB_WoodDark"},
    "SM_RoomB_Walls":    {"Plaster": "MI_RoomB_Plaster", "Lacquer": "MI_RoomB_Lacquer",
                          "Green": "MI_RoomB_Green", "Daylight": "MI_RoomB_Daylight"},
    "SM_RoomB_Ceiling":  {"Plaster": "MI_RoomB_Plaster", "Lacquer": "MI_RoomB_Lacquer", "DarkWood": "MI_RoomB_WoodDark"},
    "SM_RoomB_Screen":   {"Frame": "MI_RoomB_WoodDark", "Field": "MI_RoomB_ScreenField", "Painting": "MI_RoomB_Painting"},
    "SM_RoomB_Platform": {"DarkWood": "MI_RoomB_WoodDark", "MatRed": "MI_RoomB_MatRed", "MatBlue": "MI_RoomB_MatBlue"},
    "SM_RoomB_Stand":    {"DarkWood": "MI_RoomB_WoodDark"},
    "SM_RoomB_Table":    {"DarkWood": "MI_RoomB_WoodDark"},
    "SM_RoomB_Candle":   {"Brass": "MI_Lobby_Brass", "Wax": "MI_RoomB_Wax", "Flame": "MI_RoomB_Flame"},
}
for m in MOCKS:
    SLOTS[m] = {"Stoneware": "MI_Artifact_Stoneware"}

# satellite vessels on the two tables (cm, local). Tables are centred at (60, +-190), top 80.
SAT = {"S1": (30, -190, 80), "S2": (110, -190, 80), "S3": (30, 190, 80), "S4": (110, 190, 80)}
SAT_MESH = {"S1": "SM_Mock_Silla_Gobae", "S2": "SM_Mock_Silla_Gidae",
            "S3": "SM_Mock_Silla_Dangyeongho", "S4": "SM_Mock_Silla_Gobae_Tall"}
SAT_LABEL = {"S1": "Mock_S1_Gobae", "S2": "Mock_S2_Gidae", "S3": "Mock_S3_Dangyeongho", "S4": "Mock_S4_GobaeTall"}
CANDLES = {"Candle_L": (-22, -190, 80), "Candle_R": (-22, 190, 80)}
HERO_Z = 12 + 102


def apath(folder, name):
    return "%s/%s.%s" % (folder, name, name)


def exists(pkg):
    return call(AST, "exists", {"path": pkg}) is True


# ---------------------------------------------------------------- phase: import
def phase_import():
    for name, (lm, hulls) in list(SHELL.items()) + [(m, (128, 6)) for m in MOCKS]:
        folder = MOCK_DIR if name.startswith("SM_Mock") else MESH_DIR
        pkg = "%s/%s" % (folder, name)
        if exists(pkg):
            call(AST, "delete", {"path": pkg})
        call(SMT, "import_file", {"folder_path": folder, "asset_name": name,
                                  "source_file": "%s/%s.fbx" % (FBX, name),
                                  "import_materials": False, "import_textures": False, "combine_meshes": True})
        ap = apath(folder, name)
        tris = call(SMT, "get_triangle_count", {"mesh": ref(ap), "lod_index": 0}, strict=False)
        slots = call(SMT, "get_material_slots", {"mesh": ref(ap)}, strict=False)
        set_props(ap, {"LightMapResolution": lm})
        lmc = get_props(ap, ["LightMapCoordinateIndex"])
        call(SMT, "generate_convex_collisions", {"mesh": ref(ap), "hull_count": hulls,
                                                 "max_hull_verts": 24, "hull_precision": 100000})
        print("  %-30s tris=%s slots=%s lmUV=%s" % (name, tris, slots, lmc.get("LightMapCoordinateIndex")))


# ---------------------------------------------------------------- phase: materials
def make_scan_mi(name, key, tile, rough, tint):
    ap = apath(MAT_DIR, name)
    if not exists(MAT_DIR + "/" + name):
        call(MIT, "create", {"folder_path": MAT_DIR, "asset_name": name, "parent": ref(apath(MAT_DIR, "M_Museum_Scan"))})
        print("  create", name)
    b, n = TEX[key]
    b, n = b + "." + b.split("/")[-1], n + "." + n.split("/")[-1]
    call(MIT, "set_texture_parameter", {"instance": ref(ap), "name": "BaseColorTex", "value": ref(b)})
    call(MIT, "set_texture_parameter", {"instance": ref(ap), "name": "NormalTex", "value": ref(n)})
    call(MIT, "set_scalar_parameter", {"instance": ref(ap), "name": "TileSize", "value": tile})
    call(MIT, "set_scalar_parameter", {"instance": ref(ap), "name": "Roughness", "value": rough})
    call(MIT, "set_vector_parameter", {"instance": ref(ap), "name": "Tint", "value": {"r": tint[0], "g": tint[1], "b": tint[2], "a": 1}})


def dup_mi(src, name, scalars=None, vectors=None):
    ap = apath(MAT_DIR, name)
    if not exists(MAT_DIR + "/" + name):
        call(AST, "duplicate", {"path": apath(MAT_DIR, src), "new_path": ap})
        print("  duplicate", src, "->", name)
    for k, v in (scalars or {}).items():
        call(MIT, "set_scalar_parameter", {"instance": ref(ap), "name": k, "value": v})
    for k, v in (vectors or {}).items():
        call(MIT, "set_vector_parameter", {"instance": ref(ap), "name": k, "value": {"r": v[0], "g": v[1], "b": v[2], "a": 1}})


def phase_materials():
    for name, (key, tile, rough, tint) in SCAN_MIS.items():
        make_scan_mi(name, key, tile, rough, tint)
    for name, (src, sc, vec) in DUP_MIS.items():
        dup_mi(src, name, sc, vec)
    for mesh, slots in SLOTS.items():
        folder = MOCK_DIR if mesh.startswith("SM_Mock") else MESH_DIR
        for slot, mi in slots.items():
            call(SMT, "set_material", {"mesh": ref(apath(folder, mesh)), "slot_name": slot, "material": ref(apath(MAT_DIR, mi))})
        print("  %-22s %s" % (mesh, [call(SMT, "get_material", {"mesh": ref(apath(folder, mesh)), "slot_name": s})["refPath"].split(".")[-1] for s in slots]))


# ---------------------------------------------------------------- phase: level
REMOVE = ["RoomB_Backdrop", "RoomB_Dais", "RoomB_Altar", "RoomB_Plinth", "RoomB_Bench",
          # leftovers from the template copy, in case a fresh L_Room_B is used
          "Wall_0", "Wall_1", "Wall_3", "Wall_4", "Wall_5", "Wall_6", "Floor",
          "Pedestal_02", "Backdrop_Main", "Pedestal_Plinth", "Ceiling_Gallery"]
PLACE = [  # (label, asset, loc, rot, folder)
    ("RoomB_Floor",    apath(MESH_DIR, "SM_RoomB_Floor"),    (0, 0, 0),    (0, 0, 0), "01_Structure"),
    ("RoomB_Walls",    apath(MESH_DIR, "SM_RoomB_Walls"),    (0, 0, 0),    (0, 0, 0), "01_Structure"),
    ("RoomB_Ceiling",  apath(MESH_DIR, "SM_RoomB_Ceiling"),  (0, 0, 0),    (0, 0, 0), "01_Structure"),
    ("RoomB_Screen",   apath(MESH_DIR, "SM_RoomB_Screen"),   (180, 0, 0),  (0, 0, 0), "04_Props"),
    ("RoomB_Platform", apath(MESH_DIR, "SM_RoomB_Platform"), (0, 0, 0),    (0, 0, 0), "03_Exhibit"),
    ("RoomB_Stand",    apath(MESH_DIR, "SM_RoomB_Stand"),    (0, 0, 12),   (0, 0, 0), "03_Exhibit"),
    ("RoomB_Table_L",  apath(MESH_DIR, "SM_RoomB_Table"),    (60, -190, 0), (0, 0, 0), "03_Exhibit"),
    ("RoomB_Table_R",  apath(MESH_DIR, "SM_RoomB_Table"),    (60, 190, 0),  (0, 0, 0), "03_Exhibit"),
]
for k in ("S1", "S2", "S3", "S4"):
    PLACE.append((SAT_LABEL[k], apath(MOCK_DIR, SAT_MESH[k]), SAT[k], (0, 0, 0), "03_Exhibit"))
for lbl, loc in CANDLES.items():
    PLACE.append((lbl, apath(MESH_DIR, "SM_RoomB_Candle"), loc, (0, 0, 0), "04_Props"))


def ensure_level():
    if call(SCN, "get_current_level", {}) != LEVEL:
        print("  load_level", LEVEL)
        call(SCN, "load_level", {"level_path": LEVEL})
    assert call(SCN, "get_current_level", {}) == LEVEL


def phase_level():
    ensure_level()
    labels = actors_by_label("L_Room_B")
    for lbl in REMOVE:
        if lbl in labels:
            call(SCN, "remove_from_scene", {"actor": labels[lbl]})
            print("  removed", lbl)
    labels = actors_by_label("L_Room_B")
    for lbl, asset, loc, rot, folder in PLACE:
        if lbl in labels:
            path = labels[lbl]
        else:
            r = call(SCN, "add_to_scene_from_asset", {"asset_path": asset, "name": lbl, "xform": xform(loc, rot)})
            path = r.get("refPath") if isinstance(r, dict) else r
            call(ACT, "set_label", {"actor": path, "label": lbl})
            print("  placed", lbl)
        call(ACT, "set_actor_transform", {"actor": path, "xform": xform(loc, rot), "worldspace": True})
        call(SCN, "set_actor_folder", {"actor": path, "folder_path": folder})
        set_props(path + ".StaticMeshComponent0", {"Mobility": "Static", "StaticMesh": ref(asset), "OverrideMaterials": []})
    # hero: keep the BP_Artifact actor (interaction + tag), mock mesh, on top of the stand
    labels = actors_by_label("L_Room_B")
    hero = labels.get("Artifact_BitsalPot") or labels.get("Artifact_Hero_Mock")
    if hero:
        set_props(hero + ".ArtifactMesh", {"StaticMesh": ref(apath(MOCK_DIR, "SM_Mock_Silla_Janggyeongho")),
                                           "RelativeScale3D": {"x": 1, "y": 1, "z": 1}})
        call(ACT, "set_actor_transform", {"actor": hero, "xform": xform((0, 0, HERO_Z)), "worldspace": True})
        call(ACT, "set_label", {"actor": hero, "label": "Artifact_Hero_Mock"})
        call(SCN, "set_actor_folder", {"actor": hero, "folder_path": "03_Exhibit"})
        print("  hero at z", HERO_Z)
    for lbl, loc in (("Refl_RoomB_Center", (0, 0, 180)), ("Refl_RoomB_Artifact", (-60, 0, 130))):
        if lbl not in labels:
            r = call(SCN, "add_to_scene_from_class", {"actor_type": ref("/Script/Engine.SphereReflectionCapture"), "name": lbl, "xform": xform(loc)})
            path = r.get("refPath") if isinstance(r, dict) else r
            call(ACT, "set_label", {"actor": path, "label": lbl})
            call(SCN, "set_actor_folder", {"actor": path, "folder_path": "05_Lighting"})


# ---------------------------------------------------------------- phase: lights
def aim(frm, to):
    dx, dy, dz = to[0]-frm[0], to[1]-frm[1], to[2]-frm[2]
    return (math.degrees(math.atan2(dz, math.hypot(dx, dy))), math.degrees(math.atan2(dy, dx)), 0)


SPOTS = []  # (label, from, to, intensity, kelvin, outer, inner, radius)
for k in ("S1", "S2", "S3", "S4"):
    x, y, z = SAT[k]
    side = -1 if y < 0 else 1
    SPOTS.append(("Spot_%s" % k, (x - 90, y - side*60, 340), (x, y, z + 12), 150, 4300, 22, 12, 520))
SPOTS.append(("Spot_Backdrop", (-60, 0, 350), (175, 0, 115), 160, 4500, 50, 35, 700))
# daylight through the four windows (outer bays of the side walls, window centre z 195)
for name, (x, y) in {"NW": (-333, -475), "NE": (333, -475), "SW": (-333, 475), "SE": (333, 475)}.items():
    SPOTS.append(("Spot_Window_%s" % name, (x, y, 195), (x * 0.5, 0, 40), 320, 5600, 66, 45, 950))
POINTS = {lbl: ((x, y, z + 58), 30, 1900, 220) for lbl, (x, y, z) in CANDLES.items()}   # candle flames

RETUNE = {
    "Fill_01": {"Intensity": 170, "Temperature": 5000}, "Fill_02": {"Intensity": 170, "Temperature": 5000},
    "Fill_03": {"Intensity": 170, "Temperature": 5000}, "Fill_04": {"Intensity": 170, "Temperature": 5000},
    "Spot_Exhibit_02": {"Intensity": 450, "Temperature": 4800, "OuterConeAngle": 18, "InnerConeAngle": 11},
    "Spot_Rim": {"Intensity": 200, "Temperature": 5600},
}


def phase_lights():
    ensure_level()
    labels = actors_by_label("L_Room_B")
    for lbl, vals in RETUNE.items():
        if lbl in labels:
            v = dict(vals); v["bUseTemperature"] = True
            set_props(labels[lbl] + ".LightComponent0", v)
    for lbl, frm, to, inten, kelvin, outer, inner, radius in SPOTS:
        if lbl in labels:
            path = labels[lbl]
        else:
            r = call(SCN, "add_to_scene_from_class", {"actor_type": ref("/Script/Engine.SpotLight"), "name": lbl, "xform": xform(frm, aim(frm, to))})
            path = r.get("refPath") if isinstance(r, dict) else r
            call(ACT, "set_label", {"actor": path, "label": lbl})
        call(ACT, "set_actor_transform", {"actor": path, "xform": xform(frm, aim(frm, to)), "worldspace": True})
        call(SCN, "set_actor_folder", {"actor": path, "folder_path": "05_Lighting"})
        set_props(path + ".LightComponent0", {"Mobility": "Static", "Intensity": inten, "bUseTemperature": True, "Temperature": kelvin,
                                              "OuterConeAngle": outer, "InnerConeAngle": inner, "AttenuationRadius": radius})
        print("  spot", lbl, frm, [round(a) for a in aim(frm, to)])
    for lbl, (loc, inten, kelvin, radius) in POINTS.items():
        plbl = "Light_" + lbl
        if plbl in labels:
            path = labels[plbl]
        else:
            r = call(SCN, "add_to_scene_from_class", {"actor_type": ref("/Script/Engine.PointLight"), "name": plbl, "xform": xform(loc)})
            path = r.get("refPath") if isinstance(r, dict) else r
            call(ACT, "set_label", {"actor": path, "label": plbl})
        call(ACT, "set_actor_transform", {"actor": path, "xform": xform(loc), "worldspace": True})
        call(SCN, "set_actor_folder", {"actor": path, "folder_path": "05_Lighting"})
        set_props(path + ".LightComponent0", {"Mobility": "Static", "Intensity": inten, "bUseTemperature": True, "Temperature": kelvin, "AttenuationRadius": radius})
        print("  point", plbl, loc)


# ---------------------------------------------------------------- phase: cleanup (old tomb-pass assets)
def phase_cleanup():
    for n in OLD_MESHES:
        p = "%s/%s" % (MESH_DIR, n)
        if exists(p):
            print("  delete", n, call(AST, "delete", {"path": p}, strict=False))
    for n in OLD_MIS:
        p = "%s/%s" % (MAT_DIR, n)
        if exists(p):
            refs = call(AST, "get_referencers", {"asset_path": p}, strict=False)
            print("  delete", n, "referencers:", str(refs)[:120], call(AST, "delete", {"path": p}, strict=False))


# ---------------------------------------------------------------- phase: save
def phase_save():
    assets = [LEVEL] + ["%s/%s" % (MESH_DIR, n) for n in SHELL] + ["%s/%s" % (MOCK_DIR, n) for n in MOCKS] \
        + ["%s/%s" % (MAT_DIR, n) for n in list(SCAN_MIS) + list(DUP_MIS)]
    for a in assets:
        r = call(AST, "save_assets", {"asset_paths": [a]}, strict=False)
        d = call(AST, "is_dirty", {"asset_path": a}, strict=False)
        if r is not True or d is not False:
            print("  PROBLEM", a, "save=", r, "dirty=", d)
    print("  saved", len(assets), "assets")


if __name__ == "__main__":
    for ph in sys.argv[1:]:
        print("=== phase", ph)
        globals()["phase_" + ph]()
    print("DONE")
