"""Room B folding screen into UE: textures, a UV-mapped master material, instances, mesh reimport, relink."""
import sys
from uemcp import *

FBX = "C:/Users/<USER>/AppData/Local/Temp/museum_fbx"
MESH_DIR, MAT_DIR, TEX_DIR = "/Game/Museum/Meshes", "/Game/Museum/Materials", "/Game/Museum/Textures"
LEVEL = "/Game/Museum/Levels/L_Room_B"
MTL = "editor_toolset.toolsets.material.MaterialTools"


def apath(folder, name):
    return "%s/%s.%s" % (folder, name, name)


def exists(pkg):
    return call(AST, "exists", {"path": pkg}) is True


def phase_textures():
    call(AST, "create_folder", {"path": TEX_DIR}, strict=False)
    for n in ("T_RoomB_ScreenPainting", "T_RoomB_Brocade"):
        if exists(TEX_DIR + "/" + n):
            call(AST, "delete", {"path": TEX_DIR + "/" + n})
        r = call("editor_toolset.toolsets.texture.TextureTools", "import_file",
                 {"folder_path": TEX_DIR, "asset_name": n, "source_file": "%s/%s.png" % (FBX, n)})
        print("  texture", n, call("editor_toolset.toolsets.texture.TextureTools", "get_size", {"texture": ref(apath(TEX_DIR, n))}, strict=False))
    # the painting must not tile and should keep its sharpness at 4K
    set_props(apath(TEX_DIR, "T_RoomB_ScreenPainting"), {"AddressX": "TA_Clamp", "AddressY": "TA_Clamp", "LODGroup": "TEXTUREGROUP_UI"})


def phase_master():
    """M_Museum_UVTex: BaseColorTex (UV0) -> BaseColor, Roughness scalar -> Roughness."""
    name = "M_Museum_UVTex"
    mp = apath(MAT_DIR, name)
    if exists(MAT_DIR + "/" + name):
        print("  master exists")
        return
    call(MTL, "create_material", {"folder_path": MAT_DIR, "asset_name": name})
    tex = call(MTL, "add_expression", {"material_or_function": ref(mp),
                                       "expression_class": ref("/Script/Engine.MaterialExpressionTextureSampleParameter2D"), "x": -400, "y": 0})
    texp = tex["refPath"] if isinstance(tex, dict) else tex
    set_props(texp, {"ParameterName": "BaseColorTex",
                     "Texture": ref(apath(TEX_DIR, "T_RoomB_ScreenPainting"))})
    r = call(MTL, "connect_to_output", {"expression": ref(texp), "output_name": "RGB", "material_property": "MP_BaseColor"})
    print("  basecolor wired", r)
    rough = call(MTL, "add_expression", {"material_or_function": ref(mp),
                                         "expression_class": ref("/Script/Engine.MaterialExpressionScalarParameter"), "x": -400, "y": 300})
    roughp = rough["refPath"] if isinstance(rough, dict) else rough
    set_props(roughp, {"ParameterName": "Roughness", "DefaultValue": 0.85})
    r = call(MTL, "connect_to_output", {"expression": ref(roughp), "output_name": "", "material_property": "MP_Roughness"})
    print("  roughness wired", r)
    r = call(MTL, "recompile", {"material_or_function": ref(mp)})
    print("  recompile", r)


def mi_scan(name, base, normal, tile, rough, tint):
    ap = apath(MAT_DIR, name)
    if not exists(MAT_DIR + "/" + name):
        call(MIT, "create", {"folder_path": MAT_DIR, "asset_name": name, "parent": ref(apath(MAT_DIR, "M_Museum_Scan"))})
    call(MIT, "set_texture_parameter", {"instance": ref(ap), "name": "BaseColorTex", "value": ref(base)})
    call(MIT, "set_texture_parameter", {"instance": ref(ap), "name": "NormalTex", "value": ref(normal)})
    call(MIT, "set_scalar_parameter", {"instance": ref(ap), "name": "TileSize", "value": tile})
    call(MIT, "set_scalar_parameter", {"instance": ref(ap), "name": "Roughness", "value": rough})
    call(MIT, "set_vector_parameter", {"instance": ref(ap), "name": "Tint", "value": {"r": tint[0], "g": tint[1], "b": tint[2], "a": 1}})


def phase_materials():
    # painting: re-parent the existing placeholder instance onto the UV master
    ap = apath(MAT_DIR, "MI_RoomB_Painting")
    call(MIT, "set_parent", {"instance": ref(ap), "parent": ref(apath(MAT_DIR, "M_Museum_UVTex"))})
    call(MIT, "clear_parameters", {"instance": ref(ap)}, strict=False)
    call(MIT, "set_texture_parameter", {"instance": ref(ap), "name": "BaseColorTex", "value": ref(apath(TEX_DIR, "T_RoomB_ScreenPainting"))})
    call(MIT, "set_scalar_parameter", {"instance": ref(ap), "name": "Roughness", "value": 0.85})
    print("  painting parent", get_props(ap, ["Parent"]))
    # brocade: world-aligned scan master with our tileable texture (15 cm repeat)
    plaster_n = "/Game/Fab/Megascans/Surfaces/Damaged_Wall_Plaster_uc2lddzcw/Medium/uc2lddzcw_tier_2/Textures/T_uc2lddzcw_2K_N.T_uc2lddzcw_2K_N"
    mi_scan("MI_RoomB_Brocade", apath(TEX_DIR, "T_RoomB_Brocade"), plaster_n, 15, 0.7, (0.72, 0.62, 0.40))
    # near-black lacquer for the frame rails
    fl = apath(MAT_DIR, "MI_RoomB_FrameLacquer")
    if not exists(MAT_DIR + "/MI_RoomB_FrameLacquer"):
        call(AST, "duplicate", {"path": apath(MAT_DIR, "MI_Lobby_Brass"), "new_path": fl})
    for k, v in {"Roughness": 0.30, "Metallic": 0.0, "NoiseScale": 0.05}.items():
        call(MIT, "set_scalar_parameter", {"instance": ref(fl), "name": k, "value": v})
    for k, v in {"ColorA": (0.06, 0.035, 0.025), "ColorB": (0.10, 0.06, 0.04)}.items():
        call(MIT, "set_vector_parameter", {"instance": ref(fl), "name": k, "value": {"r": v[0], "g": v[1], "b": v[2], "a": 1}})


SLOTS = {"Frame": "MI_RoomB_FrameLacquer", "Brocade": "MI_RoomB_Brocade", "Painting": "MI_RoomB_Painting",
         "Paper": "MI_RoomB_ScreenField", "Brass": "MI_Lobby_Brass"}


def phase_mesh():
    pkg = MESH_DIR + "/SM_RoomB_Screen"
    if exists(pkg):
        ok = call(AST, "delete", {"path": pkg})
        if ok is not True:
            raise SystemExit("delete failed (read-only? lock it): %r" % ok)
    call(SMT, "import_file", {"folder_path": MESH_DIR, "asset_name": "SM_RoomB_Screen",
                              "source_file": FBX + "/SM_RoomB_Screen.fbx", "import_materials": False,
                              "import_textures": False, "combine_meshes": True})
    ap = apath(MESH_DIR, "SM_RoomB_Screen")
    set_props(ap, {"LightMapResolution": 512})
    call(SMT, "generate_convex_collisions", {"mesh": ref(ap), "hull_count": 16, "max_hull_verts": 24, "hull_precision": 100000})
    for slot, mi in SLOTS.items():
        call(SMT, "set_material", {"mesh": ref(ap), "slot_name": slot, "material": ref(apath(MAT_DIR, mi))})
    print("  mesh", call(SMT, "get_triangle_count", {"mesh": ref(ap), "lod_index": 0}), "tris, slots",
          call(SMT, "get_material_slots", {"mesh": ref(ap)}), "lmUV", get_props(ap, ["LightMapCoordinateIndex"]))


def phase_relink():
    labels = actors_by_label("L_Room_B")
    p = labels["RoomB_Screen"]
    set_props(p + ".StaticMeshComponent0", {"StaticMesh": ref(apath(MESH_DIR, "SM_RoomB_Screen")), "OverrideMaterials": [], "Mobility": "Static"})
    print("  relinked", p.split(":PersistentLevel.")[-1], get_props(p + ".StaticMeshComponent0", ["StaticMesh"]))


def phase_save():
    for a in [LEVEL, MESH_DIR + "/SM_RoomB_Screen", TEX_DIR + "/T_RoomB_ScreenPainting", TEX_DIR + "/T_RoomB_Brocade",
              MAT_DIR + "/M_Museum_UVTex", MAT_DIR + "/MI_RoomB_Painting", MAT_DIR + "/MI_RoomB_Brocade", MAT_DIR + "/MI_RoomB_FrameLacquer"]:
        r = call(AST, "save_assets", {"asset_paths": [a]}, strict=False)
        d = call(AST, "is_dirty", {"asset_path": a}, strict=False)
        if d is not False:
            print("  PROBLEM", a, "save=", r, "dirty=", d)
    print("  saved")


if __name__ == "__main__":
    for ph in sys.argv[1:]:
        print("=== phase", ph)
        globals()["phase_" + ph]()
    print("DONE")
