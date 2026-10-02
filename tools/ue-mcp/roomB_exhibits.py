"""Give every Room B vessel its own red press button and its own info panel.

Run with L_Room_B open standalone (local coordinates).

  buttons  a finger-sized red disc lying face-up on the table in front of each vessel, pressed
           downward (PressAxis 0,0,-1) — the old vertical disc was buried inside the plinth
  panels   one WBP per vessel, shown through two actors (Stage_Damaged / Stage_Restored) so the
           panel stays up in both button states; the hero keeps its existing pair off to the side
"""
import sys
from uemcp import *

UI = "/Game/Museum/UI"
MAT = "/Game/Museum/Materials"
TOGGLE = "/Game/Museum/Blueprints/BP_StageToggle.BP_StageToggle_C"

# label -> (vessel xy, mesh height cm, widget asset, texts)
VESSELS = {
    "Mock_S1_Gobae": ((30, -190), 26.8, "WBP_RoomB_S1_Gobae", {
        "Text_Title": "유개고배 — 자리표시",
        "Text_Stage": "근거 : 실제 유물 에셋 대기",
        "Text_Description": "뚜껑을 갖춘 굽다리접시입니다.\n제사와 부장에 함께 쓰였습니다.\n\n지금 놓인 것은 회전 프로파일로\n만든 자리표시용 모형입니다.",
        "Text_Disclaimer": "※ 배치 확인용 목업입니다. 실제 유물 에셋으로 교체됩니다.",
        "Text_Hint": "▶ 집어서 가까이 살펴볼 수 있습니다"}),
    "Mock_S2_Gidae": ((110, -190), 49.1, "WBP_RoomB_S2_Gidae", {
        "Text_Title": "기대(그릇받침) — 자리표시",
        "Text_Stage": "근거 : 실제 유물 에셋 대기",
        "Text_Description": "바닥이 둥근 항아리를 받치는\n받침입니다. 허리가 좁고 위아래가\n나팔처럼 벌어집니다.\n\n지금 놓인 것은 자리표시용입니다.",
        "Text_Disclaimer": "※ 배치 확인용 목업입니다. 실제 유물 에셋으로 교체됩니다.",
        "Text_Hint": "▶ 집어서 가까이 살펴볼 수 있습니다"}),
    "Mock_S3_Dangyeongho": ((30, 190), 27.3, "WBP_RoomB_S3_Dangyeongho", {
        "Text_Title": "단경호 — 자리표시",
        "Text_Stage": "근거 : 실제 유물 에셋 대기",
        "Text_Description": "목이 짧고 몸통이 부른 항아리입니다.\n곡식과 액체를 담아 무덤에 넣었습니다.\n\n지금 놓인 것은 자리표시용입니다.",
        "Text_Disclaimer": "※ 배치 확인용 목업입니다. 실제 유물 에셋으로 교체됩니다.",
        "Text_Hint": "▶ 집어서 가까이 살펴볼 수 있습니다"}),
    "Mock_S4_GobaeTall": ((110, 190), 28.4, "WBP_RoomB_S4_GobaeTall", {
        "Text_Title": "대각고배 — 자리표시",
        "Text_Stage": "근거 : 실제 유물 에셋 대기",
        "Text_Description": "굽다리가 높은 접시입니다.\n굽에 뚫은 구멍(투창)이 특징입니다.\n\n지금 놓인 것은 자리표시용입니다.",
        "Text_Disclaimer": "※ 배치 확인용 목업입니다. 실제 유물 에셋으로 교체됩니다.",
        "Text_Hint": "▶ 집어서 가까이 살펴볼 수 있습니다"}),
}

TABLE_TOP = 80.0          # long tables
STAND_TOP = 114.0         # 사방탁자 under the hero
BTN_SCALE = (0.045, 0.045, 0.018)   # 4.5 cm disc, 1.8 cm tall — a fingertip target
BTN_LIFT = 0.9            # half the disc height, so it rests on the surface
AISLE_INSET = 24.0        # button sits this far from the vessel, toward the aisle
PANEL_BEHIND = 32.0       # panel sits this far behind the vessel
PANEL_Z = 112.0
PANEL_SCALE = 0.07


def exists(pkg):
    return call(AST, "exists", {"path": pkg}) is True


def make_widget(name, texts):
    dst = "%s/%s.%s" % (UI, name, name)
    if not exists("%s/%s" % (UI, name)):
        call(AST, "duplicate", {"path": "%s/WBP_ArtifactInfo.WBP_ArtifactInfo" % UI, "new_path": dst})
        print("  widget", name, "duplicated")
    wt = dst + ":WidgetTree."
    for k, v in texts.items():
        set_props(wt + k, {"Text": v})
    call(UMG, "CompileWidgetBlueprint", {"widgetBlueprint": ref(dst)})
    return dst + "_C"


def panel_actor(label, widget_class, loc, yaw, tag):
    L = actors_by_label("L_Room_B")
    if label in L:
        p = L[label]
    else:
        r = call(SCN, "add_to_scene_from_class", {"actor_type": ref("/Script/Engine.Actor"),
                                                  "name": label, "xform": xform(loc, (0, yaw, 0))})
        p = r["refPath"] if isinstance(r, dict) else r
        call(ACT, "set_label", {"actor": p, "label": label})
        call(ACT, "add_component", {"owner": p, "component_type": ref("/Script/UMG.WidgetComponent"), "name": "Panel"})
        print("  panel", label, "created")
    call(ACT, "set_actor_transform", {"actor": p, "xform": xform(loc, (0, yaw, 0)), "worldspace": True})
    call(SCN, "set_actor_folder", {"actor": p, "folder_path": "03_Exhibit"})
    if tag not in (call(ACT, "get_tags", {"actor": p}, strict=False) or []):
        call(ACT, "add_tag", {"actor": p, "tag": tag})
    set_props(p + ".Panel", {"WidgetClass": ref(widget_class), "DrawSize": {"x": 700, "y": 560},
                             "Space": "World",
                             "RelativeScale3D": {"x": PANEL_SCALE, "y": PANEL_SCALE, "z": PANEL_SCALE},
                             "RelativeLocation": {"x": 0, "y": 0, "z": 0}})
    # the restored twin starts hidden in game, exactly like Room A's pair
    set_props(p, {"bHidden": tag == "Stage_Restored"})
    return p


def button_actor(label, loc):
    L = actors_by_label("L_Room_B")
    if label in L:
        p = L[label]
    else:
        r = call(SCN, "add_to_scene_from_class", {"actor_type": ref(TOGGLE), "name": label,
                                                  "xform": xform(loc, (0, 0, 0), BTN_SCALE)})
        p = r["refPath"] if isinstance(r, dict) else r
        call(ACT, "set_label", {"actor": p, "label": label})
        print("  button", label, "created")
    call(ACT, "set_actor_transform", {"actor": p, "xform": xform(loc, (0, 0, 0), BTN_SCALE), "worldspace": True})
    call(SCN, "set_actor_folder", {"actor": p, "folder_path": "03_Exhibit"})
    set_props(p, {"PressAxis": {"x": 0, "y": 0, "z": -1}, "LateralLimit": 3.0,
                  "MaxTravel": 1.2, "TriggerDepth": 0.9, "ShowRadius": 180, "HideRadius": 260})
    return p


def main():
    L = actors_by_label("L_Room_B")
    if "TMP_Panel" in L:
        call(SCN, "remove_from_scene", {"actor": L["TMP_Panel"]})
        print("  removed probe actor")

    # red, finger-pressable
    call(MIT, "set_vector_parameter", {"instance": ref("%s/MI_Button.MI_Button" % MAT), "name": "Color",
                                       "value": {"r": 0.72, "g": 0.06, "b": 0.05, "a": 1}})
    call(MIT, "set_scalar_parameter", {"instance": ref("%s/MI_Button.MI_Button" % MAT), "name": "Roughness", "value": 0.25})
    print("  MI_Button ->", call(MIT, "get_vector_parameter", {"instance": ref("%s/MI_Button.MI_Button" % MAT), "name": "Color"}))

    # hero: keep its side panels, just move its button onto the stand top
    hero_btn = actors_by_label("L_Room_B")["StageToggleButton"]
    call(ACT, "set_actor_transform", {"actor": hero_btn,
                                      "xform": xform((-25, 0, STAND_TOP + BTN_LIFT), (0, 0, 0), BTN_SCALE),
                                      "worldspace": True})
    set_props(hero_btn, {"PressAxis": {"x": 0, "y": 0, "z": -1}, "LateralLimit": 3.0})
    print("  hero button", call(ACT, "get_actor_transform", {"actor": hero_btn})["location"])

    for label, ((vx, vy), h, wname, texts) in VESSELS.items():
        key = label.split("_")[1]
        wclass = make_widget(wname, texts)
        side = -1 if vy < 0 else 1          # left table faces +Y, right table faces -Y
        # button: on the table, between the vessel and the aisle
        button_actor("Btn_%s" % key, (vx, vy - side * AISLE_INSET,
                                      TABLE_TOP + BTN_LIFT))
        # panel: behind the vessel, facing the aisle
        yaw = 90 if side < 0 else -90
        for tag, suffix in (("Stage_Damaged", "A"), ("Stage_Restored", "B")):
            panel_actor("Panel_%s_%s" % (key, suffix), wclass, (vx, vy + side * PANEL_BEHIND, PANEL_Z), yaw, tag)
        # this vessel has no damaged/restored pair yet, so its toggle must not swap the mesh
        art = actors_by_label("L_Room_B")[label]
        set_props(art + ".StaticMeshComponent0", {"ComponentTags": []})
        print("  %-20s button+2 panels, mesh tag cleared" % label)


if __name__ == "__main__":
    main()
    print("DONE")
