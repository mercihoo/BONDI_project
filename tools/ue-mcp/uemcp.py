"""Tiny client for the UE 5.8 editor MCP server (streamable HTTP on 127.0.0.1:8000)."""
import json, os, urllib.request

URL = "http://127.0.0.1:8000/mcp"
SID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ue_sid")
HDR = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


def _rpc(method, params, sid=None, rid=1):
    h = dict(HDR)
    if sid:
        h["Mcp-Session-Id"] = sid
    body = json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}).encode()
    req = urllib.request.Request(URL, data=body, headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        raw = r.read().decode("utf-8", "replace")
        new_sid = r.headers.get("Mcp-Session-Id")
    if raw.startswith("event:") or raw.startswith("data: "):
        raw = "".join(l[6:] for l in raw.splitlines() if l.startswith("data: "))
    return (json.loads(raw) if raw.strip() else {}), new_sid


def _session():
    if os.path.exists(SID_FILE) and os.path.getsize(SID_FILE) > 0:
        return open(SID_FILE).read().strip()
    _, sid = _rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                 "clientInfo": {"name": "cc", "version": "1"}}, rid=0)
    open(SID_FILE, "w").write(sid)
    try:
        _rpc("notifications/initialized", {}, sid)
    except Exception:
        pass
    return sid


SID = _session()


class UEError(RuntimeError):
    pass


def call(toolset, tool, args, strict=True):
    d, _ = _rpc("tools/call", {"name": "call_tool", "arguments": {
        "toolset_name": toolset, "tool_name": tool, "arguments": args}}, SID)
    if "error" in d:
        if strict:
            raise UEError("%s.%s: %s" % (toolset.split(".")[-1], tool, d["error"]))
        return {"_err": d["error"]}
    r = d.get("result", d)
    txt = "".join(c.get("text", "") for c in r.get("content", []))
    if r.get("isError"):
        if strict:
            raise UEError("%s.%s%s: %s" % (toolset.split(".")[-1], tool, json.dumps(args)[:200], txt[:600]))
        return {"_err": txt}
    try:
        j = json.loads(txt)
    except Exception:
        return txt
    rv = j.get("returnValue", j) if isinstance(j, dict) else j
    if isinstance(rv, str):
        try:
            rv = json.loads(rv)
        except Exception:
            pass
    return rv


ACT = "editor_toolset.toolsets.actor.ActorTools"
OBJ = "editor_toolset.toolsets.object.ObjectTools"
SCN = "editor_toolset.toolsets.scene.SceneTools"
SMT = "editor_toolset.toolsets.static_mesh.StaticMeshTools"
AST = "editor_toolset.toolsets.asset.AssetTools"
MIT = "editor_toolset.toolsets.material_instance.MaterialInstanceTools"
PRM = "editor_toolset.toolsets.primitive.PrimitiveTools"
UMG = "UMGToolSet.UMGToolSet"
APP = "EditorToolset.EditorAppToolset"


def ref(path):
    return {"refPath": path}


def xform(loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
    return {"location": {"x": loc[0], "y": loc[1], "z": loc[2]},
            "rotation": {"pitch": rot[0], "yaw": rot[1], "roll": rot[2]},
            "scale": {"x": scale[0], "y": scale[1], "z": scale[2]}}


def get_props(path, names):
    return call(OBJ, "get_properties", {"instance": ref(path), "properties": names})


def set_props(path, values):
    """values: dict -> sent as a JSON *string* (an object is silently ignored)."""
    r = call(OBJ, "set_properties", {"instance": ref(path), "values": json.dumps(values)})
    if r is not True and r != "true":
        raise UEError("set_properties on %s returned %r for %s" % (path, r, json.dumps(values)[:200]))
    return r


def find_actors():
    a = call(SCN, "find_actors", {"name": "", "tag": "", "collision_channels": []})
    if isinstance(a, dict):
        a = a.get("actors", [])
    return [(x.get("path") or x.get("refPath")) if isinstance(x, dict) else str(x) for x in a]


def label_of(path):
    return call(ACT, "get_label", {"actor": path})


def actors_by_label(prefix=""):
    out = {}
    for p in find_actors():
        if prefix and prefix not in p:
            continue
        out[label_of(p)] = p
    return out
