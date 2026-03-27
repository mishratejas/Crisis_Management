"""
streamlit_app.py  —  AEGIS Crisis Management AI  (LangGraph Edition)
======================================================================
Run:  streamlit run streamlit_app.py

Stage 1 now runs a real folder scan instead of single-image upload.
  - User picks a folder path + enters geo metadata once
  - UNet runs on each image in order, results shown live
  - First flood image auto-fires the LangGraph pipeline
  - Remaining images are skipped

Geo metadata (lat/lon/coverage) is still entered manually because
plain JPG/PNG images carry no spatial information. This is the same
data you were entering before — just explained clearly now.
"""

import streamlit as st

st.set_page_config(
    page_title="AEGIS — Crisis Management AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import io, os, sys, json, sqlite3, contextlib, tempfile, traceback, time
import numpy as np
from datetime import datetime
from pathlib import Path

import pandas as pd
import folium
from PIL import Image
from streamlit_folium import st_folium

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, str(_ROOT))

DB_PATH = os.path.join(_ROOT, "crisis.db")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
PIXEL_THRESHOLD = 0.45   # per-pixel flood probability (matches vision_agent / route_agent)
FLOOD_FRACTION  = 0.10   # >= 10% of pixels flooded → image is flood-positive

# ============================================================================
#  VISION MODEL — loaded once per server process
# ============================================================================

@st.cache_resource
def _load_vision():
    """Load UNet + preprocess — cached so the model loads only once."""
    from agents.vision_agent.preprocess         import load_image
    from agents.vision_agent.flood_segmentation import detect_flood
    return load_image, detect_flood


def _analyse_image(img_path: Path) -> dict:
    """
    Run the real UNet on one image.
    Returns flooded_fraction, max_prob, mean_prob, is_flood.
    Matches the pixel-threshold logic used in vision_agent and route_agent.
    """
    load_image, detect_flood = _load_vision()
    image    = load_image(str(img_path))
    prob_map = detect_flood(image)          # H×W float32

    total   = prob_map.size
    flooded = int((prob_map >= PIXEL_THRESHOLD).sum())
    frac    = flooded / total

    return {
        "name":             img_path.name,
        "path":             str(img_path),
        "mean_prob":        float(prob_map.mean()),
        "max_prob":         float(prob_map.max()),
        "flooded_pixels":   flooded,
        "total_pixels":     total,
        "flooded_fraction": frac,
        "is_flood":         frac >= FLOOD_FRACTION,
    }


def _collect_images(folder: Path) -> list:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


# ============================================================================
#  LANGGRAPH — build + compile ONCE per server process
# ============================================================================

@st.cache_resource
def _get_graph():
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import StateGraph, END
    from master_agent.master_state import MasterState
    from master_agent.master_nodes import (
        vision_node, store_zone_node, drone_analysis_node,
        drone_decision_node, drone_dispatch_node, drone_vision_node,
        update_people_node, rescue_decision_node,
        admin_resource_node, resource_approval_router,
        route_planner_node, admin_route_node, route_approval_router,
        communication_node,
    )

    b = StateGraph(MasterState)
    for name, fn in [
        ("vision",          vision_node),
        ("store_zone",      store_zone_node),
        ("drone_analysis",  drone_analysis_node),
        ("drone_decision",  drone_decision_node),
        ("drone_dispatch",  drone_dispatch_node),
        ("drone_vision",    drone_vision_node),
        ("update_people",   update_people_node),
        ("rescue_decision", rescue_decision_node),
        ("admin_resource",  admin_resource_node),
        ("route_planner",   route_planner_node),
        ("admin_route",     admin_route_node),
        ("communication",   communication_node),
    ]:
        b.add_node(name, fn)

    b.set_entry_point("vision")
    for src, dst in [
        ("vision",          "store_zone"),
        ("store_zone",      "drone_analysis"),
        ("drone_analysis",  "drone_decision"),
        ("drone_decision",  "drone_dispatch"),
        ("drone_dispatch",  "drone_vision"),
        ("drone_vision",    "update_people"),
        ("update_people",   "rescue_decision"),
        ("rescue_decision", "admin_resource"),
        ("route_planner",   "admin_route"),
        ("communication",   END),
    ]:
        b.add_edge(src, dst)

    b.add_conditional_edges("admin_resource", resource_approval_router,
                            {"approved": "route_planner", "rejected": "rescue_decision"})
    b.add_conditional_edges("admin_route", route_approval_router,
                            {"approved": "communication", "rejected": "route_planner"})

    return b.compile(
        checkpointer     = MemorySaver(),
        interrupt_before = ["admin_resource", "admin_route"],
    )


def _cfg():
    return {"configurable": {"thread_id": st.session_state.get("thread_id", "aegis_main")}}


def _graph_state() -> dict:
    try:
        snap = _get_graph().get_state(_cfg())
        if snap and snap.values:
            return dict(snap.values)
    except Exception:
        pass
    return {}


def _next_nodes() -> list:
    try:
        snap = _get_graph().get_state(_cfg())
        return list(snap.next) if snap and snap.next else []
    except Exception:
        return []


# ── Phase runners ──────────────────────────────────────────────────────────────

def _invoke(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    if buf.getvalue().strip():
        _log(buf.getvalue().strip())
    return result


def _run_phase1(img_path: str, image_meta: dict):
    _log("LangGraph Phase 1 starting …")
    from dotenv import load_dotenv; load_dotenv()
    _invoke(
        _get_graph().invoke,
        {
            "satellite_image": img_path,
            "image_meta":      image_meta,
            "base_locations": {
                "ambulance":   {"name": "Hospital",      "lat": 19.06546856543151,  "lon": 72.86100899070198},
                "rescue_team": {"name": "Rescue Center", "lat": 19.06847079812735,  "lon": 72.85793995490616},
                "boat":        {"name": "Boat Depot",    "lat": 19.063380373548366, "lon": 72.85538649195271},
            },
            "field_reports":   [],
            "dispatch_config": {
                "send_sms":       bool(os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("YOUR_PHONE_NUMBER")),
                "generate_audio": True,
                "language":       st.session_state.get("comm_language", "English"),
                "to_number":      os.getenv("YOUR_PHONE_NUMBER"),
            },
        },
        config=_cfg(),
    )
    st.session_state["pipeline_phase"] = "awaiting_resource"
    _log("Phase 1 complete — interrupted before admin_resource")


def _run_phase2(approved: bool):
    graph = _get_graph(); config = _cfg()
    graph.update_state(config, {"resource_approved": approved}, as_node="admin_resource")
    _invoke(graph.invoke, None, config=config)
    st.session_state["pipeline_phase"] = "awaiting_route" if approved else "awaiting_resource"
    if approved:
        _log("Phase 2 complete — interrupted before admin_route")


def _run_phase3(approved: bool):
    graph = _get_graph(); config = _cfg()
    graph.update_state(config, {"route_approved": approved}, as_node="admin_route")
    _invoke(graph.invoke, None, config=config)
    st.session_state["pipeline_phase"] = "complete" if approved else "awaiting_route"
    if approved:
        _log("Phase 3 complete — pipeline DONE")


# ============================================================================
#  THEME
# ============================================================================

THEME = {
    "bg":     "#080d14", "bg2":    "#0d1520",
    "cyan":   "#00d4ff", "red":    "#ff2d55",
    "orange": "#ff9500", "green":  "#30d158",
    "yellow": "#ffd60a", "text":   "#e5e5e7",
    "mono":   "#00ff88",
}

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Share+Tech+Mono&family=Exo+2:wght@400;600&display=swap');
  body,.stApp{{background:{THEME['bg']};color:{THEME['text']};}}
  h1,h2,h3{{font-family:'Rajdhani',sans-serif;}}
  .stButton>button{{background:{THEME['bg2']};color:{THEME['cyan']};border:2px solid {THEME['cyan']};
    border-radius:6px;padding:8px 18px;font-family:'Share Tech Mono',monospace;transition:.25s;}}
  .stButton>button:hover{{background:{THEME['cyan']};color:{THEME['bg']};}}
  .stMetric{{background:{THEME['bg2']};padding:16px;border-radius:8px;border-left:4px solid {THEME['cyan']};}}
  .terminal-log{{background:#000;color:{THEME['mono']};font-family:'Share Tech Mono',monospace;
    font-size:11px;line-height:1.6;padding:12px;border-radius:4px;border:1px solid {THEME['green']};
    max-height:280px;overflow-y:auto;white-space:pre-wrap;word-break:break-word;}}
  .card{{background:{THEME['bg2']};border-left:4px solid {THEME['cyan']};border-radius:6px;
    padding:12px;margin:6px 0;}}
  .card-warn{{background:#1a0f00;border-left:4px solid {THEME['orange']};border-radius:6px;
    padding:12px;margin:6px 0;}}
  .card-flood{{background:#1a0005;border-left:4px solid {THEME['red']};border-radius:6px;
    padding:12px;margin:6px 0;}}
  .card-ok{{background:#001a08;border-left:4px solid {THEME['green']};border-radius:6px;
    padding:12px;margin:6px 0;}}
  .scan-row{{font-family:'Share Tech Mono',monospace;font-size:12px;padding:4px 0;}}
  div[data-testid="stDataFrame"]{{background:{THEME['bg2']};}}
</style>
""", unsafe_allow_html=True)

# ============================================================================
#  CONSTANTS
# ============================================================================

_DEFAULT_META = {
    "center_lat": 19.062061, "center_lon": 72.863542,
    "coverage_km": 1.6, "width_px": 1024, "height_px": 522,
}
ROUTE_COLORS   = {"ambulance":"#e74c3c","rescue_team":"#2980b9","boat":"#16a085",
                  "helicopter":"#8e44ad","fire_truck":"#e67e22","truck":"#7f8c8d"}
RESOURCE_EMOJI = {"ambulance":"🚑","rescue_team":"🚒","boat":"🚤",
                  "helicopter":"🚁","fire_truck":"🚒","truck":"🚛"}
BASE_ICON      = {"ambulance":("red","plus-sign"),"rescue_team":("blue","home"),
                  "boat":("darkblue","tint"),"helicopter":("purple","plane")}
_DEFAULT_BASES = {
    "ambulance":   {"name":"Hospital",      "lat":19.06546856543151,  "lon":72.86100899070198},
    "rescue_team": {"name":"Rescue Center", "lat":19.06847079812735,  "lon":72.85793995490616},
    "boat":        {"name":"Boat Depot",    "lat":19.063380373548366, "lon":72.85538649195271},
}

STAGES = [
    "1️⃣ Folder Scan","2️⃣ Zone Map","3️⃣ Drones","4️⃣ Gallery","5️⃣ Analysis",
    "6️⃣ Resources","7️⃣ Approve I","8️⃣ Routes","9️⃣ Approve II","🔟 Comms",
]

_PHASE_INFO = {
    "idle":              ("#555",          "⚫ Idle"),
    "scanning":          (THEME["cyan"],   "🔵 Scanning folder…"),
    "running_phase1":    (THEME["yellow"], "🟡 Phase 1 — Running Agents"),
    "awaiting_resource": (THEME["orange"], "🟠 Awaiting Resource Approval"),
    "running_phase2":    (THEME["yellow"], "🟡 Phase 2 — Planning Routes"),
    "awaiting_route":    (THEME["orange"], "🟠 Awaiting Route Approval"),
    "running_phase3":    (THEME["yellow"], "🟡 Phase 3 — Dispatching"),
    "complete":          (THEME["green"],  "🟢 Pipeline Complete"),
}

# ============================================================================
#  HELPERS
# ============================================================================

def _ts(): return datetime.now().strftime("%H:%M:%S")

def _log(text):
    st.session_state.setdefault("log", "")
    for line in (text or "").strip().splitlines():
        if line.strip():
            st.session_state["log"] += f"[{_ts()}] {line}\n"

def _terminal():
    log = st.session_state.get("log", "(no output yet)")
    st.markdown(
        f'<div class="terminal-log" id="tlog">{log}</div>'
        '<script>var t=document.getElementById("tlog");if(t)t.scrollTop=t.scrollHeight;</script>',
        unsafe_allow_html=True)

def _sev_label(s):
    if s >= 0.8: return "🔴 CRITICAL"
    if s >= 0.6: return "🟠 HIGH"
    if s >= 0.4: return "🟡 MODERATE"
    return "🟢 LOW"

def _rcolor(rt):
    k = rt.lower().rstrip("s")
    return ROUTE_COLORS.get(k, ROUTE_COLORS.get(rt.lower(), "#888"))

def _remoji(rt):
    k = rt.lower().rstrip("s")
    return RESOURCE_EMOJI.get(k, RESOURCE_EMOJI.get(rt.lower(), "🚗"))

_nav_calls = {}

def _reset_nav_counter():
    _nav_calls.clear()

def _nav(back=None, fwd=None, fwd_label="▶ PROCEED"):
    stage = st.session_state.get("stage", 0)
    _nav_calls[stage] = _nav_calls.get(stage, 0) + 1
    uid = f"{stage}_{_nav_calls[stage]}"
    c1, c2 = st.columns(2)
    with c1:
        if back is not None and st.button("◀ BACK", key=f"nb_{uid}"):
            st.session_state["stage"] = back; st.rerun()
    with c2:
        if fwd is not None and st.button(fwd_label, key=f"nf_{uid}"):
            st.session_state["stage"] = fwd; st.rerun()

def _phase_badge():
    phase = st.session_state.get("pipeline_phase", "idle")
    color, label = _PHASE_INFO.get(phase, ("#555", phase))
    st.markdown(
        f'<span style="background:{color};color:#000;padding:4px 14px;border-radius:12px;'
        f'font-family:\'Share Tech Mono\',monospace;font-size:11px;font-weight:bold;">'
        f'{label}</span><br><br>', unsafe_allow_html=True)

def _prob_bar(frac: float, width: int = 16) -> str:
    filled   = int(frac * width)
    thresh_i = int(FLOOD_FRACTION * width)
    bar = ""
    for i in range(width):
        if i < filled:
            bar += "█" if i < thresh_i else "▓"
        else:
            bar += "░"
    return bar

# ============================================================================
#  SIDEBAR
# ============================================================================

def _sidebar():
    with st.sidebar:
        st.markdown("### 🛰️ AEGIS · LangGraph Pipeline")
        st.divider()

        phase = st.session_state.get("pipeline_phase", "idle")
        color, label = _PHASE_INFO.get(phase, ("#555", phase))
        st.markdown(f'<div class="card">🔗 <b>Master Graph</b><br>'
                    f'<span style="color:{color};font-size:12px;">{label}</span></div>',
                    unsafe_allow_html=True)

        nxt = _next_nodes()
        if nxt:
            st.markdown(f'<div class="card">⏸️ <b>Interrupted Before</b><br>'
                        f'<span style="color:{THEME["cyan"]};font-size:12px;">'
                        f'{", ".join(nxt)}</span></div>', unsafe_allow_html=True)

        # Folder scan status
        scan_results = st.session_state.get("scan_results", [])
        if scan_results:
            st.divider()
            st.markdown("**📂 Folder Scan Results**")
            for r in scan_results:
                icon  = "🚨" if r["is_flood"] else ("⏩" if r.get("skipped") else "✅")
                color = THEME["red"] if r["is_flood"] else ("#888" if r.get("skipped") else THEME["green"])
                pct   = f'{r["flooded_fraction"]*100:.1f}%' if not r.get("skipped") else "—"
                st.markdown(
                    f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:10px;'
                    f'color:{color};padding:2px 0;">{icon} {r["name"][:22]} · {pct}</div>',
                    unsafe_allow_html=True)

        st.divider()
        st.markdown(
            f'<div class="card" style="font-family:\'Share Tech Mono\',monospace;font-size:10px;">'
            f'<b style="color:{THEME["cyan"]};">LangGraph Node Order</b><br><br>'
            f'vision<br>&nbsp;└─ store_zone<br>&nbsp;&nbsp;&nbsp;&nbsp;└─ drone_analysis<br>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ drone_decision<br>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ drone_dispatch<br>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ drone_vision<br>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ update_people<br>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ rescue_decision<br>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ <span style="color:{THEME["orange"]};">[⏸ admin_resource]</span><br>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ route_planner<br>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ <span style="color:{THEME["orange"]};">[⏸ admin_route]</span><br>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ communication<br>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ END'
            f'</div>', unsafe_allow_html=True)

        st.divider()
        gs = _graph_state()
        for name, icon, key in [
            ("Vision Agent",   "👁️",  "zone_map"),
            ("Drone Agent",    "🚁",  "drone_allocation"),
            ("Resource Agent", "📦",  "rescue_plan"),
            ("Route Agent",    "🗺️", "route_plan"),
            ("Comm Agent",     "📡",  "dispatch_result"),
        ]:
            val = gs.get(key)
            if val:
                s = f'<span style="color:{THEME["green"]};">🟢 Done</span>'
            else:
                s = f'<span style="color:#888;">⚪ Idle</span>'
            st.markdown(f'<div class="card">{icon} <b>{name}</b><br>{s}</div>',
                        unsafe_allow_html=True)

        st.divider()
        st.metric("Stage", f"{st.session_state.get('stage',0)+1} / {len(STAGES)}")
        st.divider()
        if st.button("🔄 Full Reset", use_container_width=True):
            import uuid
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.session_state["thread_id"] = f"aegis_{uuid.uuid4().hex[:8]}"
            st.rerun()

# ============================================================================
#  STEPPER
# ============================================================================

def _stepper():
    s = st.session_state.get("stage", 0)
    cols = st.columns(len(STAGES))
    for i, label in enumerate(STAGES):
        with cols[i]:
            if i < s:    bg, fg = THEME["green"], "black"
            elif i == s: bg, fg = THEME["cyan"],  THEME["bg"]
            else:         bg, fg = THEME["bg2"],   THEME["text"]
            st.markdown(
                f'<div style="background:{bg};color:{fg};padding:6px 2px;text-align:center;'
                f'border-radius:4px;font-size:9px;font-weight:bold;'
                f'border:1px solid {THEME["cyan"]};{"" }>{"✅ " if i<s else ""}{label}</div>',
                unsafe_allow_html=True)

# ============================================================================
#  FOLIUM MAP
# ============================================================================

def _folium_map():
    gs     = _graph_state()
    routes = gs.get("route_plan", [])
    meta   = gs.get("image_meta") or _DEFAULT_META
    fmap   = folium.Map(location=[meta.get("center_lat",19.06), meta.get("center_lon",72.86)],
                        zoom_start=15, tiles="CartoDB positron")

    seen_b, seen_z = set(), set()
    for r in routes:
        rk   = r.get("resource_type","").lower().rstrip("s")
        base = _DEFAULT_BASES.get(rk)
        if base and base["name"] not in seen_b:
            ic_c, ic_i = BASE_ICON.get(rk, ("gray","info-sign"))
            folium.Marker([base["lat"],base["lon"]], tooltip=f"📍 {base['name']}",
                          icon=folium.Icon(color=ic_c,icon=ic_i)).add_to(fmap)
            seen_b.add(base["name"])
        dest = r.get("destination_latlon")
        if dest and r.get("zone") not in seen_z:
            folium.Marker(list(dest), tooltip=f"🚨 Zone {r['zone']}",
                          icon=folium.Icon(color="orange",icon="exclamation-sign")).add_to(fmap)
            seen_z.add(r["zone"])

    for r in routes:
        if not r.get("success"): continue
        color = _rcolor(r["resource_type"]); emoji = _remoji(r["resource_type"])
        wpts  = r.get("waypoints", [])
        if len(wpts) < 2:
            dest = r.get("destination_latlon"); rk = r["resource_type"].lower().rstrip("s")
            base = _DEFAULT_BASES.get(rk)
            if base and dest: wpts = [(base["lat"],base["lon"]),dest]
            else: continue
        folium.PolyLine(wpts, color=color, weight=5, opacity=0.9,
                        tooltip=(f"{emoji} {r.get('unit_count',1)}× {r['resource_type']}\n"
                                 f"Zone {r['zone']} · {r.get('distance_km',0)} km "
                                 f"ETA {r.get('eta_minutes',0)} min")).add_to(fmap)
        for lat, lon in wpts[1:-1]:
            folium.CircleMarker([lat,lon],radius=3,color=color,fill=True,
                                fill_color=color,fill_opacity=0.8).add_to(fmap)
        folium.Marker(list(wpts[-1]), tooltip=f"{emoji} Zone {r['zone']}",
                      icon=folium.DivIcon(html=f'<div style="font-size:16px;color:{color};">▼</div>',
                                          icon_size=(16,16),icon_anchor=(8,8))).add_to(fmap)

    seen_types = sorted({r["resource_type"] for r in routes if r.get("success")})
    lines = "".join(f'<span style="color:{_rcolor(t)};font-size:16px;">━━</span> '
                    f'{_remoji(t)} {t.replace("_"," ").title()}<br>' for t in seen_types)
    fmap.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;top:12px;right:12px;z-index:9999;background:white;'
        f'padding:12px 16px;border-radius:8px;border:2px solid #333;font-family:Arial;'
        f'font-size:12px;box-shadow:3px 3px 8px rgba(0,0,0,.3);">'
        f'<b>🗺 Route Legend</b><br>{lines}'
        f'<span style="background:#f39c12;padding:0 5px;border-radius:3px;">■</span> Crisis Zone<br>'
        f'<hr style="margin:6px 0;"><span style="font-size:10px;color:#666;">'
        f'LangGraph · Real OSM waypoints</span></div>'))
    return fmap

# ============================================================================
#  STAGE 1 — FOLDER SCAN  (replaces single-image upload)
# ============================================================================

def stage_1():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">📂 Stage 1: Satellite Folder Scan</h2>',
                unsafe_allow_html=True)
    _phase_badge()

    # ── If scan already completed, show summary + proceed button ──────────────
    if st.session_state.get("scan_complete"):
        flood_path = st.session_state.get("flood_image_path")
        scan_results = st.session_state.get("scan_results", [])

        if flood_path:
            st.markdown(
                f'<div class="card-flood">🚨 <b>Flood detected</b> in '
                f'<code>{Path(flood_path).name}</code> — pipeline ready to fire.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="card-ok">✅ All images scanned — <b>no flood detected</b>.</div>',
                unsafe_allow_html=True)

        # Show scan results table
        if scan_results:
            df_rows = []
            for r in scan_results:
                if r.get("skipped"):
                    df_rows.append({"Image": r["name"], "Status": "⏩ skipped",
                                    "Flooded %": "—", "Max prob": "—", "Result": "—"})
                else:
                    df_rows.append({
                        "Image":     r["name"],
                        "Status":    "🚨 FLOOD" if r["is_flood"] else "✅ clear",
                        "Flooded %": f'{r["flooded_fraction"]*100:.1f}%',
                        "Max prob":  f'{r["max_prob"]:.3f}',
                        "Result":    _prob_bar(r["flooded_fraction"]),
                    })
            st.dataframe(pd.DataFrame(df_rows), use_container_width=True, hide_index=True)

        col_r, col_scan = st.columns(2)
        with col_r:
            if st.button("🔁 Re-scan folder", key="rescan"):
                for k in ["scan_complete","scan_results","flood_image_path","scan_flood_result"]:
                    st.session_state.pop(k, None)
                st.rerun()
        with col_scan:
            if flood_path and st.button("▶ START PIPELINE with flood image", key="start_pipe"):
                _log(f"Flood image confirmed: {Path(flood_path).name} — Phase 1 starting …")
                st.session_state["pipeline_phase"] = "running_phase1"
                st.session_state["stage"] = 1
                st.rerun()
            elif not flood_path:
                st.info("No flood image found — add more images to the folder and re-scan.")
        return

    # ── Folder path input ──────────────────────────────────────────────────────
    c_left, c_right = st.columns([3, 2])

    with c_left:
        st.markdown("### 📂 Image Folder")
        st.caption(
            "Enter the full path to your `satellite_images/` folder. "
            "All JPG / PNG / TIF images inside will be scanned in alphabetical order."
        )
        default_folder = str(Path(_ROOT) / "satellite_images")
        folder_str = st.text_input(
            "Folder path",
            value=st.session_state.get("folder_path", default_folder),
            placeholder=r"C:\Dev\Crisis_Management\satellite_images",
            key="folder_input",
        )
        st.session_state["folder_path"] = folder_str

        folder = Path(folder_str.strip()) if folder_str.strip() else None
        folder_ok = folder and folder.exists() and folder.is_dir()
        images = _collect_images(folder) if folder_ok else []

        if folder_ok:
            st.success(f"✅ Folder found — **{len(images)} image(s)** detected")
            if images:
                st.markdown("**Images in scan order:**")
                for i, p in enumerate(images, 1):
                    st.markdown(
                        f'<div class="scan-row" style="color:{THEME["text"]};">'
                        f'[{i:02d}] {p.name}</div>', unsafe_allow_html=True)
        elif folder_str.strip():
            st.error("❌ Folder not found — check the path")

    with c_right:
        st.markdown("### 📍 Location Metadata")
        st.caption(
            "Plain JPG/PNG images carry **no GPS data**, so the location must be entered "
            "manually. This is used by the Route Agent to convert pixel coordinates → "
            "lat/lon for OSM road routing. Enter the approximate **centre** of the area "
            "your images cover and how many kilometres across each image is."
        )
        lat = st.number_input("Centre Latitude",  value=float(st.session_state.get("meta_lat", 19.062061)), format="%.6f", key="mlat")
        lon = st.number_input("Centre Longitude", value=float(st.session_state.get("meta_lon", 72.863542)), format="%.6f", key="mlon")
        cov = st.number_input("Coverage (km)",    value=float(st.session_state.get("meta_cov", 1.6)),       min_value=0.1, key="mcov")
        st.session_state["meta_lat"] = lat
        st.session_state["meta_lon"] = lon
        st.session_state["meta_cov"] = cov

        st.markdown(
            f'<div class="card" style="font-size:11px;">'
            f'<b>How to find your lat/lon:</b><br>'
            f'Open Google Maps → right-click the centre of your image area → '
            f'copy the first two numbers shown.<br><br>'
            f'<b>Coverage km</b> = how wide the area is that one image covers. '
            f'For a city-block image ≈ 1–2 km. For a district image ≈ 5–10 km.'
            f'</div>', unsafe_allow_html=True)

    st.divider()

    # ── Scan button ───────────────────────────────────────────────────────────
    if not folder_ok or not images:
        st.info("Enter a valid folder path with images to begin scanning.")
        return

    st.markdown(
        f'<div class="card">'
        f'<b>Detection settings</b> (matches vision_agent thresholds):<br>'
        f'Pixel threshold: UNet prob ≥ <b>{PIXEL_THRESHOLD}</b> → pixel flagged flooded &nbsp;|&nbsp; '
        f'Image trigger: ≥ <b>{FLOOD_FRACTION*100:.0f}%</b> of pixels flooded → FLOOD confirmed'
        f'</div>', unsafe_allow_html=True)

    if st.button("🔍 START FOLDER SCAN", key="start_scan", use_container_width=True):
        _log(f"=== Folder scan started — {len(images)} image(s) ===")
        _log(f"Folder: {folder}")
        _log(f"Pixel threshold: {PIXEL_THRESHOLD}  |  Flood trigger: {FLOOD_FRACTION*100:.0f}% pixels")

        st.session_state["pipeline_phase"] = "scanning"

        # ── Live scan UI elements ──────────────────────────────────────────
        progress_bar  = st.progress(0, text="Initialising UNet model…")
        status_slot   = st.empty()
        img_col1, img_col2 = st.columns([1, 2])
        img_slot      = img_col1.empty()    # current image thumbnail
        metrics_slot  = img_col2.empty()    # live metric readout
        table_slot    = st.empty()          # running results table

        scan_results  = []
        flood_found   = False
        flood_path    = None

        for idx, img_path in enumerate(images):
            pct = (idx) / len(images)
            progress_bar.progress(pct, text=f"Scanning [{idx+1}/{len(images)}]: {img_path.name}")
            status_slot.info(f"🔬 Running UNet on **{img_path.name}**…")

            # Show thumbnail
            try:
                pil = Image.open(img_path)
                img_slot.image(pil, caption=img_path.name, use_container_width=True)
            except Exception:
                pass

            t0 = time.time()
            try:
                result = _analyse_image(img_path)
            except Exception as exc:
                _log(f"[ERROR] {img_path.name}: {exc}")
                scan_results.append({"name": img_path.name, "path": str(img_path),
                                     "is_flood": False, "flooded_fraction": 0,
                                     "max_prob": 0, "mean_prob": 0, "skipped": False,
                                     "error": str(exc)})
                continue

            elapsed = time.time() - t0

            # Live metrics display
            frac = result["flooded_fraction"]
            bar  = _prob_bar(frac)
            colour = THEME["red"] if result["is_flood"] else THEME["green"]
            metrics_slot.markdown(
                f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:12px;'
                f'background:{THEME["bg2"]};padding:12px;border-radius:6px;'
                f'border-left:4px solid {colour};">'
                f'<b style="color:{colour};">{"🚨 FLOOD" if result["is_flood"] else "✅ clear"}</b><br><br>'
                f'Flooded pixels : <b>{frac*100:.1f}%</b> (trigger ≥ {FLOOD_FRACTION*100:.0f}%)<br>'
                f'Max prob       : <b>{result["max_prob"]:.3f}</b><br>'
                f'Mean prob      : {result["mean_prob"]:.3f}<br>'
                f'Progress       : <span style="letter-spacing:1px;">{bar}</span><br>'
                f'Time           : {elapsed:.1f}s'
                f'</div>', unsafe_allow_html=True)

            _log(f'[{idx+1}/{len(images)}] {img_path.name} | '
                 f'flooded={frac*100:.1f}% max={result["max_prob"]:.3f} → '
                 f'{"FLOOD DETECTED" if result["is_flood"] else "clear"}')

            scan_results.append(result)

            # Update running table
            df_rows = [{"Image": r["name"],
                        "Flooded %": f'{r["flooded_fraction"]*100:.1f}%',
                        "Max prob":  f'{r["max_prob"]:.3f}',
                        "Status":    "🚨 FLOOD" if r["is_flood"] else "✅ clear"}
                       for r in scan_results]
            table_slot.dataframe(pd.DataFrame(df_rows), use_container_width=True, hide_index=True)

            if result["is_flood"]:
                flood_found = True
                flood_path  = str(img_path)
                progress_bar.progress(1.0, text="🚨 Flood detected — halting scan")
                status_slot.error(
                    f"🚨 **FLOOD DETECTED** in `{img_path.name}` — "
                    f"{frac*100:.1f}% of pixels flooded | max_prob={result['max_prob']:.3f}"
                )
                _log(f"FLOOD DETECTED — {img_path.name} — halting scan, pipeline ready")
                # Mark remaining as skipped
                for rem in images[idx+1:]:
                    scan_results.append({"name": rem.name, "path": str(rem),
                                         "is_flood": False, "flooded_fraction": 0,
                                         "max_prob": 0, "mean_prob": 0, "skipped": True})
                    _log(f"Skipped (flood already found): {rem.name}")
                break

        progress_bar.progress(1.0, text="Scan complete")

        # Save metadata for pipeline
        meta = {"center_lat": lat, "center_lon": lon, "coverage_km": cov,
                "width_px": 1024, "height_px": 1024}  # updated by vision_node
        if flood_path:
            try:
                pil_tmp = Image.open(flood_path)
                meta["width_px"]  = pil_tmp.width
                meta["height_px"] = pil_tmp.height
            except Exception:
                pass

        st.session_state["scan_results"]      = scan_results
        st.session_state["scan_complete"]     = True
        st.session_state["flood_found"]       = flood_found
        st.session_state["flood_image_path"]  = flood_path
        st.session_state["image_meta"]        = meta
        st.session_state["pipeline_phase"]    = "idle" if not flood_found else "idle"
        st.rerun()

# ============================================================================
#  STAGE 2 — ZONE MAP  (Phase 1 runs here on first load)
# ============================================================================

def stage_2():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">🗺️ Stage 2: Zone Map Analysis</h2>',
                unsafe_allow_html=True)
    _phase_badge()

    if st.session_state.get("pipeline_phase") == "running_phase1":
        # Use flood image from folder scan, fall back to upload_obj for compatibility
        img_path = (st.session_state.get("flood_image_path")
                    or "Images_for_testing/image.png")
        st.session_state["img_path"] = img_path
        meta = st.session_state.get("image_meta", _DEFAULT_META.copy())

        st.info(f"🚀 Running LangGraph Phase 1 on `{Path(img_path).name}` …")
        with st.spinner("vision → drones → LLM rescue plan  (~60-120 s)"):
            try:
                _run_phase1(img_path, meta)
                st.success("✅ Phase 1 complete — graph interrupted before admin_resource")
            except Exception as e:
                _log(f"[ERROR] Phase 1:\n{traceback.format_exc()}")
                st.error(f"Phase 1 failed: {e}"); st.code(traceback.format_exc())
                _nav(back=0); return
        st.rerun()

    gs = _graph_state(); zone_map = gs.get("zone_map", {})
    if not zone_map:
        st.warning("Zone map not in graph state yet. Check terminal for errors.")
        _terminal(); st.divider(); _nav(back=0); return

    c1, c2 = st.columns([3,2])
    with c1:
        grid = os.path.join(_ROOT,"zone_results","grid_output.jpg")
        if os.path.exists(grid):
            st.image(Image.open(grid), caption="Zone Severity Grid (10×10)",
                     use_container_width=True)
        flood_path = st.session_state.get("flood_image_path")
        if flood_path and os.path.exists(flood_path):
            st.image(Image.open(flood_path), caption=f"Flood image: {Path(flood_path).name}",
                     use_container_width=True)
    with c2:
        st.markdown("**Top Affected Zones**")
        top = sorted(zone_map.items(), key=lambda x:x[1].get("severity",0), reverse=True)[:15]
        st.dataframe(pd.DataFrame([{
            "Zone":zid, "Sev":f'{d.get("severity",0):.3f}',
            "Flood":f'{d.get("flood_score",0):.3f}',
            "Damage":f'{d.get("damage_score",0):.3f}',
            "Level":_sev_label(d.get("severity",0))}
            for zid,d in top]),
            use_container_width=True, hide_index=True)
        st.success(f"✅ {len(zone_map)} zones analysed via LangGraph Vision Node")

    _terminal(); st.divider()
    _nav(back=0, fwd=2, fwd_label="▶ PROCEED TO DRONE ALLOCATION")

# ============================================================================
#  STAGES 3–10  (unchanged from original — all read from LangGraph checkpoint)
# ============================================================================

def stage_3():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">🚁 Stage 3: Drone Allocation</h2>',
                unsafe_allow_html=True)
    _phase_badge()
    gs = _graph_state()
    alloc = gs.get("drone_allocation", {})
    most  = gs.get("most_affected_zones", [])

    if not alloc:
        st.warning("Drone allocation not in graph state yet.")
        _terminal(); st.divider(); _nav(back=1); return

    if most:
        st.info(f"Top affected zones (from crisis.db): **{', '.join(most)}**")

    n = min(len(alloc), 6)
    if n:
        cols = st.columns(n)
        for i,(d_id,z_id) in enumerate(list(alloc.items())[:n]):
            with cols[i]:
                st.markdown(
                    f'<div class="card" style="text-align:center;">'
                    f'<b style="color:{THEME["cyan"]};">{d_id.upper()}</b><br>'
                    f'<span style="font-size:22px;">🚁</span><br>→ <b>{z_id}</b><br>'
                    f'<span style="color:{THEME["green"]};font-size:12px;">✅ DISPATCHED</span>'
                    f'</div>', unsafe_allow_html=True)

    st.dataframe(pd.DataFrame([{"Drone":k,"Zone":v,"Status":"✅ Dispatched"}
                                for k,v in alloc.items()]),
                 use_container_width=True, hide_index=True)
    _terminal(); st.divider()
    _nav(back=1, fwd=3, fwd_label="▶ PROCEED TO GALLERY")


def stage_4():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">📸 Stage 4: Drone Imagery Gallery</h2>',
                unsafe_allow_html=True)
    _phase_badge()

    gs     = _graph_state()
    alloc  = gs.get("drone_allocation", {})
    counts = gs.get("people_counts", {})
    akv    = list(alloc.items())

    st.markdown(f'<h4 style="color:{THEME["cyan"]};">📷 Raw Drone Footage</h4>',
                unsafe_allow_html=True)

    imgs = {}
    p = Path(os.path.join(_ROOT, "zone_images"))
    if p.exists():
        for f in sorted(list(p.glob("*.jpg")) + list(p.glob("*.jpeg")) + list(p.glob("*.png"))):
            try: imgs[f.stem] = Image.open(f)
            except: pass

    if imgs:
        cols = st.columns(3)
        for idx, (name, img) in enumerate(list(imgs.items())[:9]):
            with cols[idx % 3]:
                st.image(img, use_container_width=True)
                d_id   = akv[idx % len(akv)][0] if akv else f"drone_{idx+1}"
                z_id   = alloc.get(d_id, "—")
                pcount = counts.get(z_id)
                if pcount is not None:
                    st.caption(f"**{name}** · {d_id} → {z_id}  |  👤 **{pcount} people**")
                else:
                    st.caption(f"**{name}** · {d_id} → {z_id}")
    else:
        st.info("No zone images found in `zone_images/`.")

    rp = Path(os.path.join(_ROOT, "zone_results"))
    annotated = {}
    if rp.exists():
        for f in sorted(rp.glob("*_analysis.jpg")):
            zone_key = f.stem.replace("_analysis", "")
            try: annotated[zone_key] = Image.open(f)
            except: pass

    if annotated:
        st.divider()
        st.markdown(f'<h4 style="color:{THEME["cyan"]};">🔍 YOLO Detection Results</h4>',
                    unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, (zone_id, img) in enumerate(annotated.items()):
            with cols[idx % 3]:
                st.image(img, use_container_width=True)
                st.caption(f"Zone **{zone_id}** — 👤 {counts.get(zone_id, 0)} people detected")

    if counts:
        st.divider()
        df = pd.DataFrame([{"Zone": k, "👤 People": v,
                             "Status": "✅ Detected" if v > 0 else "⚠️ 0 detected"}
                           for k, v in counts.items()])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.success(f"✅ **{sum(counts.values())} people** across **{len(counts)} zones**")

    _terminal(); st.divider()
    _nav(back=2, fwd=4, fwd_label="▶ PROCEED TO ANALYSIS")


def stage_5():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">📊 Stage 5: Zone Analysis Results</h2>',
                unsafe_allow_html=True)
    _phase_badge()
    gs = _graph_state()
    people = gs.get("people_counts", {})
    zm     = gs.get("zone_map", {})
    top    = gs.get("most_affected_zones", [])

    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("**Zone Severity, Flood & People**")
        zones = top or list(zm.keys())[:15]
        rows  = [{"Zone":z, "👤 People":people.get(z,0),
                  "Severity":f'{zm.get(z,{}).get("severity",0):.3f}',
                  "Flood":f'{zm.get(z,{}).get("flood_score",0):.3f}',
                  "Damage":f'{zm.get(z,{}).get("damage_score",0):.3f}',
                  "Level":_sev_label(zm.get(z,{}).get("severity",0))}
                 for z in zones]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if os.path.exists(DB_PATH):
            try:
                conn=sqlite3.connect(DB_PATH)
                df_db=pd.read_sql_query(
                    "SELECT zone_id,severity,flood_score,damage_score,people_count,last_updated "
                    "FROM zones ORDER BY severity DESC LIMIT 10", conn)
                conn.close()
                st.markdown("**📦 crisis.db — Live Snapshot**")
                st.dataframe(df_db, use_container_width=True, hide_index=True)
            except Exception as e:
                st.caption(f"DB read error: {e}")
    with c2:
        st.markdown("**Zone Grid + Detection Images**")
        shown = 0
        rp = Path(os.path.join(_ROOT,"zone_results"))
        if rp.exists():
            for f in sorted(list(rp.glob("*.jpg"))+list(rp.glob("*.png"))):
                if f.name.startswith("route_map"): continue
                try:
                    st.image(Image.open(f), caption=f.stem, use_container_width=True)
                    shown += 1
                except: pass
                if shown >= 5: break
        if not shown:
            st.info("Result images appear in `zone_results/` after Phase 1 runs.")

    _terminal(); st.divider()
    _nav(back=3, fwd=5, fwd_label="▶ PROCEED TO RESOURCE ALLOCATION")


def stage_6():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">📦 Stage 6: Resource Allocation</h2>',
                unsafe_allow_html=True)
    _phase_badge()
    gs   = _graph_state()
    plan = gs.get("rescue_plan", {})

    if not plan:
        st.warning("Rescue plan not in graph state yet — check terminal.")
        _terminal(); st.divider(); _nav(back=4); return

    st.success("✅ Rescue plan generated by Gemini LLM via rescue_decision_node")
    rows = []; totals = {}
    for z, alloc in plan.items():
        row = {"Zone": z}
        for rt, cnt in alloc.items():
            row[rt] = cnt; totals[rt] = totals.get(rt, 0) + cnt
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).fillna(0), use_container_width=True, hide_index=True)

    if totals:
        st.divider()
        mc = st.columns(len(totals))
        for i,(k,v) in enumerate(totals.items()):
            with mc[i]: st.metric(k.replace("_"," ").title(), int(v))

    _terminal(); st.divider()
    _nav(back=4, fwd=6, fwd_label="▶ PROCEED TO APPROVAL GATE")


def stage_7():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">✅ Stage 7: Admin Approval Gate #1 — Resources</h2>',
                unsafe_allow_html=True)
    _phase_badge()
    gs    = _graph_state()
    plan  = gs.get("rescue_plan", {})
    phase = st.session_state.get("pipeline_phase", "idle")

    if not plan:
        st.warning("No rescue plan — go back to Stage 6.")
        _nav(back=5); return

    if gs.get("resource_approved") or phase in ("awaiting_route","running_phase2","running_phase3","complete"):
        st.success("✅ Resource allocation APPROVED — route planning triggered.")
        _nav(back=5, fwd=7, fwd_label="▶ VIEW ROUTE PLANNING")
        return

    st.markdown("**Proposed Rescue Resource Allocation (Gemini LLM)**")
    for z, alloc in plan.items():
        desc = " · ".join(f"{v}× {k}" for k,v in alloc.items() if v)
        st.markdown(f'<div class="card"><b style="color:{THEME["cyan"]};">Zone {z}</b>  →  {desc}</div>',
                    unsafe_allow_html=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ APPROVE — TRIGGER ROUTE PLANNING", key="app1", use_container_width=True):
            _log("ADMIN ✓ Resources APPROVED — resuming LangGraph (Phase 2) …")
            st.session_state["pipeline_phase"] = "running_phase2"
            with st.spinner("🗺️ Route Agent planning OSM routes … (~30-60 s)"):
                try:
                    _run_phase2(approved=True)
                    st.success("✅ Routes planned!"); st.balloons()
                except Exception as e:
                    _log(f"[ERROR] Phase 2:\n{traceback.format_exc()}")
                    st.error(f"Phase 2 error: {e}"); st.code(traceback.format_exc())
            st.rerun()
    with c2:
        if st.button("🔴 REJECT — RE-RUN LLM", key="hold1", use_container_width=True):
            import uuid
            _log("ADMIN ✗ Rejected — restarting Phase 1 with new thread …")
            st.session_state["thread_id"] = f"aegis_{uuid.uuid4().hex[:8]}"
            img_path = st.session_state.get("flood_image_path") or "Images_for_testing/image.png"
            meta     = st.session_state.get("image_meta", _DEFAULT_META.copy())
            st.session_state["pipeline_phase"] = "running_phase1"
            with st.spinner("🔄 Re-running Phase 1 …"):
                try: _run_phase1(img_path, meta)
                except Exception as e: _log(f"[ERROR] Re-run: {e}")
            st.rerun()
    _terminal()


def stage_8():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">🗺️ Stage 8: Route Planning</h2>',
                unsafe_allow_html=True)
    _phase_badge()

    if st.session_state.get("pipeline_phase") == "running_phase2":
        st.info("🗺️ Route planning running … check terminal."); _terminal(); return

    gs     = _graph_state()
    routes = gs.get("route_plan", [])

    if not routes:
        st.warning("Route plan not in graph state yet.")
        _terminal(); st.divider(); _nav(back=6); return

    st.success(f"✅ {sum(1 for r in routes if r.get('success'))} / {len(routes)} routes planned")
    st.dataframe(pd.DataFrame([{
        "Zone":r.get("zone"),
        "Resource":f'{_remoji(r.get("resource_type",""))} {r.get("resource_type","")}',
        "Units":r.get("unit_count",1),
        "From":r.get("origin_name"),
        "Dist km":r.get("distance_km",0),
        "ETA min":r.get("eta_minutes",0),
        "Note":r.get("eta_note",""),
        "Status":"✓ OK" if r.get("success") else f'✗ {r.get("error","?")}',
    } for r in routes]), use_container_width=True, hide_index=True)

    st.markdown("**Interactive Route Map — Real OSM Waypoints**")
    st_folium(_folium_map(), width=None, height=480, key="fmap8", returned_objects=[])

    mp = gs.get("route_map_path")
    if mp and os.path.exists(mp):
        st.success(f"📄 Full HTML map saved: `{mp}`")

    _terminal(); st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("◀ BACK", key="b8"): st.session_state["stage"]=6; st.rerun()
    with c2:
        if st.button("🔄 Re-plan Routes", key="rp8"):
            st.session_state["pipeline_phase"] = "running_phase2"
            with st.spinner("🔄 Re-running route planner …"):
                try: _run_phase2(approved=True)
                except Exception as e: _log(f"[ERROR] {e}")
            st.rerun()
    with c3:
        if routes and st.button("▶ PROCEED TO APPROVAL #2", key="f8"):
            st.session_state["stage"] = 8; st.rerun()


def stage_9():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">✅ Stage 9: Admin Approval Gate #2 — Routes</h2>',
                unsafe_allow_html=True)
    _phase_badge()
    gs     = _graph_state()
    routes = gs.get("route_plan", [])
    phase  = st.session_state.get("pipeline_phase", "idle")

    if not routes:
        st.warning("No route plan — go back to Stage 8."); _nav(back=7); return

    if gs.get("route_approved") or phase in ("running_phase3","complete"):
        st.success("✅ Routes APPROVED — Communication Agent running / complete.")
        _nav(back=7, fwd=9, fwd_label="▶ VIEW DISPATCH COMMUNICATIONS")
        return

    st.markdown("**Review all planned routes before dispatching:**")
    st.dataframe(pd.DataFrame([{
        "Resource":f'{_remoji(r.get("resource_type",""))} {r.get("resource_type","")}',
        "Units":r.get("unit_count",1), "Zone":r.get("zone"),
        "From":r.get("origin_name"), "Dist km":r.get("distance_km",0),
        "ETA min":r.get("eta_minutes",0),
        "Status":"✓ OK" if r.get("success") else "✗ FAILED",
    } for r in routes]), use_container_width=True, hide_index=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ APPROVE ROUTES & DISPATCH", key="app2", use_container_width=True):
            _log("ADMIN ✓ Routes APPROVED — Phase 3 starting …")
            st.session_state["pipeline_phase"] = "running_phase3"
            with st.spinner("📡 Communication Agent generating dispatch instructions …"):
                try:
                    _run_phase3(approved=True)
                    st.success("✅ Dispatch instructions ready!"); st.balloons()
                except Exception as e:
                    _log(f"[ERROR] Phase 3:\n{traceback.format_exc()}")
                    st.error(f"Phase 3 error: {e}"); st.code(traceback.format_exc())
            st.rerun()
    with c2:
        if st.button("🔴 REJECT — RE-PLAN ROUTES", key="hold2", use_container_width=True):
            _log("ADMIN ✗ Routes REJECTED — re-running route planner …")
            st.session_state["pipeline_phase"] = "running_phase2"
            with st.spinner("🔄 Re-running route planner …"):
                try: _run_phase2(approved=True)
                except Exception as e: _log(f"[ERROR] {e}")
            st.rerun()
    _terminal()


def stage_10():
    st.markdown(f'<h2 style="color:{THEME["cyan"]};">📡 Stage 10: Communication Agent</h2>',
                unsafe_allow_html=True)
    _phase_badge()

    if st.session_state.get("pipeline_phase") == "running_phase3":
        st.info("📡 Communication Agent running … check terminal."); _terminal(); return

    gs       = _graph_state()
    dispatch = gs.get("dispatch_result", {})
    routes   = gs.get("route_plan", [])

    if not dispatch and not routes:
        st.warning("Dispatch data not available yet.")
        _terminal(); st.divider(); _nav(back=8); return

    if st.session_state.get("pipeline_phase") == "complete":
        st.markdown(
            f'<div style="background:{THEME["bg2"]};border:2px solid {THEME["green"]};'
            f'border-radius:8px;padding:16px;text-align:center;margin-bottom:16px;">'
            f'<span style="color:{THEME["green"]};font-family:\'Rajdhani\';font-size:24px;font-weight:bold;">'
            f'🎯 AEGIS PIPELINE COMPLETE</span><br>'
            f'<span style="color:{THEME["text"]};font-size:13px;">'
            f'All {len(STAGES)} stages executed via LangGraph master_graph</span></div>',
            unsafe_allow_html=True)

    instructions = (dispatch or {}).get("instructions", {})
    st.markdown("**📋 Dispatch Instructions (Gemini LLM)**")
    if instructions:
        for z, instr in instructions.items():
            text = instr if isinstance(instr, str) else json.dumps(instr, indent=2)
            st.markdown(
                f'<div class="card"><b style="color:{THEME["cyan"]};">Zone {z}</b><br>'
                f'<pre style="margin:8px 0 0;font-size:11px;color:{THEME["text"]};'
                f'white-space:pre-wrap;">{text}</pre></div>', unsafe_allow_html=True)
    else:
        for r in routes:
            em    = _remoji(r.get("resource_type",""))
            rtype = r.get("resource_type","").replace("_"," ").title()
            st.markdown(
                f'<div class="card"><b style="color:{THEME["cyan"]};">'
                f'{em} {r.get("unit_count",1)}× {rtype} → Zone {r.get("zone")}</b><br>'
                f'<span style="font-family:\'Share Tech Mono\';font-size:11px;">'
                f'From: {r.get("origin_name","?")} · {r.get("distance_km","?")} km · '
                f'ETA {r.get("eta_minutes","?")} min'
                f'</span></div>', unsafe_allow_html=True)

    summary = (dispatch or {}).get("summary","")
    if summary:
        st.info(f"**Commander Summary:** {summary}")

    st.markdown("**📱 SMS Dispatch Status**")
    from dotenv import load_dotenv; load_dotenv()
    sms_results  = (dispatch or {}).get("sms_results", [])
    twilio_configured = all([os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"),
                              os.getenv("TWILIO_PHONE_NUMBER"), os.getenv("YOUR_PHONE_NUMBER")])
    if sms_results:
        for res in sms_results:
            if res.get("success"):
                st.markdown(f'<div class="card-ok">✅ SMS sent → Zone <b>{res.get("zone","")}</b>  ·  '
                            f'SID: <code>{res.get("sid","")}</code></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="card-flood">❌ SMS FAILED → Zone <b>{res.get("zone","")}</b>  ·  '
                            f'{res.get("error","unknown error")}</div>', unsafe_allow_html=True)
    elif not twilio_configured:
        st.markdown(
            f'<div class="card-warn">⚠️ <b>SMS not sent</b> — Twilio credentials not set in <code>.env</code>.<br>'
            f'Dispatch instructions above were generated — only SMS delivery is skipped.</div>',
            unsafe_allow_html=True)

    audio = (dispatch or {}).get("audio_files", [])
    if audio:
        st.markdown("**🔊 Audio Dispatch Files**")
        for fpath in audio:
            if os.path.exists(fpath):
                st.audio(fpath); st.caption(fpath)

    _terminal(); st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("◀ BACK", key="b10"): st.session_state["stage"]=8; st.rerun()
    with c2:
        if st.button("🏁 COMPLETE & RESET", key="done10"):
            import uuid
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.session_state["thread_id"] = f"aegis_{uuid.uuid4().hex[:8]}"
            st.balloons(); st.rerun()

# ============================================================================
#  MAIN
# ============================================================================

STAGE_FNS = [stage_1,stage_2,stage_3,stage_4,stage_5,
             stage_6,stage_7,stage_8,stage_9,stage_10]

def main():
    _reset_nav_counter()
    st.session_state.setdefault("stage",0)
    st.session_state.setdefault("log","")
    st.session_state.setdefault("pipeline_phase","idle")
    st.session_state.setdefault("thread_id","aegis_main")
    _sidebar()
    st.markdown(
        f'<h1 style="color:{THEME["cyan"]};font-family:\'Rajdhani\';text-align:center;">'
        f'🛰️ AEGIS · Crisis Management AI</h1>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:{THEME["mono"]};text-align:center;font-family:\'Share Tech Mono\';">'
        f'Agentic Emergency Response &amp; Intelligence System  ·  '
        f'<b>LangGraph Master Agent</b></p>', unsafe_allow_html=True)
    st.divider()
    st.markdown(f'<p style="color:{THEME["text"]};font-family:\'Share Tech Mono\';font-size:12px;">'
                f'PIPELINE PROGRESS</p>', unsafe_allow_html=True)
    _stepper(); st.divider()
    STAGE_FNS[st.session_state.get("stage",0)]()

if __name__ == "__main__":
    main()