"""
verify_osm.py
=============
Run this after installing the packages from requirements.txt to confirm:
  1. OSMnx, shapely, geopandas, pyproj are all installed
  2. plan_all_routes(use_real_osm=True) routes on REAL OpenStreetMap roads
  3. The 9-node synthetic fallback is NOT being used

HOW TO RUN:
    cd <project_root>
    python verify_osm.py

WHAT "NOT SYNTHETIC" MEANS:
  - Synthetic graph has exactly 9 nodes and 20 edges
  - Synthetic routes have 2-5 waypoints (imaginary grid intersections)
  - Real OSM Prayagraj has 500-2000+ nodes and thousands of edges
  - Real routes have 10-200+ waypoints (every road bend and intersection)
"""

import sys
import os
import io
import contextlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "  ✅ PASS"
FAIL = "  ❌ FAIL"
INFO = "  ℹ  INFO"
all_ok = True

print("\n" + "="*60)
print("  OSMnx VERIFICATION — Crisis Route Agent")
print("="*60)

# ─── 1. Package imports ──────────────────────────────────
print("\n── 1. Package imports ───────────────────────────────")

REQUIRED = {
    "osmnx":     "pip install osmnx==2.0.1",
    "shapely":   "pip install shapely==2.1.0",
    "geopandas": "pip install geopandas==1.0.1",
    "pyproj":    "pip install pyproj==3.7.1",
    "pyogrio":   "pip install pyogrio==0.10.0",
}

import_ok = {}
for pkg, cmd in REQUIRED.items():
    try:
        m = __import__(pkg)
        v = getattr(m, "__version__", "?")
        print(f"{PASS}  {pkg}=={v}")
        import_ok[pkg] = True
    except ImportError:
        print(f"{FAIL}  {pkg} missing  →  run: {cmd}")
        import_ok[pkg] = False
        all_ok = False

if not import_ok.get("osmnx"):
    print("\n  Install order matters on Windows:")
    print("    pip install shapely==2.1.0")
    print("    pip install pyproj==3.7.1")
    print("    pip install pyogrio==0.10.0")
    print("    pip install geopandas==1.0.1")
    print("    pip install osmnx==2.0.1")
    sys.exit(1)

# ─── 2. OSMnx settings check (confirms BUG 1 fix) ───────
print("\n── 2. OSMnx settings (BUG 1 fix check) ─────────────")
import osmnx as ox
try:
    # BUG 1 was: ox.config() removed in >=1.3.0 → AttributeError at import
    # Fix: use ox.settings attributes directly
    ox.settings.use_cache   = True
    ox.settings.log_console = False
    print(f"{PASS}  ox.settings.use_cache and log_console set without error")
    print(f"{INFO}  osmnx version: {ox.__version__}")
except Exception as e:
    print(f"{FAIL}  {e}")
    all_ok = False

# ─── 3. OSMNX_AVAILABLE flag ─────────────────────────────
print("\n── 3. OSMNX_AVAILABLE flag in road_network.py ───────")
from agents.route_agent.road_network import (
    OSMNX_AVAILABLE, SHAPELY_AVAILABLE,
    download_road_network, build_synthetic_graph
)

if OSMNX_AVAILABLE:
    print(f"{PASS}  OSMNX_AVAILABLE = True")
else:
    print(f"{FAIL}  OSMNX_AVAILABLE = False despite osmnx being installed")
    print("       road_network.py import block failed — check for errors there")
    all_ok = False

if SHAPELY_AVAILABLE:
    print(f"{PASS}  SHAPELY_AVAILABLE = True")
else:
    print(f"{FAIL}  SHAPELY_AVAILABLE = False — flood mask blocking will not work")
    all_ok = False

# ─── 4. Download real OSM graph ──────────────────────────
print("\n── 4. Real OSM road graph download ──────────────────")

G_synth = build_synthetic_graph(25.435, 81.846)
print(f"{INFO}  Synthetic graph: {len(G_synth.nodes)} nodes, {len(G_synth.edges)} edges")

print("  Downloading Prayagraj road graph (cached after first run)…")
G_real = None
try:
    G_real = download_road_network(25.435, 81.846, radius_m=3000)
    n, e = len(G_real.nodes), len(G_real.edges)
    print(f"{INFO}  Real OSM graph: {n} nodes, {e} edges")

    if n > 50:
        print(f"{PASS}  {n} nodes >> 9 synthetic — real graph confirmed")
    else:
        print(f"{FAIL}  Only {n} nodes — still looks synthetic!")
        all_ok = False

    # BUG 2 check: travel_time must be set on every edge
    missing_tt = sum(1 for u,v,d in G_real.edges(data=True) if "travel_time" not in d)
    if missing_tt == 0:
        print(f"{PASS}  All edges have travel_time (BUG 2 fix confirmed)")
    else:
        print(f"{FAIL}  {missing_tt} edges missing travel_time — BUG 2 not fully fixed")
        all_ok = False

except Exception as e:
    print(f"{FAIL}  Download failed: {e}")
    print("       Check internet connection. OSMnx needs internet on first run.")
    all_ok = False

# ─── 5. plan_all_routes with use_real_osm=True ───────────
print("\n── 5. plan_all_routes(use_real_osm=True) ────────────")

from agents.route_agent.route_agent import plan_all_routes

IMAGE_META = {
    "center_lat": 25.435, "center_lon": 81.846,
    "coverage_km": 5.0, "width_px": 640, "height_px": 640,
}
BASE_LOCATIONS = {
    "ambulance":   {"name": "SRN Hospital Prayagraj", "lat": 25.4406, "lon": 81.8432},
    "rescue_team": {"name": "NDRF 9th Bn Prayagraj",  "lat": 25.4285, "lon": 81.8553},
    "boat":        {"name": "Sangam Nauka Ghat",       "lat": 25.4235, "lon": 81.8842},
}
ASSIGNMENTS = {
    "Z35": {"ambulances": 2, "rescue_teams": 1},
    "Z72": {"boats": 1},
}

# Capture stdout to check which graph mode was used
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    routes_real = plan_all_routes(
        image_meta=IMAGE_META,
        resource_assignments=ASSIGNMENTS,
        base_locations=BASE_LOCATIONS,
        use_real_osm=True,
    )
log = buf.getvalue()

# Check log output for which mode was selected
if "Using REAL OSM road graph" in log:
    print(f"{PASS}  Log: 'Using REAL OSM road graph (online mode)'")
elif "WARNING: use_real_osm=True but osmnx not installed" in log:
    print(f"{FAIL}  Fell back to synthetic! osmnx not detected at runtime")
    all_ok = False
else:
    print(f"{FAIL}  Unexpected mode — check log:")
    for line in log.splitlines():
        if "graph" in line.lower() or "synthetic" in line.lower() or "osm" in line.lower():
            print(f"       {line.strip()}")
    all_ok = False

# ─── 6. Prove results are NOT synthetic ──────────────────
print("\n── 6. Waypoint count comparison (key proof) ─────────")
print(f"  {'Route':<32} {'Synth wpts':>12} {'Real wpts':>12} {'Verdict'}")
print(f"  {'-'*32} {'-'*12} {'-'*12} {'-'*20}")

buf2 = io.StringIO()
with contextlib.redirect_stdout(buf2):
    routes_synth = plan_all_routes(
        image_meta=IMAGE_META,
        resource_assignments=ASSIGNMENTS,
        base_locations=BASE_LOCATIONS,
        use_real_osm=False,
    )

synth_map = {(r["zone"], r["resource_type"]): r for r in routes_synth}

for r in routes_real:
    key   = (r["zone"], r["resource_type"])
    s     = synth_map.get(key)
    label = f"{r['resource_type']}→{r['zone']}"
    sw    = len(s["waypoints"]) if s else 0
    rw    = len(r["waypoints"])

    if rw > sw and rw > 5:
        verdict = f"✅ REAL ({rw} road pts)"
    elif rw <= 5:
        verdict = f"❌ SYNTHETIC (only {rw} wpts)"
        all_ok = False
    else:
        verdict = f"⚠ CHECK ({rw} wpts)"

    print(f"  {label:<32} {sw:>12} {rw:>12}  {verdict}")

print()
print(f"{INFO}  Synthetic routes: 2-5 waypoints (imaginary 3x3 grid intersections)")
print(f"{INFO}  Real OSM routes:  10-300+ waypoints (every road bend and junction)")

# ─── 7. Distance sanity check ────────────────────────────
print("\n── 7. Distance sanity check ─────────────────────────")
print("  Real road distances should differ from direct-line distances.")
print("  Ratio > 1.0 means the route follows actual winding roads.\n")

import math
for r in routes_real:
    # Great-circle distance between origin and destination
    base_key = r["resource_type"].rstrip("s") if r["resource_type"].endswith("s") else r["resource_type"]
    base = BASE_LOCATIONS.get(base_key, list(BASE_LOCATIONS.values())[0])
    dest_lat, dest_lon = r["destination_latlon"]
    
    # Haversine formula
    lat1, lon1 = math.radians(base["lat"]), math.radians(base["lon"])
    lat2, lon2 = math.radians(dest_lat), math.radians(dest_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    direct_km = round(6371 * 2 * math.asin(math.sqrt(a)), 3)
    road_km   = r["distance_km"]
    
    if direct_km > 0:
        ratio = road_km / direct_km
        label = f"{r['resource_type']}→{r['zone']}"
        verdict = "✅ follows roads" if ratio >= 1.0 else "⚠ shorter than straight line?"
        print(f"  {label:<32}  direct={direct_km}km  road={road_km}km  ratio={ratio:.2f}  {verdict}")
    else:
        print(f"  {r['resource_type']}→{r['zone']}: same location (0 km)")

# ─── 8. Cache status ─────────────────────────────────────
print("\n── 8. Cache status ──────────────────────────────────")
cache_dir = os.path.join("agents", "route_agent", "_road_cache")
if os.path.isdir(cache_dir):
    files = os.listdir(cache_dir)
    total = sum(os.path.getsize(os.path.join(cache_dir,f)) for f in files) / 1e6
    print(f"{PASS}  Cache at {cache_dir}")
    print(f"{INFO}  {len(files)} file(s), {total:.1f} MB total")
    for f in files:
        sz = os.path.getsize(os.path.join(cache_dir,f)) / 1e6
        print(f"       {f}  ({sz:.1f} MB)")
    print(f"{INFO}  Next run will load from cache instantly (no download)")
else:
    print(f"{INFO}  No cache yet — will be created after first successful download")

# ─── FINAL RESULT ────────────────────────────────────────
print("\n" + "="*60)
if all_ok:
    print("  ✅  ALL CHECKS PASSED")
    print("      OSMnx is installed and working.")
    print("      plan_all_routes(use_real_osm=True) uses REAL road data.")
    print("      Synthetic graph is only used when use_real_osm=False.")
    print("      (Tests still use synthetic — that is intentional.)")
else:
    print("  ❌  SOME CHECKS FAILED — read messages above.")
    print("      Most common fix:")
    print("        pip install shapely==2.1.0 pyproj==3.7.1 pyogrio==0.10.0")
    print("        pip install geopandas==1.0.1 osmnx==2.0.1")
print("="*60 + "\n")

sys.exit(0 if all_ok else 1)