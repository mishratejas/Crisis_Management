
"""
master_state.py
---------------
Defines the shared state TypedDict that flows through the LangGraph pipeline.

Fields are populated progressively as each node runs.

BUG FIX: Added missing fields that downstream nodes needed but could not get:
  • image_meta      — GPS coverage info, needed by route_planner_node
  • flood_mask      — raw float array from Vision Agent, needed by Route Agent
  • base_locations  — where ambulances/boats/etc. start from
  • resource_approved — was missing, caused KeyError in resource_approval_router
"""

from typing import TypedDict, Dict, List, Optional, Any


class MasterState(TypedDict):

    # ── Input (provided by caller in invoke()) ────────────────────────────────
    satellite_image: str       # file path to the satellite/aerial image

   #  Route Agent couldn't build
    # geo-transform without it. Must be provided in initial invoke() call
    # or defaults will be used (Prayagraj area).
    image_meta: Optional[Dict]  # {"center_lat", "center_lon", "coverage_km",
                                 #  "width_px", "height_px"}

    # Optional: override default base_locations for resources
    base_locations: Optional[Dict]  # {"ambulance": {"name":..,"lat":..,"lon":..}, ...}

    # ── Vision Agent output ───────────────────────────────────────────────────
    zone_map: Dict              # {"Z00": {"flood_score", "damage_score", "severity"}, ...}

   
    # Route Agent needs it to mark flooded roads as impassable.
    flood_mask: Optional[Any]   # float numpy array H×W (values 0–1) or None

    # ── Drone Analysis output ─────────────────────────────────────────────────
    most_affected_zones: List[str]   # e.g. ["Z35", "Z01", "Z72", "Z58", "Z24"]

    # ── Drone Decision & Dispatch output ─────────────────────────────────────
    drone_zones:     List[str]       # zones receiving drone coverage
    drone_allocation: Dict           # {"drone_1": "Z35", "drone_2": "Z01", ...}
    zone_image_map:  Dict            # {"Z35": "zone_images/img1.jpg", ...}

    # ── Drone Vision output ───────────────────────────────────────────────────
    people_counts: Dict              # {"Z35": 12, "Z01": 5, ...}

    # ── Rescue Decision output ────────────────────────────────────────────────
    rescue_plan: Optional[Dict]      # {"Z35": {"boats":2,"ambulances":1}, ...}

    # ── Admin Resource Approval output ───────────────────────────────────────
    
    resource_approved: Optional[bool]

    # ── Admin Route Approval output ───────────────────────────────────────────
    # True  → proceed to communication
    # False → loop back to route_planner for re-planning
    route_approved: Optional[bool]

    # ── Route Planner output ──────────────────────────────────────────────────
    route_plan: Optional[List]       # list of route dicts from plan_all_routes()

    # ── Communication Agent output ────────────────────────────────────────────
    dispatch_message: Optional[str]
    
    dispatch_result:  Optional[Dict]  # full output from dispatch_all():
                                      # {"instructions", "sms_results",
                                      #  "audio_files", "summary"}
    dispatch_config: Optional[Dict]   # {"language", "send_sms", "generate_audio",
                                      #  "to_number"}
    field_reports: Optional[List[str]] # raw reports from ground teams


# from typing import TypedDict, Dict, List, Optional


# class MasterState(TypedDict):

#     # Input
#     satellite_image: str

#     # Vision Agent output
#     zone_map: Dict

#     # Drone Analysis output (resource agent)
#     most_affected_zones: List[str]

#     # Drone Decision & Dispatch output
#     drone_zones: List[str]
#     drone_allocation: Dict         # { "drone_1": "Z23", ... }
#     zone_image_map: Dict           # { "Z23": "zone_images/img1.jpg", ... }

#     # Drone Vision output
#     people_counts: Dict            # { "Z23": 5, "Z45": 12, ... }

#     # (Future nodes)
#     rescue_plan: Optional[Dict]
#     route_plan: Optional[Dict]
#     dispatch_message: Optional[str]
