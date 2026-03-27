"""
enhanced_master_graph.py
------------------------
Optional enhanced version of master_graph that includes better debugging
and output capture capabilities. Use this if you want more granular control
over output logging from each node.
"""

from langgraph.graph import StateGraph, END
from master_agent.master_state import MasterState
from master_agent.master_nodes import (
    vision_node,
    store_zone_node,
    drone_analysis_node,
    drone_decision_node,
    drone_dispatch_node,
    drone_vision_node,
    update_people_node,
    rescue_decision_node,
    admin_resource_node,
    resource_approval_router,
    route_planner_node,
    admin_route_node,
    route_approval_router,
    communication_node,
)

# ── Build the enhanced graph ──────────────────────────────────────────────────

builder = StateGraph(MasterState)

# ── Register nodes ────────────────────────────────────────────────────────────

node_list = [
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
]

for node_name, node_func in node_list:
    builder.add_node(node_name, node_func)

# ── Entry point ───────────────────────────────────────────────────────────────

builder.set_entry_point("vision")

# ── Linear edges ──────────────────────────────────────────────────────────────

linear_edges = [
    ("vision",          "store_zone"),
    ("store_zone",      "drone_analysis"),
    ("drone_analysis",  "drone_decision"),
    ("drone_decision",  "drone_dispatch"),
    ("drone_dispatch",  "drone_vision"),
    ("drone_vision",    "update_people"),
    ("update_people",   "rescue_decision"),
    ("rescue_decision", "admin_resource"),
]

for source, target in linear_edges:
    builder.add_edge(source, target)

# ── Conditional: admin approves resources ─────────────────────────────────────

builder.add_conditional_edges(
    "admin_resource",
    resource_approval_router,
    {
        "approved": "route_planner",
        "rejected": "rescue_decision",
        "pending": END
    }
)

builder.add_edge("route_planner", "admin_route")

# ── Conditional: admin approves routes ────────────────────────────────────────

builder.add_conditional_edges(
    "admin_route",
    route_approval_router,
    {
        "approved": "communication",
        "rejected": "route_planner",
        "pending": END
    }
)

# ── Finalize ──────────────────────────────────────────────────────────────────

builder.add_edge("communication", END)

# ── Compile ───────────────────────────────────────────────────────────────────

enhanced_master_graph = builder.compile()
