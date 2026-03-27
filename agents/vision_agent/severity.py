def add_severity(zone_map,
                 flood_weight=0.5,
                 damage_weight=0.3,
                 building_weight=0.2):

    for zone_id, data in zone_map.items():

        flood = data.get("flood_score", 0)
        damage = data.get("damage_score", 0)
        building = data.get("building_score", 0)

        # 🔥 CORE INTELLIGENCE
        if building > 0.1:
            # Buildings present → flood matters
            severity = (
                flood_weight * flood +
                damage_weight * damage +
                building_weight * building
            )
        else:
            # No buildings → reduce flood importance
            severity = (
                0.2 * flood +   # 👈 heavily reduced
                0.8 * damage
            )

        data["severity"] = round(severity, 3)

    return zone_map