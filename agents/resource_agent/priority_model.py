def compute_priority(severity_map, victims_map, wait_hours):

    severity_weights = {
        "Critical": 5,
        "Moderate": 3,
        "Low": 1
    }

    priority = {}

    for zone, severity in severity_map.items():

        victims = victims_map.get(zone, 0)
        wait = wait_hours.get(zone, 1)

        score = severity_weights[severity] * (victims + 1) * wait

        priority[zone] = score

    return priority