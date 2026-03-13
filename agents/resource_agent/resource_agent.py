from .priority_model import compute_priority
from .optimizer import optimize_allocation


def allocate_resources(severity_map, victims_map, wait_hours, resources):

    # step 1 compute zone priority
    priority = compute_priority(
        severity_map,
        victims_map,
        wait_hours
    )

    # step 2 optimize allocation
    allocation = optimize_allocation(
        priority,
        resources
    )

    return {
        "priority": priority,
        "allocation": allocation
    }