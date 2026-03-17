from agents.resource_agent.resource_agent import allocate_resources


# Example severity output from Vision Agent
severity_map = {
    "E5": "Critical",
    "D4": "Critical",
    "G2": "Moderate",
    "A1": "Low"
}

# Example victims detected
victims_map = {
    "E5": 3,
    "D4": 2,
    "G2": 1
}

# Example waiting time (hours without help)
wait_hours = {
    "E5": 5,
    "D4": 3,
    "G2": 2,
    "A1": 1
}

# Available resources
resources = {
    "ambulances": 5
}

result = allocate_resources(
    severity_map,
    victims_map,
    wait_hours,
    resources
)

print("\n=== Resource Allocation Result ===\n")
print(result)