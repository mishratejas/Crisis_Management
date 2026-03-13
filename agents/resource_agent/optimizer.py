from ortools.linear_solver import pywraplp


def optimize_allocation(priority, resources):

    solver = pywraplp.Solver.CreateSolver("SCIP")

    zones = list(priority.keys())

    # decision variables
    x = {}

    for zone in zones:
        x[zone] = solver.IntVar(0, 10, f"x_{zone}")

    # resource constraint
    solver.Add(sum(x[z] for z in zones) <= resources["ambulances"])

    # objective: maximize priority served
    solver.Maximize(
        sum(priority[z] * x[z] for z in zones)
    )

    status = solver.Solve()

    allocation = {}

    if status == pywraplp.Solver.OPTIMAL:

        for zone in zones:

            val = int(x[zone].solution_value())

            if val > 0:
                allocation[zone] = val

    return allocation