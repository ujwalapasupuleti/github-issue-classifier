import numpy as np


def calculate_flow_metrics(
    total_issues,
    pending_issues,
    duplicate_issues,
    processing_capacity,
    time_minutes=60,
    hourly_history=None
):

    total_issues = max(total_issues, 0)
    pending_issues = max(pending_issues, 0)
    duplicate_issues = max(duplicate_issues, 0)
    processing_capacity = max(processing_capacity, 1)
    time_minutes = max(time_minutes, 1)

    # Fluid flow rate analogy
    flow_rate = total_issues / time_minutes

    # Backlog pressure relative to processing capacity
    pressure = pending_issues / processing_capacity

    # Recirculation caused by duplicate issues
    recirculation = (
        duplicate_issues / max(total_issues, 1)
    ) * 100

    # Flow efficiency
    processed = max(total_issues - pending_issues, 0)

    efficiency = (
        processed / max(total_issues, 1)
    ) * 100

    # Turbulence: coefficient of variation of issue arrivals
    if hourly_history is not None and len(hourly_history) > 1:
        values = np.array(hourly_history, dtype=float)
        mean = values.mean()

        turbulence = (
            values.std() / mean * 100
            if mean > 0 else 0
        )
    else:
        turbulence = 0

    if pressure > 1:
        status = "CRITICAL - High Backlog Pressure"
    elif pressure > 0.7:
        status = "WARNING - Increasing Pressure"
    else:
        status = "STABLE - Healthy Flow"

    return {
        "flow_rate": round(flow_rate, 3),
        "pressure": round(pressure, 3),
        "recirculation": round(recirculation, 2),
        "efficiency": round(efficiency, 2),
        "turbulence": round(turbulence, 2),
        "status": status
    }


if __name__ == "__main__":

    result = calculate_flow_metrics(
        total_issues=100,
        pending_issues=35,
        duplicate_issues=15,
        processing_capacity=50,
        time_minutes=60,
        hourly_history=[10, 15, 12, 40, 18, 50]
    )

    print("FLUID MECHANICS ANALYTICS")
    for key, value in result.items():
        print(key, ":", value)
