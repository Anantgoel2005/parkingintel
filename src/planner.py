"""Enforcement Planner — predicts where violations will concentrate."""

import pandas as pd
import numpy as np
from src.cis_engine import compute_cis, HOURLY_DEMAND, VEHICLE_SEVERITY


def predict_hotspots(df, day_of_week, start_hour, end_hour, n_officers=3):
    """Predict top hotspots with congestion-impact weighting.

    Accuracy improvements:
    - Finer grid (330m cells)
    - Commercial vehicles weighted 2x (more congestion impact)
    - Consistency bonus for zones that appear on many dates
    - Exponential recency decay (not flat 50%)
    - Junction proximity bonus
    """
    days = day_of_week if isinstance(day_of_week, (list, tuple)) else [day_of_week]
    mask = df["day_of_week"].isin(days) & df["hour"].between(start_hour, end_hour)
    window_df = df[mask].copy()
    if len(window_df) == 0:
        return pd.DataFrame(columns=["latitude","longitude","expected_violations","priority_score","confidence"])

    # Finer grid: 0.003 degrees (~330m at Bangalore's latitude)
    grid_size = 0.003
    window_df["lat_bin"] = (window_df["latitude"] // grid_size) * grid_size + grid_size / 2
    window_df["lng_bin"] = (window_df["longitude"] // grid_size) * grid_size + grid_size / 2

    # Commercial vehicles cause more congestion — weight them 2x
    commercial_types = ["LGV", "HGV", "TANKER", "MAXI-CAB", "PRIVATE BUS",
                        "BUS (BMTC/KSRTC)", "LORRY/GOODS VEHICLE"]
    window_df["weight"] = window_df["vehicle_type"].apply(
        lambda x: 2.0 if x in commercial_types else 1.0
    )

    # Aggregate by zone with weighted counts
    zones = window_df.groupby(["lat_bin","lng_bin"]).agg(
        raw_count=("latitude","count"),
        weighted_count=("weight","sum"),
        unique_vehicles=("vehicle_type","nunique"),
        unique_dates=("date","nunique"),
        # Junction bonus: zones with named junctions get higher priority
        has_junction=("junction_name", lambda x: (x.notna() & (x != "No Junction") & (x != "NULL")).sum()),
    ).reset_index()

    # Junction proximity score (0-1)
    zones["junction_score"] = (zones["has_junction"] / zones["raw_count"]).clip(upper=1.0)

    # Recency: exponential decay (newer = much more relevant)
    max_date = pd.to_datetime(df["date"]).max()
    window_df["days_ago"] = (max_date - pd.to_datetime(window_df["date"])).dt.days
    window_df["recency_weight"] = np.exp(-window_df["days_ago"] / 60)  # half-life ~42 days
    recency = window_df.groupby(["lat_bin","lng_bin"])["recency_weight"].sum().reset_index(name="recency_sum")
    zones = zones.merge(recency, on=["lat_bin","lng_bin"], how="left")
    zones["recency_sum"] = zones["recency_sum"].fillna(0)

    # Consistency: % of possible days this zone appears
    total_days = window_df["date"].nunique()
    zones["consistency"] = (zones["unique_dates"] / max(total_days, 1)).clip(upper=1.0)

    # Priority score:
    # weighted_count * (1 + junction_bonus * 0.3) * (1 + recency_factor) * consistency
    avg_recency = zones["recency_sum"].mean()
    zones["recency_factor"] = zones["recency_sum"] / max(avg_recency, 0.01)
    # Compute CIS for the window data to get congestion-aware weighting
    window_df_cis = compute_cis(window_df)
    window_df_cis["lat_bin"] = (window_df_cis["latitude"] // grid_size) * grid_size + grid_size / 2
    window_df_cis["lng_bin"] = (window_df_cis["longitude"] // grid_size) * grid_size + grid_size / 2
    cis_agg = window_df_cis.groupby(["lat_bin", "lng_bin"]).agg(
        cis_total=("cis_score", "sum"),
        cis_mean=("cis_score", "mean"),
    ).reset_index()
    zones = zones.merge(cis_agg, on=["lat_bin", "lng_bin"], how="left")
    zones["cis_total"] = zones["cis_total"].fillna(zones["weighted_count"])
    zones["cis_mean"] = zones["cis_mean"].fillna(2.0)

    # CIS factor: normalize to 0.5-2.0 range (higher congestion impact = higher priority)
    cis_median = zones["cis_mean"].median()
    zones["cis_factor"] = (zones["cis_mean"] / max(cis_median, 0.01)).clip(0.5, 3.0)

    zones["priority_score"] = (
        zones["weighted_count"]
        * zones["cis_factor"]
        * (1 + zones["junction_score"] * 0.3)
        * (1 + zones["recency_factor"] * 0.5)
        * (0.5 + zones["consistency"] * 0.5)
    )

    # Confidence: combination of consistency + vehicle diversity signal
    zones["confidence"] = (
        zones["consistency"] * 60
        + (zones["unique_vehicles"] / max(zones["unique_vehicles"].max(), 1)) * 40
    ).clip(upper=100)

    # Return top zones (3 per officer)
    n_zones = n_officers * 3
    result = zones.nlargest(n_zones, "priority_score")[
        ["lat_bin","lng_bin","weighted_count","priority_score","confidence","consistency","junction_score"]
    ]
    result.columns = ["latitude","longitude","expected_violations","priority_score","confidence","consistency","junction_score"]
    return result.reset_index(drop=True)

def cluster_and_assign(zones_df, n_officers, time_window_hours):
    """Assign officers to geographic clusters with realistic time constraints.

    - Clusters hotspots geographically (one cluster per officer)
    - Estimates travel time at 20 km/h (Bangalore average with traffic)
    - Assumes 20 min per stop for actual enforcement
    - Only includes zones reachable within the time window
    """
    if len(zones_df) == 0:
        return []

    from sklearn.cluster import KMeans
    import numpy as np
    from src.cis_engine import compute_cis, HOURLY_DEMAND, VEHICLE_SEVERITY

    # Convert lat/lng to approximate km for clustering
    lat_mean = zones_df["latitude"].mean()
    zones = zones_df.copy()
    zones["x_km"] = (zones["longitude"] - zones["longitude"].mean()) * 111.32 * np.cos(np.radians(lat_mean))
    zones["y_km"] = (zones["latitude"] - zones["latitude"].mean()) * 111.32

    # Cluster into n_officers groups (or fewer if not enough zones)
    n_clusters = min(n_officers, len(zones))
    if n_clusters > 1 and len(zones) >= n_clusters:
        coords = zones[["x_km", "y_km"]].values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        zones["cluster"] = kmeans.fit_predict(coords)
    else:
        zones["cluster"] = 0

    # Constants for time estimation
    AVG_SPEED_KMPH = 20  # Bangalore average with traffic
    MINUTES_PER_STOP = 20  # Time to actually enforce at each location
    TOTAL_MINUTES = time_window_hours * 60

    assignments = []
    for cluster_id in range(n_clusters):
        cluster_zones = zones[zones["cluster"] == cluster_id].copy()
        if len(cluster_zones) == 0:
            continue

        # Greedy nearest-neighbor route within cluster
        remaining = cluster_zones.reset_index(drop=True)
        route = []
        current = remaining.iloc[0]
        route.append(current)
        remaining = remaining.drop(0)

        while len(remaining) > 0:
            dists = np.sqrt(
                (remaining["x_km"] - current["x_km"])**2
                + (remaining["y_km"] - current["y_km"])**2
            )
            nearest = dists.idxmin()
            current = remaining.loc[nearest]
            route.append(current)
            remaining = remaining.drop(nearest)

        # Calculate time: travel between stops + enforcement at each stop
        total_km = 0
        feasible_stops = []
        time_used = 0

        # Start from an arbitrary central point (officer deployment)
        prev = route[0]

        for i, stop in enumerate(route):
            # Travel from previous stop
            travel_km = np.sqrt(
                (stop["x_km"] - prev["x_km"])**2
                + (stop["y_km"] - prev["y_km"])**2
            )
            travel_min = (travel_km / AVG_SPEED_KMPH) * 60
            stop_time = travel_min + MINUTES_PER_STOP

            if time_used + stop_time > TOTAL_MINUTES:
                break  # Can't reach this stop within the time window

            time_used += stop_time
            total_km += travel_km
            feasible_stops.append({
                **stop.to_dict(),
                "stop_number": len(feasible_stops) + 1,
                "travel_km": travel_km,
                "travel_min": travel_min,
                "arrive_by_min": time_used - MINUTES_PER_STOP,
            })
            prev = stop

        if feasible_stops:
            assignments.append({
                "officer_id": cluster_id + 1,
                "stops": feasible_stops,
                "total_zones": len(feasible_stops),
                "total_km": round(total_km, 1),
                "total_min": round(time_used, 0),
                "expected_violations": sum(s["expected_violations"] for s in feasible_stops),
            })

    return assignments


def format_plan(assignments, day_name, start_hour, end_hour, n_officers):
    """Generate a realistic, per-officer patrol plan."""
    if not assignments:
        return "No reliable hotspots found for this time window."

    window_hours = end_hour - start_hour
    lines = [
        f"**Patrol Plan: {day_name}, {start_hour:02d}:00 - {end_hour:02d}:00**",
        "",
        f"{n_officers} officer(s) | {window_hours}-hour window | Bangalore traffic estimate: 20 km/h",
        "",
    ]

    total_zones = sum(a["total_zones"] for a in assignments)
    total_violations = sum(a["expected_violations"] for a in assignments)
    lines.append(f"**Total: {total_zones} zones reachable | ~{total_violations:.0f} expected violations**")
    lines.append("")

    for a in assignments:
        lines.append(f"### Officer {a['officer_id']}")
        lines.append(f"{a['total_zones']} zones | ~{a['total_km']} km | ~{a['total_min']} min total")
        lines.append("")
        for stop in a["stops"]:
            lines.append(
                f"**Stop {stop['stop_number']}:** ({stop['latitude']:.4f}, {stop['longitude']:.4f}) | "
                f"~{stop['travel_min']:.0f} min travel | ~{stop['expected_violations']:.0f} violations "
                f"({stop['confidence']:.0f}% conf)"
            )
        lines.append("")

    return "  \n".join(lines)