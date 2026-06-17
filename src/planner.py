"""Enforcement Planner — predicts where violations will concentrate."""

import pandas as pd
import numpy as np


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
    zones["priority_score"] = (
        zones["weighted_count"]
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

def suggest_patrol_order(zones_df):
    if len(zones_df) <= 1:
        zones_df["stop_number"] = 1
        return zones_df
    remaining = zones_df.copy()
    ordered = []
    current = remaining.iloc[0]
    ordered.append(current)
    remaining = remaining.iloc[1:]
    while len(remaining) > 0:
        d = np.sqrt((remaining["latitude"]-current["latitude"])**2 + (remaining["longitude"]-current["longitude"])**2)
        nearest = d.idxmin()
        current = remaining.loc[nearest]
        ordered.append(current)
        remaining = remaining.drop(nearest)
    result = pd.DataFrame(ordered).reset_index(drop=True)
    result["stop_number"] = range(1, len(result)+1)
    return result



def format_plan(zones_df, day_name, start_hour, end_hour, n_officers):
    if len(zones_df) == 0:
        return "No reliable hotspots found for this time window."
    total = zones_df["expected_violations"].sum()
    conf = zones_df["confidence"].mean()
    lines = [
        f"**Patrol Plan: {day_name}, {start_hour:02d}:00 - {end_hour:02d}:00**",
        "",
        f"{n_officers} officer(s) | {len(zones_df)} zones | ~{total:.0f} violations expected | {conf:.0f}% confidence",
        "",
    ]
    for _, row in zones_df.iterrows():
        sn = int(row["stop_number"])
        lat = row["latitude"]
        lng = row["longitude"]
        ev = row["expected_violations"]
        cf = row["confidence"]
        jn = " (near junction)" if row.get("junction_score", 0) > 0.3 else ""
        lines.append(f"**Stop {sn}:** ({lat:.4f}, {lng:.4f}){jn} - ~{ev:.0f} violations ({cf:.0f}% confidence)")
    return "  \n".join(lines)