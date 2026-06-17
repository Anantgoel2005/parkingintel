"""Enforcement Planner — predicts where violations will concentrate."""

import pandas as pd
import numpy as np


def predict_hotspots(df, day_of_week, start_hour, end_hour, n_officers=3):
    days = day_of_week if isinstance(day_of_week, (list, tuple)) else [day_of_week]
    mask = df["day_of_week"].isin(days) & df["hour"].between(start_hour, end_hour)
    window_df = df[mask]
    if len(window_df) == 0:
        return pd.DataFrame(columns=["latitude","longitude","expected_violations","priority_score","confidence"])

    grid_size = 0.005
    window_df = window_df.copy()
    window_df["lat_bin"] = (window_df["latitude"] // grid_size) * grid_size + grid_size / 2
    window_df["lng_bin"] = (window_df["longitude"] // grid_size) * grid_size + grid_size / 2

    zones = window_df.groupby(["lat_bin","lng_bin"]).agg(
        expected_violations=("latitude","count"),
        unique_vehicles=("vehicle_type","nunique"),
        unique_dates=("date","nunique"),
    ).reset_index()
    zones = zones[zones["unique_dates"] >= 3]

    max_date = pd.to_datetime(df["date"]).max()
    recent_cutoff = pd.to_datetime(max_date) - pd.Timedelta(days=30)
    recent = window_df[pd.to_datetime(window_df["date"]) >= recent_cutoff].groupby(["lat_bin","lng_bin"]).size().reset_index(name="recent_count")
    zones = zones.merge(recent, on=["lat_bin","lng_bin"], how="left")
    zones["recent_count"] = zones["recent_count"].fillna(0)
    zones["priority_score"] = zones["expected_violations"] + zones["recent_count"] * 0.5

    total_days = window_df["date"].nunique()
    zones["confidence"] = (zones["unique_dates"] / max(total_days, 1) * 100).clip(upper=100)

    n_zones = n_officers * 3
    result = zones.nlargest(n_zones, "priority_score")[
        ["lat_bin","lng_bin","expected_violations","priority_score","confidence"]
    ]
    result.columns = ["latitude","longitude","expected_violations","priority_score","confidence"]
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
        lines.append(
            f"**Stop {sn}:** ({lat:.4f}, {lng:.4f})"
            f" - ~{ev:.0f} violations ({cf:.0f}% confidence)"
        )
    return "  \n".join(lines)