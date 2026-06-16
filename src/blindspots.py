
"""Blind spot detection — find zones where enforcement is absent."""

import pandas as pd
import numpy as np


def vehicle_diversity_blindspots(df: pd.DataFrame) -> pd.DataFrame:
    """Method 1: Zones with high vehicle diversity but low violation counts.

    High diversity = active area with many vehicle types.
    Low violations in an active area = enforcement isn't reaching it.
    """
    station_stats = df.groupby("police_station").agg(
        total_violations=("police_station", "count"),
        unique_vehicle_types=("vehicle_type", "nunique"),
    ).reset_index()

    # Blind spot score: diversity normalized by violation count
    # High score = many vehicle types but few violations
    station_stats["blind_spot_score"] = (
        station_stats["unique_vehicle_types"]
        / np.log1p(station_stats["total_violations"])
    )

    # Flag stations above 90th percentile
    threshold = station_stats["blind_spot_score"].quantile(0.90)
    station_stats["is_blind_spot"] = station_stats["blind_spot_score"] >= threshold

    return station_stats.sort_values("blind_spot_score", ascending=False)


def time_coverage_blindspots(df: pd.DataFrame) -> pd.DataFrame:
    """Method 2: Stations with narrow enforcement time windows.

    If a station only records violations 4-7 AM, the remaining 21 hours
    are a blind spot.
    """
    # Count distinct hours with significant activity (>10 violations)
    hourly = df.groupby(["police_station", "hour"]).size().reset_index(name="count")
    hourly["is_active"] = hourly["count"] >= 10

    coverage = hourly.groupby("police_station").agg(
        active_hours=("is_active", "sum"),
        total_hours=("hour", "nunique"),
    ).reset_index()

    coverage["coverage_pct"] = coverage["active_hours"] / 24 * 100
    coverage["gap_score"] = 100 - coverage["coverage_pct"]

    return coverage.sort_values("coverage_pct")


def junction_density_blindspots(df: pd.DataFrame) -> pd.DataFrame:
    """Method 3: Stations covering many junctions but low ticket volume.

    High junction count + low violations = spread too thin.
    """
    junction_stats = df.groupby("police_station").agg(
        total_violations=("police_station", "count"),
        unique_junctions=("junction_name", "nunique"),
    ).reset_index()

    # Violations per junction — low = spread thin
    junction_stats["violations_per_junction"] = (
        junction_stats["total_violations"]
        / junction_stats["unique_junctions"].clip(lower=1)
    )

    # Flag stations below 10th percentile
    threshold = junction_stats["violations_per_junction"].quantile(0.10)
    junction_stats["is_spread_thin"] = (
        junction_stats["violations_per_junction"] <= threshold
    )

    return junction_stats.sort_values("violations_per_junction")


def combined_blind_spot_score(df: pd.DataFrame) -> pd.DataFrame:
    """Combine all three methods into a unified blind spot ranking."""
    div = vehicle_diversity_blindspots(df)
    time = time_coverage_blindspots(df)
    junc = junction_density_blindspots(df)

    # Merge all scores
    combined = div[["police_station", "blind_spot_score"]].copy()
    combined = combined.merge(
        time[["police_station", "gap_score"]], on="police_station", how="left"
    )
    combined = combined.merge(
        junc[["police_station", "violations_per_junction"]],
        on="police_station", how="left"
    )

    # Normalize and combine
    for col in ["blind_spot_score", "gap_score"]:
        max_val = combined[col].max()
        if max_val > 0:
            combined[f"{col}_norm"] = combined[col] / max_val

    # Invert violations_per_junction (lower = more blind-spot-like)
    max_junc = combined["violations_per_junction"].max()
    combined["spread_thin_norm"] = 1 - (
        combined["violations_per_junction"] / max(max_junc, 1)
    )

    combined["combined_blind_spot_score"] = (
        combined["blind_spot_score_norm"] * 0.4
        + combined["gap_score_norm"] * 0.35
        + combined["spread_thin_norm"] * 0.25
    )

    # Filter out data artifacts (e.g., "No Police Station")
    combined = combined[~combined["police_station"].isin(["No Police Station"])]

    return combined.sort_values("combined_blind_spot_score", ascending=False)


if __name__ == "__main__":
    from src.data_loader import load_clean
    df = load_clean()
    results = combined_blind_spot_score(df)
    print("Top 10 enforcement blind spots:")
    print(results.head(10)[["police_station", "combined_blind_spot_score"]])
