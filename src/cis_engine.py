"""
Congestion Impact Score (CIS) — quantifies how much each parking violation
actually affects traffic flow. Simplified from PARKVISION's PCIS approach.

Components:
  1. Vehicle Severity — larger vehicles block more road
  2. Junction Proximity — violations near junctions cause more disruption
  3. Temporal Demand — rush hour violations have higher impact
  4. Density Multiplier — clustered violations compound each other
"""

import pandas as pd
import numpy as np

# Vehicle footprint scores (larger = more road blocked)
VEHICLE_SEVERITY = {
    "SCOOTER": 1.0, "MOPED": 1.0, "MOTOR CYCLE": 1.0,
    "CAR": 2.0,
    "PASSENGER AUTO": 2.5,
    "MAXI-CAB": 3.0, "VAN": 3.0,
    "LGV": 4.0, "GOODS AUTO": 4.0,
    "PRIVATE BUS": 6.0, "BUS (BMTC/KSRTC)": 6.0,
    "HGV": 8.0, "TANKER": 8.0, "LORRY/GOODS VEHICLE": 8.0,
}
DEFAULT_SEVERITY = 2.0

# Hourly demand curve (0-23, normalized to 1.0 at peak)
# Based on typical Bangalore traffic patterns
HOURLY_DEMAND = {
    0: 0.1, 1: 0.05, 2: 0.05, 3: 0.05,
    4: 0.2, 5: 0.3, 6: 0.5, 7: 0.8,
    8: 1.0, 9: 1.0, 10: 0.9, 11: 0.85,
    12: 0.8, 13: 0.75, 14: 0.7, 15: 0.75,
    16: 0.85, 17: 1.0, 18: 1.0, 19: 0.9,
    20: 0.8, 21: 0.6, 22: 0.4, 23: 0.2,
}


def compute_cis(df):
    """Compute Congestion Impact Score for each violation.

    Returns DataFrame with cis_score column added.
    CIS = vehicle_severity * junction_proximity * temporal_demand
    Range: ~0.05 (scooter at 3am, no junction) to ~24 (truck at 8am, at junction)
    """
    df = df.copy()

    # 1. Vehicle Severity
    df["vehicle_severity"] = df["vehicle_type"].map(VEHICLE_SEVERITY).fillna(DEFAULT_SEVERITY)

    # 2. Junction Proximity (1.0 = far from junction, 3.0 = at junction)
    df["near_junction"] = df["junction_name"].apply(
        lambda x: 3.0 if pd.notna(x) and x not in ("No Junction", "NULL", "") else 1.0
    )

    # 3. Temporal Demand
    df["temporal_demand"] = df["hour"].map(HOURLY_DEMAND).fillna(0.5)

    # Combined CIS
    df["cis_score"] = (
        df["vehicle_severity"]
        * df["near_junction"]
        * df["temporal_demand"]
    )

    return df


def compute_zone_cis(zone_df, violation_df):
    """Compute aggregate CIS for each zone (grid cell).

    Args:
        zone_df: DataFrame with lat_bin, lng_bin columns defining zones
        violation_df: DataFrame with cis_score column

    Returns:
        zone_df with cis_total, cis_mean, peak_impact_hour columns
    """
    if "cis_score" not in violation_df.columns:
        violation_df = compute_cis(violation_df)

    # Map violations to zones
    g = 0.003  # grid size
    v = violation_df.copy()
    v["lat_bin"] = (v["latitude"] // g) * g + g / 2
    v["lng_bin"] = (v["longitude"] // g) * g + g / 2

    # Find peak impact hour per zone
    peak = v.groupby(["lat_bin", "lng_bin", "hour"]).size().reset_index(name="hourly_count")
    peak["weighted"] = peak["hourly_count"] * peak["hour"].map(HOURLY_DEMAND).fillna(0.5)
    peak_hour = peak.loc[peak.groupby(["lat_bin", "lng_bin"])["weighted"].idxmax()][
        ["lat_bin", "lng_bin", "hour"]
    ].rename(columns={"hour": "peak_impact_hour"})

    # Aggregate CIS by zone
    agg = v.groupby(["lat_bin", "lng_bin"]).agg(
        cis_total=("cis_score", "sum"),
        cis_mean=("cis_score", "mean"),
        violation_count=("cis_score", "count"),
    ).reset_index()

    agg = agg.merge(peak_hour, on=["lat_bin", "lng_bin"], how="left")

    return agg
