"""Parking intelligence analytics — observational only. No ranking of officers."""

import pandas as pd
import numpy as np


def station_summary(df):
    """Per-station metrics for informational comparison. NOT a ranking.

    Dimensions:
    - quality_score: ticket accuracy (may reflect device quality or zone complexity)
    - coverage_score: % of 24 hours with recorded activity
    - responsiveness_score: processing speed
    - balance_score: weekday/weekend fit
    - zone_complexity: estimated difficulty of the enforcement environment
    """
    summary = df.groupby("police_station").agg(
        total_violations=("police_station", "count"),
        approved=("is_approved", "sum"),
        rejected=("is_rejected", "sum"),
        unprocessed=("is_unprocessed", "sum"),
        median_processing_hours=("processing_hours", "median"),
        weekend_share=("is_weekend", "mean"),
        unique_vehicle_types=("vehicle_type", "nunique"),
        unique_junctions=("junction_name", "nunique"),
    ).reset_index()

    summary["approval_rate"] = (
        summary["approved"] / (summary["approved"] + summary["rejected"]) * 100
    )
    summary["rejection_rate"] = 100 - summary["approval_rate"]

    summary["zone_complexity"] = (
        summary["unique_vehicle_types"] * 0.6
        + summary["unique_junctions"].clip(lower=1) * 0.4
    )
    max_c = summary["zone_complexity"].max()
    summary["zone_complexity"] = summary["zone_complexity"] / max(max_c, 1) * 100

    summary["quality_score"] = (summary["approval_rate"] + summary["zone_complexity"] * 0.05).clip(upper=100)

    hourly = df.groupby(["police_station", "hour"]).size().reset_index(name="n")
    hourly["active"] = hourly["n"] >= 5
    coverage = hourly.groupby("police_station")["active"].sum().reset_index()
    coverage.columns = ["police_station", "active_hours"]
    coverage["coverage_score"] = coverage["active_hours"] / 24 * 100
    summary = summary.merge(coverage, on="police_station", how="left")
    summary["coverage_score"] = summary["coverage_score"].fillna(0)

    summary["responsiveness_score"] = (100 / summary["median_processing_hours"].clip(lower=1)).clip(upper=100)

    summary["weekend_share"] = summary["weekend_share"] * 100
    summary["balance_score"] = 100 - abs(summary["weekend_share"] - 28.6)
    summary["unprocessed_rate"] = summary["unprocessed"] / summary["total_violations"] * 100

    summary["overall_score"] = (
        summary["quality_score"] * 0.35
        + summary["coverage_score"] * 0.25
        + summary["responsiveness_score"] * 0.25
        + summary["balance_score"] * 0.15
    )

    return summary.round(1).sort_values("total_violations", ascending=False)


def peak_hour_profile(df):
    hourly = df.groupby("hour").size().reset_index(name="count")
    hourly["pct"] = hourly["count"] / hourly["count"].sum() * 100
    return hourly


def vehicle_fingerprint(df, station):
    sdf = df[df["police_station"] == station]
    if len(sdf) == 0:
        return {}
    counts = sdf["vehicle_type"].value_counts()
    return (counts / counts.sum() * 100).to_dict()


def weekday_vs_weekend(df):
    pivot = df.pivot_table(index="police_station", columns="is_weekend", aggfunc="size", fill_value=0)
    pivot.columns = ["weekday", "weekend"]
    pivot["total"] = pivot["weekday"] + pivot["weekend"]
    pivot["weekend_pct"] = pivot["weekend"] / pivot["total"] * 100
    pivot["weekday_pct"] = pivot["weekday"] / pivot["total"] * 100
    return pivot.sort_values("total", ascending=False)


def enforcement_gap_analysis(df):
    hdf = pd.DataFrame([
        {"hour": h, "violations": len(df[df["hour"] == h])} for h in range(24)
    ])
    enforcement = hdf[hdf["hour"].between(4, 7)]["violations"].sum()
    congestion = hdf[hdf["hour"].between(17, 20)]["violations"].sum()
    hdf["pct"] = hdf["violations"] / hdf["violations"].sum() * 100
    return hdf, enforcement, congestion


def violation_type_breakdown(df):
    counts = df["violation_primary"].value_counts().head(10)
    return pd.DataFrame({"violation_type": counts.index, "count": counts.values,
                         "pct": (counts.values / counts.sum() * 100).round(1)})
