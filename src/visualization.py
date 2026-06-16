"""Visualization helpers for ParkingIntel."""

import folium
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from folium.plugins import HeatMap

BANGALORE_CENTER = [12.9716, 77.5946]
DEFAULT_ZOOM = 12


def create_base_map(center=None, zoom=None):
    """Create a base Folium map centered on Bangalore."""
    return folium.Map(
        location=center or BANGALORE_CENTER,
        zoom_start=zoom or DEFAULT_ZOOM,
        tiles="OpenStreetMap",
        control_scale=True,
    )

def add_violation_heatmap(m, df, max_points=20000):
    """Add a heatmap layer for violation density."""
    sample = df.sample(n=min(len(df), max_points), random_state=42)
    heat_data = sample[["latitude", "longitude"]].values.tolist()
    HeatMap(heat_data, radius=12, blur=8, max_zoom=15).add_to(m)
    return m

def add_hotspot_markers(m, hotspots):
    """Add circle markers for detected hotspots."""
    for _, row in hotspots.iterrows():
        r = max(min(row["violation_count"] / 150, 40), 8)
        folium.CircleMarker(
            location=[row["center_lat"], row["center_lng"]],
            radius=r, color="#ff0000", weight=3,
            fill=True, fill_color="#ff4444", fill_opacity=0.4,
            popup=folium.Popup(
                f"<b>{row['hotspot_id']}</b><br>"
                f"{row['violation_count']:,} violations<br>"
                f"Radius: {row['radius_m']:.0f}m",
                max_width=250,
            ),
        ).add_to(m)
    return m


def plot_station_scorecard(summary, metric="rejection_rate"):
    labels = {
        "overall_score": "Overall Score", "quality_score": "Quality Score",
        "coverage_score": "Coverage Score", "responsiveness_score": "Responsiveness",
        "balance_score": "Balance Score", "rejection_rate": "Rejection Rate (%)",
        "zone_complexity": "Zone Complexity", "unprocessed_rate": "Backlog (%)",
    }
    top = summary.nlargest(15, "total_violations").sort_values(metric)
    if metric in ("rejection_rate", "unprocessed_rate"):
        colors = ["#ff6b35" if v > top[metric].median() else "#2ecc71" for v in top[metric]]
    else:
        colors = ["#2ecc71" if v > top[metric].median() else "#ff6b35" for v in top[metric]]
    fig = go.Figure(go.Bar(
        x=top[metric], y=top["police_station"], orientation="h",
        marker_color=colors, text=top[metric].round(1), textposition="outside",
    ))
    fig.update_layout(
        title=labels.get(metric, metric), xaxis_title=labels.get(metric, metric),
        template="plotly_dark", height=500, margin=dict(l=0, r=50, t=40, b=0),
    )
    return fig


def plot_hourly_heatmap(df):
    dow = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    p = df.pivot_table(index="day_of_week", columns="hour", aggfunc="size", fill_value=0)
    p.index = dow
    fig = go.Figure(go.Heatmap(
        z=p.values, x=list(range(24)), y=dow,
        colorscale="YlOrRd", colorbar=dict(title="Violations"),
    ))
    fig.update_layout(
        title="Violations by Day and Hour", xaxis_title="Hour",
        template="plotly_dark", height=400,
    )
    return fig


def plot_vehicle_pie(df, station=None):
    data = df[df["police_station"] == station] if station else df
    title = f"Vehicle Types - {station}" if station else "Vehicle Types - All Stations"
    c = data["vehicle_type"].value_counts().head(8)
    fig = go.Figure(go.Pie(labels=c.index, values=c.values, hole=0.4, textinfo="label+percent"))
    fig.update_layout(title=title, template="plotly_dark", height=450)
    return fig


def plot_weekend_scatter(weekend_df):
    top = weekend_df.head(20)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=top["weekday_pct"], y=top["weekend_pct"],
        mode="markers+text", text=top.index, textposition="top center",
        marker=dict(size=top["total"]/200, color=top["weekend_pct"],
                    colorscale="RdYlGn", showscale=True,
                    colorbar=dict(title="Weekend %")),
    ))
    fig.add_trace(go.Scatter(
        x=[0,100], y=[0,100], mode="lines",
        line=dict(dash="dash", color="gray"), showlegend=False,
    ))
    fig.update_layout(
        title="Weekday vs Weekend Split by Station",
        xaxis_title="Weekday %", yaxis_title="Weekend %",
        template="plotly_dark", height=500,
    )
    return fig


def plot_hourly_bar(hourly_df):
    fig = go.Figure(go.Bar(
        x=hourly_df["hour"], y=hourly_df["count"],
        marker_color=["#ff6b35" if 4<=h<=7 else "#4a90d9" for h in hourly_df["hour"]],
    ))
    fig.update_layout(
        title="Violations by Hour", xaxis_title="Hour",
        yaxis_title="Violations Recorded", template="plotly_dark", height=400,
    )
    return fig


def plot_blindspot_ranking(bs_df):
    top = bs_df.head(10).sort_values("combined_blind_spot_score")
    fig = go.Figure(go.Bar(
        x=top["combined_blind_spot_score"], y=top["police_station"],
        orientation="h", marker_color="#e74c3c",
        text=top["combined_blind_spot_score"].round(2), textposition="outside",
    ))
    fig.update_layout(
        title="Areas for Further Review", xaxis_title="Score",
        template="plotly_dark", height=450,
    )
    return fig
