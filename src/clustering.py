
"""Spatial clustering for parking violation hotspot detection."""

import pandas as pd
import numpy as np
from typing import List, Tuple


def prepare_coordinates(df: pd.DataFrame,
                        sample_size: int = 50000) -> np.ndarray:
    """Extract lat/lng array, optionally sampling for performance."""
    coords = df[["latitude", "longitude"]].dropna()

    if len(coords) > sample_size:
        coords = coords.sample(n=sample_size, random_state=42)

    return coords.to_numpy()


def run_dbscan(coords: np.ndarray,
               eps: float = 0.0005,
               min_samples: int = 50) -> Tuple[np.ndarray, dict]:
    """Run DBSCAN clustering on GPS coordinates.

    eps=0.0005 degrees ≈ ~55 meters at Bangalore's latitude.
    Adjust based on desired hotspot granularity.

    Returns (labels, stats_dict).
    """
    from sklearn.cluster import DBSCAN

    # Use haversine-friendly approximation:
    # Convert lat/lng to meters (approximate for small areas)
    lat_scale = 111_320  # meters per degree latitude
    lng_scale = 111_320 * np.cos(np.radians(coords[:, 0].mean()))

    coords_meters = coords.copy()
    coords_meters[:, 0] *= lat_scale
    coords_meters[:, 1] *= lng_scale

    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = clusterer.fit_predict(coords_meters)

    # Per-cluster stats
    unique, counts = np.unique(labels, return_counts=True)
    stats = {}
    for label, count in zip(unique, counts):
        if label == -1:
            stats["noise"] = count
        else:
            mask = labels == label
            stats[f"cluster_{label}"] = {
                "count": int(count),
                "center_lat": float(coords[mask, 0].mean()),
                "center_lng": float(coords[mask, 1].mean()),
                "radius_m": float(
                    np.max(np.sqrt(
                        (coords_meters[mask, 0] - coords_meters[mask, 0].mean())**2
                        + (coords_meters[mask, 1] - coords_meters[mask, 1].mean())**2
                    ))
                ),
            }

    return labels, stats


def get_hotspots(df: pd.DataFrame,
                 eps_meters: float = 55,
                 min_violations: int = 50) -> pd.DataFrame:
    """High-level function: detect hotspots and return as DataFrame.

    Returns DataFrame with columns: center_lat, center_lng,
    violation_count, radius_m.
    """
    coords = prepare_coordinates(df)
    labels, stats = run_dbscan(coords, eps=eps_meters, min_samples=min_violations)

    hotspots = []
    for key, value in stats.items():
        if key == "noise":
            continue
        hotspots.append({
            "hotspot_id": key,
            "center_lat": value["center_lat"],
            "center_lng": value["center_lng"],
            "violation_count": value["count"],
            "radius_m": value["radius_m"],
        })

    result = pd.DataFrame(hotspots)
    if len(result) > 0:
        result = result.sort_values("violation_count", ascending=False)

    return result


def grid_aggregation(df: pd.DataFrame,
                     grid_size: float = 0.005) -> pd.DataFrame:
    """Simple grid-based aggregation as an alternative to DBSCAN.

    grid_size=0.005 degrees ≈ ~550m cells.
    """
    df = df.copy()
    df["lat_bin"] = (df["latitude"] // grid_size) * grid_size + grid_size / 2
    df["lng_bin"] = (df["longitude"] // grid_size) * grid_size + grid_size / 2

    grid = df.groupby(["lat_bin", "lng_bin"]).agg(
        violation_count=("latitude", "count"),
        unique_vehicles=("vehicle_type", "nunique"),
    ).reset_index()

    grid = grid[grid["violation_count"] >= 5]
    return grid.sort_values("violation_count", ascending=False)


if __name__ == "__main__":
    from src.data_loader import load_clean
    df = load_clean()
    hotspots = get_hotspots(df)
    print(f"Found {len(hotspots)} hotspots")
    print(hotspots.head())
