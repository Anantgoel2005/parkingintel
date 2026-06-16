
"""Data loading and preprocessing pipeline for parking violation data."""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# ── Configuration ────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_FILE = DATA_DIR / "raw" / "jan to may police violation_anonymized791b166.csv"
PROCESSED_DIR = DATA_DIR / "processed"

DATASET_DOWNLOAD_LINK = "[Google Drive link — add here]"

# ── Core columns we use ──────────────────────────────────
KEEP_COLUMNS = [
    "latitude", "longitude",
    "vehicle_type",
    "violation_type",
    "offence_code",
    "created_datetime",
    "police_station",
    "junction_name",
    "validation_status",
    "validation_timestamp",
]


def load_raw() -> pd.DataFrame:
    """Load the raw CSV, validating that the file exists."""
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_FILE}\n"
            f"Download it from: {DATASET_DOWNLOAD_LINK}\n"
            f"Then place it in data/raw/"
        )
    
    df = pd.read_csv(RAW_FILE, usecols=KEEP_COLUMNS, low_memory=False)
    print(f"Loaded {len(df):,} rows from raw CSV")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and feature-engineer the raw DataFrame."""
    df = df.copy()
    
    # ── Parse violation_type from JSON string ────────────
    def parse_violation_types(val):
        if pd.isna(val):
            return []
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return [str(val)]
    
    df["violation_list"] = df["violation_type"].apply(parse_violation_types)
    df["violation_count"] = df["violation_list"].apply(len)
    
    # Primary violation label (most common classification)
    df["violation_primary"] = df["violation_list"].apply(
        lambda x: x[0] if x else "UNKNOWN"
    )
    
    # Is it parking-related?
    parking_keywords = ["PARKING", "NO PARKING", "WRONG PARKING"]
    df["is_parking"] = df["violation_primary"].apply(
        lambda x: any(kw in str(x).upper() for kw in parking_keywords)
    )
    
    # ── Parse timestamps ─────────────────────────────────
    df["created_dt"] = pd.to_datetime(df["created_datetime"], errors="coerce")
    df["validated_dt"] = pd.to_datetime(df["validation_timestamp"], errors="coerce")
    
    # ── Temporal features ────────────────────────────────
    df["hour"] = df["created_dt"].dt.hour
    df["day_of_week"] = df["created_dt"].dt.dayofweek  # 0=Mon, 6=Sun
    df["day_name"] = df["created_dt"].dt.day_name()
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    df["date"] = df["created_dt"].dt.date
    df["month"] = df["created_dt"].dt.month
    
    # Time buckets
    hour_bins = [0, 4, 8, 12, 16, 20, 24]
    hour_labels = ["Night (0-4)", "Early AM (4-8)", "Morning (8-12)",
                   "Afternoon (12-16)", "Evening (16-20)", "Night (20-24)"]
    df["hour_bucket"] = pd.cut(df["hour"], bins=hour_bins, labels=hour_labels,
                                right=False, include_lowest=True)
    
    # ── Processing time (hours) ──────────────────────────
    df["processing_hours"] = (
        (df["validated_dt"] - df["created_dt"]).dt.total_seconds() / 3600
    )
    
    # Filter outliers (negative or >30 days)
    df.loc[
        (df["processing_hours"] < 0) | (df["processing_hours"] > 720),
        "processing_hours"
    ] = None
    
    # ── Validation status ────────────────────────────────
    df["is_approved"] = df["validation_status"] == "approved"
    df["is_rejected"] = df["validation_status"] == "rejected"
    df["is_unprocessed"] = df["validation_status"].isna()
    
    # ── Clean coordinates ────────────────────────────────
    df = df.dropna(subset=["latitude", "longitude"])
    
    return df


def load_clean() -> pd.DataFrame:
    """Load raw data and return cleaned DataFrame (cached to parquet)."""
    cache_path = PROCESSED_DIR / "violations_clean.parquet"
    
    if cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)
    
    df = load_raw()
    df = clean(df)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"Cached cleaned data to {cache_path}")
    
    return df


if __name__ == "__main__":
    df = load_clean()
    print(f"\nCleaned DataFrame: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"\nDate range: {df['created_dt'].min()} to {df['created_dt'].max()}")
    print(f"Parking-related: {df['is_parking'].mean()*100:.1f}%")
