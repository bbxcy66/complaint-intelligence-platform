from __future__ import annotations

import pandas as pd


def load_dashboard_df(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)

    # Ensure datetime
    if "date_received" in df.columns:
        df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")

    # Month bucket for trend chart
    df["month"] = df["date_received"].dt.to_period("M").dt.to_timestamp()
    return df


def monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["month", "product"])
        .size()
        .reset_index(name="count")
        .sort_values(["month", "product"])
    )
    return out