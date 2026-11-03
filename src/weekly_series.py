"""Weekly aggregation pipeline for Tamil Nadu paddy prices.

Builds continuous weekly time-series for single benchmark mandis (Madurai,
Cuddalore)
and a regional pooled series for the Cauvery Delta.
"""

from pathlib import Path
import numpy as np
import pandas as pd

RAW_CSV = Path("data/raw/ceda_api_paddy_tn.csv")
OUT_DIR = Path("data/processed")

DELTA_DISTRICTS = [
    "Thanjavur",
    "Thiruvarur",
    "Nagapattinam",
    "Tiruchirappalli",
    "Pudukkottai",
    "Ariyalur",
]


def load_and_clean_raw(path: Path) -> pd.DataFrame:
  """Load raw CEDA dataset and filter invalid prices."""
  df = pd.read_csv(path)
  df["date"] = pd.to_datetime(df["date"])

  # Ensure numeric price columns
  for col in ["min_price", "max_price", "modal_price"]:
    if col in df.columns:
      df[col] = pd.to_numeric(df[col], errors="coerce")

  # Filter non-positive values and extreme erroneous spikes (> 10,000 Rs/quintal)
  df = df[df["modal_price"] > 0]
  df = df[df["modal_price"] <= 10000]

  return df


def aggregate_weekly_mandi(df: pd.DataFrame, market_name: str) -> pd.DataFrame:
  """Aggregate a single market into a continuous Monday-anchored weekly median series."""
  sub = df[df["market"] == market_name].copy()
  if sub.empty:
    raise ValueError(f"Market '{market_name}' not found in dataset.")

  # Group by calendar week (Monday as start of week)
  sub["week_start"] = (
      sub["date"].dt.to_period("W-SUN").apply(lambda r: r.start_time)
  )

  weekly = (
      sub.groupby("week_start")
      .agg(
          modal_price_median=("modal_price", "median"),
          modal_price_mean=("modal_price", "mean"),
          modal_price_min=("modal_price", "min"),
          modal_price_max=("modal_price", "max"),
          days_reported=("modal_price", "count"),
      )
      .reset_index()
  )

  # Reindex to full continuous weekly date range
  full_weeks = pd.date_range(
      start=weekly["week_start"].min(),
      end=weekly["week_start"].max(),
      freq="W-MON",
      name="week_start",
  )

  weekly = weekly.set_index("week_start").reindex(full_weeks).reset_index()
  weekly["market"] = market_name
  weekly["series_type"] = "single_mandi"
  weekly["days_reported"] = weekly["days_reported"].fillna(0).astype(int)

  # Linear interpolation for isolated missing weeks (up to 3 consecutive weeks)
  weekly["modal_price_interpolated"] = (
      weekly["modal_price_median"].interpolate(method="linear", limit=3).round(2)
  )

  return weekly


def aggregate_delta_pooled(df: pd.DataFrame) -> pd.DataFrame:
  """Aggregate all reporting mandis in Cauvery Delta into a pooled weekly median series."""
  sub = df[df["district"].isin(DELTA_DISTRICTS)].copy()
  if sub.empty:
    raise ValueError("No delta district records found.")

  sub["week_start"] = (
      sub["date"].dt.to_period("W-SUN").apply(lambda r: r.start_time)
  )

  pooled = (
      sub.groupby("week_start")
      .agg(
          modal_price_median=("modal_price", "median"),
          modal_price_mean=("modal_price", "mean"),
          active_markets=("market", "nunique"),
          total_reports=("modal_price", "count"),
      )
      .reset_index()
  )

  full_weeks = pd.date_range(
      start=pooled["week_start"].min(),
      end=pooled["week_start"].max(),
      freq="W-MON",
      name="week_start",
  )

  pooled = pooled.set_index("week_start").reindex(full_weeks).reset_index()
  pooled["market"] = "Cauvery_Delta_Pooled"
  pooled["series_type"] = "regional_pooled"
  pooled["active_markets"] = pooled["active_markets"].fillna(0).astype(int)
  pooled["total_reports"] = pooled["total_reports"].fillna(0).astype(int)

  pooled["modal_price_interpolated"] = (
      pooled["modal_price_median"].interpolate(method="linear", limit=3).round(2)
  )

  return pooled


def main():
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  df = load_and_clean_raw(RAW_CSV)
  print(f"Loaded {len(df)} validated price records from {RAW_CSV}")

  # 1. Single Mandis
  for mandi in ["Madurai", "Cuddalore"]:
    series = aggregate_weekly_mandi(df, mandi)
    out_path = OUT_DIR / f"weekly_{mandi.lower()}.csv"
    series.to_csv(out_path, index=False)
    missing_pct = round(series["modal_price_median"].isna().mean() * 100, 2)
    print(
        f"Saved {mandi} -> {out_path} ({len(series)} weeks, missing:"
        f" {missing_pct}%)"
    )

  # 2. Regional Delta Pooled
  delta = aggregate_delta_pooled(df)
  delta_path = OUT_DIR / "weekly_cauvery_delta_pooled.csv"
  delta.to_csv(delta_path, index=False)
  delta_missing_pct = round(delta["modal_price_median"].isna().mean() * 100, 2)
  print(
      f"Saved Delta Pooled -> {delta_path} ({len(delta)} weeks, missing:"
      f" {delta_missing_pct}%)"
  )


if __name__ == "__main__":
  main()