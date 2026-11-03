"""
Feature engineering module for GrainCast Engine weekly paddy price series.
Constructs lags, rolling statistics, seasonal crop calendars (Kuruvai & Samba),
and multi-horizon forecasting targets.
"""
from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
DATASETS = [
    "weekly_madurai.csv",
    "weekly_cuddalore.csv",
    "weekly_cauvery_delta_pooled.csv"
]

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values("week_start").reset_index(drop=True)
    
    df["year"] = df["week_start"].dt.year
    df["month"] = df["week_start"].dt.month
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    
    df["sin_week"] = np.sin(2 * np.pi * df["week_of_year"] / 52.0)
    df["cos_week"] = np.cos(2 * np.pi * df["week_of_year"] / 52.0)
    
    # Tamil Nadu paddy harvest seasons
    df["is_kuruvai_harvest"] = df["week_of_year"].between(36, 44).astype(int)
    df["is_samba_harvest"] = df["week_of_year"].between(2, 13).astype(int)
    
    return df

def add_lag_and_rolling_features(df: pd.DataFrame, price_col: str = "modal_price_interpolated") -> pd.DataFrame:
    p = df[price_col]
    
    # Returns
    df["return_1w"] = (p / p.shift(1)) - 1.0
    df["return_2w"] = (p / p.shift(2)) - 1.0
    df["return_4w"] = (p / p.shift(4)) - 1.0
    
    # Price lags
    for lag in [1, 2, 3, 4, 8]:
        df[f"price_lag_{lag}"] = p.shift(lag)
        
    # Moving averages
    df["price_sma_4w"] = p.shift(1).rolling(window=4, min_periods=3).mean()
    df["price_sma_8w"] = p.shift(1).rolling(window=8, min_periods=6).mean()
    df["price_sma_12w"] = p.shift(1).rolling(window=12, min_periods=8).mean()
    
    # Momentum ratios
    df["ratio_to_sma_4w"] = p.shift(1) / df["price_sma_4w"]
    df["ratio_to_sma_12w"] = p.shift(1) / df["price_sma_12w"]
    
    # Volatility
    df["volatility_4w"] = df["return_1w"].shift(1).rolling(window=4, min_periods=3).std()
    df["volatility_12w"] = df["return_1w"].shift(1).rolling(window=12, min_periods=8).std()
    
    # Target: 1-week and 2-week ahead prices
    df["target_price_t1"] = p.shift(-1)
    df["target_price_t2"] = p.shift(-2)
    
    return df

def main():
    print("--- Generating Feature Matrix ---")
    for fname in DATASETS:
        in_path = PROCESSED_DIR / fname
        if not in_path.exists():
            print(f"Skipping {fname}: file not found.")
            continue
            
        df = pd.read_csv(in_path)
        df = add_time_features(df)
        df = add_lag_and_rolling_features(df)
        
        out_name = fname.replace("weekly_", "features_")
        out_path = PROCESSED_DIR / out_name
        df.to_csv(out_path, index=False)
        
        valid_rows = df.dropna(subset=["price_sma_12w", "target_price_t1", "target_price_t2"])
        print(f"Saved {out_name} -> Total weeks: {len(df)}, Usable training rows: {len(valid_rows)}")

if __name__ == "__main__":
    main()