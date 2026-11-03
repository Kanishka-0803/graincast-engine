from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
PROCESSED_DIR = Path("data/processed")
# Government of India Paddy MSP (Common grade) by Marketing Season (Rs/quintal)
MSP_MAP = {
    2015: 1410, 2016: 1470, 2017: 1550, 2018: 1750, 2019: 1815,
    2020: 1868, 2021: 1940, 2022: 2040, 2023: 2183, 2024: 2300, 2025: 2300
}
FEATURE_COLS = [
    "sin_week", "cos_week",
    "is_kuruvai_harvest", "is_samba_harvest",
    "return_1w", "return_2w", "return_4w",
    "ratio_to_sma_4w", "ratio_to_sma_12w",
    "volatility_4w", "volatility_12w"
]
def generate_signals(market_label: str, csv_file: str):
    path = PROCESSED_DIR / csv_file
    if not path.exists():
        return
    df = pd.read_csv(path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["year"] = df["week_start"].dt.year
    df["msp"] = df["year"].map(MSP_MAP).fillna(2300)
    # Calculate stationary target return
    df["target_return"] = (df["target_price_t1"] - df["modal_price_interpolated"]) / df["modal_price_interpolated"]
    clean_df = df.dropna(subset=FEATURE_COLS + ["target_return", "modal_price_interpolated"]).copy()
    # Train model on data up to 2023
    train_df = clean_df[clean_df["week_start"] < "2024-01-01"]
    test_df = clean_df[clean_df["week_start"] >= "2024-01-01"]
    model = lgb.LGBMRegressor(
        n_estimators=100, learning_rate=0.03, max_depth=3,
        num_leaves=7, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=-1
    )
    model.fit(train_df[FEATURE_COLS], train_df["target_return"])
    # Predict on test set
    test_df = test_df.copy()
    test_df["pred_return"] = model.predict(test_df[FEATURE_COLS])
    test_df["pred_price_t1"] = test_df["modal_price_interpolated"] * (1.0 + test_df["pred_return"])
    # Procurement Signal Logic
    test_df["msp_spread_pct"] = ((test_df["pred_price_t1"] - test_df["msp"]) / test_df["msp"]) * 100.0
    def classify_signal(spread):
        if spread < -3.0:
            return "BUY_DPC_FLOOR (Farmer Distress Risk)"
        elif spread > 3.0:
            return "OPEN_MARKET_EXPEDIENT (Millers Dominant)"
        else:
            return "EQUILIBRIUM"
    test_df["procurement_signal"] = test_df["msp_spread_pct"].apply(classify_signal)
    out_file = PROCESSED_DIR / f"signals_{csv_file.replace('features_', '')}"
    test_df[["week_start", "market", "modal_price_interpolated", "msp", "pred_price_t1", "msp_spread_pct", "procurement_signal"]].to_csv(out_file, index=False)
    print(f"\n--- Procurement Signal Snapshot (Recent 5 Weeks): {market_label} ---")
    recent = test_df.tail(5)[["week_start", "modal_price_interpolated", "pred_price_t1", "msp", "msp_spread_pct", "procurement_signal"]]
    for _, row in recent.iterrows():
        dt_str = row['week_start'].strftime('%Y-%m-%d')
        print(f"{dt_str} | Actual: Rs {row['modal_price_interpolated']:<7.1f} | Forecast: Rs {row['pred_price_t1']:<7.1f} | MSP: Rs {row['msp']} | Spread: {row['msp_spread_pct']:>5.1f}% | Signal: {row['procurement_signal']}")
if __name__ == "__main__":
    generate_signals("Madurai", "features_madurai.csv")
    generate_signals("Cuddalore", "features_cuddalore.csv")
    generate_signals("Cauvery Delta Pooled", "features_cauvery_delta_pooled.csv")
