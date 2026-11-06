from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_COLS = [
    "sin_week", "cos_week",
    "is_kuruvai_harvest", "is_samba_harvest",
    "return_1w", "return_2w", "return_4w",
    "ratio_to_sma_4w", "ratio_to_sma_12w",
    "volatility_4w", "volatility_12w"
]
DATASETS = [
    ("Madurai", "features_madurai.csv"),
    ("Cuddalore", "features_cuddalore.csv"),
    ("Cauvery Delta Pooled", "features_cauvery_delta_pooled.csv")
]
def train_and_export(market_label: str, csv_file: str):
    slug = market_label.lower().replace(" ", "_")
    path = PROCESSED_DIR / csv_file
    if not path.exists():
        print(f"File not found: {csv_file}")
        return
    df = pd.read_csv(path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["target_return"] = (df["target_price_t1"] - df["modal_price_interpolated"]) / df["modal_price_interpolated"]
    clean_df = df.dropna(subset=FEATURE_COLS + ["target_return", "modal_price_interpolated"]).copy()
    train_df = clean_df[clean_df["week_start"] < "2024-01-01"]
    test_df = clean_df[clean_df["week_start"] >= "2024-01-01"].copy()
    model = lgb.LGBMRegressor(
        n_estimators=100, learning_rate=0.03, max_depth=3,
        num_leaves=7, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=-1
    )
    model.fit(train_df[FEATURE_COLS], train_df["target_return"])
    model_path = MODELS_DIR / f"lgbm_{slug}.joblib"
    joblib.dump(model, model_path)
    test_df["pred_return"] = model.predict(test_df[FEATURE_COLS])
    test_df["pred_price"] = test_df["modal_price_interpolated"] * (1.0 + test_df["pred_return"])
    plt.figure(figsize=(10, 4.5))
    plt.plot(test_df["week_start"], test_df["target_price_t1"], label="Actual Modal Price", color="#1f77b4", linewidth=2)
    plt.plot(test_df["week_start"], test_df["pred_price"], label="LightGBM 1-Wk Ahead Forecast", color="#d62728", linestyle="--", linewidth=2)
    plt.title(f"GrainCast Engine: Out-of-Sample Price Forecast (2024-2025) - {market_label}", fontsize=12, fontweight="bold")
    plt.xlabel("Week", fontsize=10)
    plt.ylabel("Price (Rs/Quintal)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    fig_path = FIGURES_DIR / f"forecast_eval_{slug}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    importance.to_csv(MODELS_DIR / f"importance_{slug}.csv", index=False)
    print(f"\nExported artifacts for {market_label}:")
    print(f"  - Model: {model_path}")
    print(f"  - Plot:  {fig_path}")
    print(f"  - Top Feature: {importance.iloc[0]['feature']} ({importance.iloc[0]['importance']} splits)")
if __name__ == "__main__":
    for label, fname in DATASETS:
        train_and_export(label, fname)
