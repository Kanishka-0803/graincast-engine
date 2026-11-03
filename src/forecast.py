from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
PROCESSED_DIR = Path("data/processed")
DATASETS = [
    ("Madurai", "features_madurai.csv"),
    ("Cuddalore", "features_cuddalore.csv"),
    ("Cauvery Delta Pooled", "features_cauvery_delta_pooled.csv")
]
FEATURE_COLS = [
    "sin_week", "cos_week",
    "is_kuruvai_harvest", "is_samba_harvest",
    "return_1w", "return_2w", "return_4w",
    "ratio_to_sma_4w", "ratio_to_sma_12w",
    "volatility_4w", "volatility_12w"
]
def evaluate_models(market_label: str, csv_file: str):
    path = PROCESSED_DIR / csv_file
    if not path.exists():
        print(f"Skipping {market_label}: {csv_file} does not exist.")
        return
    df = pd.read_csv(path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    # Calculate target return: (P_{t+1} - P_t) / P_t
    df["target_return"] = (df["target_price_t1"] - df["modal_price_interpolated"]) / df["modal_price_interpolated"]
    # Drop rows missing features or target
    df = df.dropna(subset=FEATURE_COLS + ["target_price_t1", "target_return", "modal_price_interpolated"]).copy()
    train_df = df[df["week_start"] < "2024-01-01"]
    test_df = df[df["week_start"] >= "2024-01-01"]
    y_test_price = test_df["target_price_t1"].values
    current_price = test_df["modal_price_interpolated"].values
    # Baselines
    y_pred_naive = current_price
    y_pred_sma4 = test_df["price_sma_4w"].values
    # LightGBM trained on stationary return
    X_train = train_df[FEATURE_COLS]
    y_train_return = train_df["target_return"]
    X_test = test_df[FEATURE_COLS]
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train_return)
    pred_return_lgb = model.predict(X_test)
    y_pred_lgb = current_price * (1.0 + pred_return_lgb)
    def calc_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        return round(mae, 2), round(rmse, 2), round(mape, 2)
    res_naive = calc_metrics(y_test_price, y_pred_naive)
    res_sma4 = calc_metrics(y_test_price, y_pred_sma4)
    res_lgb = calc_metrics(y_test_price, y_pred_lgb)
    print(f"\n=================== Market: {market_label} ===================")
    print(f"Train Weeks: {len(train_df)} | Test Weeks: {len(test_df)}")
    print(f"{'Model':<24} | {'MAE (Rs)':<10} | {'RMSE (Rs)':<10} | {'MAPE (%)':<10}")
    print("-" * 65)
    print(f"{'Naive Persistence':<24} | {res_naive[0]:<10} | {res_naive[1]:<10} | {res_naive[2]:<10}%")
    print(f"{'4-Week SMA':<24} | {res_sma4[0]:<10} | {res_sma4[1]:<10} | {res_sma4[2]:<10}%")
    print(f"{'LightGBM (Stationary)':<24} | {res_lgb[0]:<10} | {res_lgb[1]:<10} | {res_lgb[2]:<10}%")
if __name__ == "__main__":
    for label, fname in DATASETS:
        evaluate_models(label, fname)
