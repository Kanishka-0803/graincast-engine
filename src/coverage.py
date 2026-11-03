"""Audit market reporting coverage and quality for Tamil Nadu paddy."""

from pathlib import Path
import numpy as np
import pandas as pd

RAW_CSV = Path("data/raw/ceda_api_paddy_tn.csv")
OUT_CSV = Path("data/processed/coverage_report.csv")

# Thresholds (Engineering choices - adjust based on business tolerance)
MIN_ROWS = 100
MIN_SPAN_DAYS = 365
MIN_WEEK_COVERAGE_PCT = 60.0
MAX_LONGEST_GAP_DAYS = 120
MAX_REPEAT_PRICE_PCT = 50.0


def audit_markets():
  if not RAW_CSV.exists():
    print(f"Error: Raw CSV not found at {RAW_CSV}")
    return

  df = pd.read_csv(RAW_CSV)
  df["date"] = pd.to_datetime(df["date"])

  records = []

  for (district, market), group in df.groupby(["district", "market"]):
    group = group.sort_values("date").drop_duplicates(subset=["date"])
    n_rows = len(group)

    min_date = group["date"].min()
    max_date = group["date"].max()
    span_days = (max_date - min_date).days + 1

    # Date diff gap
    gaps = group["date"].diff().dt.days
    longest_gap = int(gaps.max()) if not gaps.isna().all() else 0

    # Weekly coverage: unique calendar weeks with at least 1 record
    weeks_present = group["date"].dt.to_period("W").nunique()
    total_weeks = len(pd.period_range(min_date, max_date, freq="W"))
    week_cov_pct = (
        round((weeks_present / total_weeks) * 100, 1) if total_weeks > 0 else 0
    )

    # Price data quality checks
    repeated_prices = (group["modal_price"] == group["modal_price"].shift(1))
    repeat_price_pct = round((repeated_prices.sum() / max(1, n_rows - 1)) * 100, 1)

    zero_prices = int((group["modal_price"] <= 0).sum())
    spikes_10x = int(
        (group["modal_price"] > 10000).sum()
    )  # Basic sanity threshold

    # Pass/Fail evaluation
    passed = (
        n_rows >= MIN_ROWS
        and span_days >= MIN_SPAN_DAYS
        and week_cov_pct >= MIN_WEEK_COVERAGE_PCT
        and longest_gap <= MAX_LONGEST_GAP_DAYS
        and repeat_price_pct <= MAX_REPEAT_PRICE_PCT
    )

    records.append({
        "district": district,
        "market": market,
        "rows": n_rows,
        "start_date": min_date.strftime("%Y-%m-%d"),
        "end_date": max_date.strftime("%Y-%m-%d"),
        "span_days": span_days,
        "week_coverage_pct": week_cov_pct,
        "longest_gap_days": longest_gap,
        "repeat_price_pct": repeat_price_pct,
        "zero_prices": zero_prices,
        "spikes_above_10k": spikes_10x,
        "status": "PASS" if passed else "FAIL",
    })

  report = pd.DataFrame(records).sort_values(
      by=["status", "rows"], ascending=[True, False]
  )
  OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
  report.to_csv(OUT_CSV, index=False)

  print(f"\n--- Coverage Audit Complete ({len(report)} markets evaluated) ---")
  passed_df = report[report["status"] == "PASS"]
  print(f"Markets passing all quality thresholds: {len(passed_df)}")
  if not passed_df.empty:
    print(
        passed_df[[
            "district",
            "market",
            "rows",
            "week_coverage_pct",
            "longest_gap_days",
            "repeat_price_pct",
        ]].to_string(index=False)
    )
  print(f"\nFull report saved to: {OUT_CSV}")


if __name__ == "__main__":
  audit_markets()