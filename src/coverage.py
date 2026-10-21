"""Market coverage audit for the CEDA Tamil Nadu paddy price data.

Run from the project root:   py src/coverage.py
Reads  data/raw/ceda_api_paddy_tn.csv
Writes data/processed/coverage_report.csv and prints a table.

Thresholds are starting guesses. Change them below once you see the numbers.
"""
from pathlib import Path

import pandas as pd

SRC = Path("data/raw/ceda_api_paddy_tn.csv")
OUT = Path("data/processed/coverage_report.csv")

MIN_SPAN_DAYS = 730         # at least ~2 years between first and last report
MIN_WEEKLY_COVERAGE = 0.60  # share of weeks in the span that have >= 1 report
MAX_GAP_DAYS = 60           # longest allowed silence between two reports
SPIKE_FACTOR = 5            # modal price this many times above/below the market median


def audit_market(g):
    g = g.sort_values("date")
    dates = g["date"].drop_duplicates()
    first, last = dates.min(), dates.max()
    span_days = (last - first).days + 1
    business_days = len(pd.bdate_range(first, last))
    weeks_total = (last - first).days // 7 + 1
    weeks_with_data = dates.dt.to_period("W").nunique()
    gaps = dates.diff().dt.days.dropna()
    median_price = g["modal_price"].median()
    return pd.Series({
        "rows": len(g),
        "first_date": first.date(),
        "last_date": last.date(),
        "span_days": span_days,
        "daily_coverage": round(len(dates) / business_days, 3),
        "weekly_coverage": round(weeks_with_data / weeks_total, 3),
        "median_gap_days": gaps.median() if len(gaps) else 0,
        "max_gap_days": gaps.max() if len(gaps) else 0,
        "stale_share": round((g["modal_price"].diff() == 0).mean(), 3),
        "zero_or_negative": int((g["modal_price"] <= 0).sum()),
        "spikes": int(((g["modal_price"] > SPIKE_FACTOR * median_price)
                       | (g["modal_price"] < median_price / SPIKE_FACTOR)).sum()),
    })


def main():
    df = pd.read_csv(SRC, parse_dates=["date"])
    report = (df.groupby(["district", "market"])
                .apply(audit_market, include_groups=False)
                .reset_index())
    report["usable"] = ((report["span_days"] >= MIN_SPAN_DAYS)
                        & (report["weekly_coverage"] >= MIN_WEEKLY_COVERAGE)
                        & (report["max_gap_days"] <= MAX_GAP_DAYS))
    report = report.sort_values("weekly_coverage", ascending=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUT, index=False)

    cols = ["district", "market", "rows", "first_date", "last_date",
            "daily_coverage", "weekly_coverage", "max_gap_days",
            "stale_share", "zero_or_negative", "spikes", "usable"]
    print(report[cols].to_string(index=False))
    print(f"\n{len(report)} markets, {int(report['usable'].sum())} pass the thresholds "
          f"(span >= {MIN_SPAN_DAYS} days, weekly coverage >= {MIN_WEEKLY_COVERAGE:.0%}, "
          f"max gap <= {MAX_GAP_DAYS} days).")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
