"""Download Tamil Nadu paddy prices from the CEDA Agmarknet API.

Run from the project root:   py src/download_ceda.py [FIRST_YEAR]
Raw responses are cached in data/raw/ceda_api/, so reruns skip finished work.
The API allows 40 calls per hour; the script stops before that and can be rerun.
"""
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("CEDA_API_KEY")
BASE = "https://api.ceda.ashoka.edu.in/v1/agmarknet"
COMMODITY_ID = 2   # Paddy(Dhan)(Common)
STATE_ID = 33      # Tamil Nadu
DISTRICTS = {
    "Thanjavur": 620, "Thiruvarur": 619, "Nagapattinam": 618,
    "Tiruchirappalli": 614, "Pudukkottai": 621, "Cuddalore": 617,
    "Ariyalur": 616, "Madurai": 623, "Tirunelveli": 628,
}
RAW = Path("data/raw/ceda_api")
OUT = Path("data/raw/ceda_api_paddy_tn.csv")  # not the old Gemini file name
CALL_BUDGET = 35      # stay under the 40-per-hour limit
PAUSE = 3             # seconds between calls
MIN_WINDOW_DAYS = 30  # smallest window we split down to after timeouts

calls = 0
last_status = None


def post(path, body):
    """One API call. Returns the parsed JSON, or None on timeout/server error."""
    global calls
    if calls >= CALL_BUDGET:
        sys.exit("Call budget used up. Wait an hour and rerun; finished work is cached.")
    calls += 1
    time.sleep(PAUSE)
    try:
        r = requests.post(f"{BASE}/{path}", json=body, timeout=150,
                          headers={"Authorization": f"Bearer {KEY}"})
    except requests.RequestException as e:
        print("  request error:", type(e).__name__)
        return None
    global last_status
    last_status = r.status_code
    print(f"  {path} -> HTTP {r.status_code}, calls left this hour: "
          f"{r.headers.get('ratelimit-remaining', '?')}")
    if r.status_code in (401, 403):
        sys.exit("API key rejected. Check CEDA_API_KEY in .env.")
    if r.status_code == 429:
        sys.exit("Rate limit hit. Wait for the hour to reset and rerun.")
    if r.status_code in (400, 422):
        print("  server says:", r.text[:400])
    if r.status_code != 200:
        return None
    try:
        res = r.json()
    except ValueError:
        return None
    if res.get("output", {}).get("type") != "success":
        return None
    return res


def get_markets(district_id):
    f = RAW / f"markets_{district_id}.json"
    if f.exists():
        return json.loads(f.read_text())
    # /markets takes a single district_id and an indicator (from its Swagger example)
    body = {"commodity_id": COMMODITY_ID, "state_id": STATE_ID,
            "district_id": district_id, "indicator": "price"}
    res = None
    for attempt in range(3):  # timeouts (504) are often temporary: retry
        res = post("markets", body)
        if res is not None:
            break
        if last_status in (400, 422):
            sys.exit("The server rejected the markets request body (see message above).")
        if attempt < 2:
            print("  retrying in 30 seconds...")
            time.sleep(30)
    if res is None:
        print("  skipping this district for now; rerun later to try it again")
        return None
    data = res["output"].get("data", [])
    f.write_text(json.dumps(data))
    return data


def fetch(district_id, market_ids, start, end):
    """Prices for [start, end]; splits the window in half if the server times out."""
    f = RAW / f"prices_{district_id}_{start}_{end}.json"
    if f.exists():
        return json.loads(f.read_text())
    marker = f.with_suffix(".split")  # remembers windows that timed out
    if not marker.exists():
        body = {"commodity_id": COMMODITY_ID, "state_id": STATE_ID,
                "district_id": [district_id], "market_id": market_ids,
                "from_date": str(start), "to_date": str(end)}
        res = post("prices", body)
        if res is not None:
            data = res["output"].get("data", [])
            f.write_text(json.dumps(data))
            return data
        if (end - start).days <= MIN_WINDOW_DAYS:
            print(f"  giving up on {start}..{end}")
            return []
        marker.touch()
    mid = start + (end - start) // 2
    return (fetch(district_id, market_ids, start, mid)
            + fetch(district_id, market_ids, mid + timedelta(days=1), end))


def build_csv():
    market_name = {}
    for f in RAW.glob("markets_*.json"):
        for m in json.loads(f.read_text()):
            market_name[m["market_id"]] = m["market_name"]
    rows = []
    for f in RAW.glob("prices_*.json"):
        rows += json.loads(f.read_text())
    if not rows:
        print("No price rows downloaded yet.")
        return
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["market"] = df["market_id"].map(market_name)
    district_name = {v: k for k, v in DISTRICTS.items()}
    df["district"] = df["census_district_id"].map(district_name)
    df = (df.drop_duplicates(["date", "market_id"])
            .sort_values(["district", "market", "date"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows, {df['market'].nunique()} markets -> {OUT}")


def main():
    if not KEY:
        sys.exit("CEDA_API_KEY not found in .env")
    RAW.mkdir(parents=True, exist_ok=True)
    if "--build" in sys.argv:  # no API calls: just build the CSV from saved files
        build_csv()
        return
    first_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
    today = date.today()
    for name, district_id in DISTRICTS.items():
        print(name)
        markets = get_markets(district_id)
        if not markets:
            print("  no markets found")
            continue
        ids = [m["market_id"] for m in markets]
        for year in range(first_year, today.year + 1):
            fetch(district_id, ids, date(year, 1, 1), min(date(year, 12, 31), today))
    build_csv()


if __name__ == "__main__":
    main()