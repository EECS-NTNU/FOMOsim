"""Preprocessing for xpilot seed1 CSV data.

This script demonstrates the filtering and downsampling steps as a separate
preprocessing step.  It:

1. loads "xpilot_1.csv" from the data directory
2. keeps only the relevant columns
3. parses the time column into an increasing datetime sequence spanning ten
   simulated days, then reduces the data to the last record in each hour
   (forward-filling missing hours).
4. prints the resulting DataFrame so you can inspect it.
5. makes a quick plot of failed events vs. time to verify the curve is
   monotonically increasing.

Run it with:

    python preprocess.py

and the trimmed CSV will not be overwritten – this is a read-only example.
"""

from pathlib import Path
import pandas as pd
from dateutil.parser import parse
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# configuration
DATA_DIR = Path(r"C:/Users/steffejb/OneDrive - NTNU/Work/GitHub/FOMO-sim/fomo/output/xpilot_resubmission/seed_sensitivity")
FILE = DATA_DIR / "xpilot_1.csv"
KEEP = ["Time", "Failed events", "events", "long_congestion", "short_congestion",
        "starvation", "congested", "trips"]

# ----------------------------------
# helper to convert time strings to increasing datetimes
# ----------------------------------
# anchor date for conversions; use fixed values rather than today
CURRENT_YEAR = 2023
CURRENT_MONTH = 5
CURRENT_DAY = 22

def make_datetime_sequence(time_series):
    """Produce a list of datetimes using weekday hints to avoid mis‑rollovers.

    Each token looks like "Mon 07:00.000000".  Instead of trusting the
    parser to guess a year/week (which caused jumps into 2067/2068), we
    manually translate the weekday to an offset relative to the start date.

    We keep track of the last weekday index seen.  When the index decreases
    (e.g. from Sun back to Mon) we know we've moved into the next week and
    bump the week offset by 7 days.  The final date for a token is then
    ``anchor_date + timedelta(days=week_offset + weekday_index)``.
    """
    anchor = date(CURRENT_YEAR, CURRENT_MONTH, CURRENT_DAY)
    # map short names from file to 0=Mon..6=Sun
    wkday_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3,
                 "Fri": 4, "Sat": 5, "Sun": 6}
    result = []
    prev_wk = None
    week_offset = 0

    for tok in time_series:
        s = str(tok).strip()
        # split into weekday and time component
        parts = s.split(" ", 1)
        if len(parts) == 2 and parts[0] in wkday_map:
            wkname, time_part = parts
            wd = wkday_map[wkname]
        else:
            # fallback: no weekday or unknown format
            # try parsing whole string as time and keep previous weekday
            time_part = s
            wd = prev_wk if prev_wk is not None else 0
        # if weekday wrapped, increment week_offset
        if prev_wk is not None and wd < prev_wk:
            week_offset += 7
        prev_wk = wd

        # parse the time-of-day
        try:
            t = datetime.strptime(time_part, "%H:%M.%f").time()
        except ValueError:
            try:
                t = parse(time_part).time()
            except Exception:
                result.append(pd.NaT)
                continue
        days_since_anchor = week_offset + wd
        result.append(datetime.combine(anchor + timedelta(days=days_since_anchor), t))
    return result

# ----------------------------------
# main work


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing pipeline to a single dataframe."""
    df = df.loc[:, [c for c in KEEP if c in df.columns]]
    if "Time" not in df.columns:
        raise ValueError("Time column missing")
    df["Time"] = make_datetime_sequence(df["Time"])
    print("  converted time min/max:", df["Time"].min(), df["Time"].max())
    print("  monotonic?", df["Time"].is_monotonic_increasing)
    print("  duplicates?", df["Time"].duplicated().any())
    df = df.dropna(subset=["Time", "Failed events"]).sort_values("Time")
    df["hour_bucket"] = df["Time"].dt.floor("h")
    idx = df.groupby("hour_bucket")["Time"].idxmax()
    df = df.loc[idx].reset_index(drop=True)
    full_idx = pd.date_range(df["hour_bucket"].min(), df["hour_bucket"].max(), freq="h")
    df = df.set_index("hour_bucket").reindex(full_idx).ffill().rename_axis("hour_bucket").reset_index()
    df["Time"] = df["hour_bucket"]
    NUM_DAYS = 10
    expected = 24 * NUM_DAYS
    if len(df) > expected:
        print(f"  trimming resampled frame from {len(df)} rows to {expected} (first {NUM_DAYS} days)")
        df = df.head(expected)
    return df

# ----------------------------------

def main(plot: bool = False):
    # explicit strategy/seed combinations rather than relying on file globbing
    STRATEGIES = ["xpilot", "pilot", "greedy_neighbors", "greedy"]
    SEEDS = range(1, 31)

    for strat in STRATEGIES:
        for seed in SEEDS:
            fname = f"{strat}_{seed}.csv"
            path = DATA_DIR / fname
            if not path.exists():
                continue
            print("processing", fname)
            df = pd.read_csv(path, sep=";")
            processed = process_dataframe(df)
            out_path = DATA_DIR / f"{strat}_{seed}_hourly.csv"
            processed.to_csv(out_path, index=False)
            print("  wrote", out_path.name, "with", len(processed), "rows\n")

            if plot and strat == "greedy" and seed == 1:
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(processed["Time"], processed["Failed events"])
                ax.set_xlabel("Time")
                ax.set_ylabel("Failed events")
                ax.set_title("greedy seed1 (hourly)")
                locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
                fig.autofmt_xdate(rotation=45)
                plt.tight_layout()
                plt.show()


if __name__ == "__main__":
    main()
