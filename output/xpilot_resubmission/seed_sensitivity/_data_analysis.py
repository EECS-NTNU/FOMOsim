"""Data analysis for hourly strategy results.

Reads the *_hourly.csv files produced by :mod:`preprocess` and computes
summary statistics rather than plotting.

Features:

* For each strategy and seed, compute the percentage of lost events at the
  final hour (failed/events*100).
* Display distributions (across seeds) of event and trip counts.  The script
  shows results for a single reference strategy (xpilot) since those totals
  are identical across strategies.
* Perform t-tests on the lost-event percentages to compare strategies and
  assess whether xpilot is statistically superior.

Usage::

    python _data_analysis.py

"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

DATA_DIR = Path(r"C:/Users/steffejb/OneDrive - NTNU/Work/GitHub/FOMO-sim/fomo/output/xpilot_resubmission/seed_sensitivity")
STRATEGIES = ["xpilot", "pilot", "greedy_neighbors", "greedy"]
SEEDS = range(1, 31)
TIME_COL = "Time"
FAILED_COL = "Failed events"
EVENTS_COL = "events"
TRIPS_COL = "trips"


def load_hourly(prefix: str, seed: int) -> pd.DataFrame | None:
    f = DATA_DIR / f"{prefix}_{seed}_hourly.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    # ensure time column is datetime
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    return df


def compute_percent_lost(df: pd.DataFrame) -> float:
    """Return percentage failed/events at the last record, or NaN."""
    if df.empty:
        return np.nan
    last = df.iloc[-1]
    if pd.isna(last[EVENTS_COL]) or last[EVENTS_COL] == 0:
        return np.nan
    return 100.0 * last[FAILED_COL] / last[EVENTS_COL]


# define checkpoint offsets (0-based index into hourly rows)
WEEK_HOUR = 7 * 24 - 1  # end of first week (hour 167)


def get_checkpoint(df: pd.DataFrame, idx: int):
    """Return (failed, events, pct_lost) at row index ``idx`` or nans."""
    if df.shape[0] <= idx:
        return (np.nan, np.nan, np.nan)
    row = df.iloc[idx]
    failed = row.get(FAILED_COL, np.nan)
    events = row.get(EVENTS_COL, np.nan)
    pct = np.nan
    if not pd.isna(events) and events != 0:
        pct = 100.0 * failed / events
    return (failed, events, pct)


def main():
    # gather stats for each seed/strategy at two checkpoints
    records = []
    for strat in STRATEGIES:
        for seed in SEEDS:
            df = load_hourly(strat, seed)
            if df is None:
                continue
            # obtain numeric values at the final row and at first-week hour
            failed_last, events_last, pct_last = get_checkpoint(df, df.shape[0] - 1)
            failed_week, events_week, pct_week = get_checkpoint(df, WEEK_HOUR)
            records.append({
                "strategy": strat,
                "seed": seed,
                "failed_last": failed_last,
                "events_last": events_last,
                "pct_last": pct_last,
                "failed_week": failed_week,
                "events_week": events_week,
                "pct_week": pct_week,
                "trips": df[TRIPS_COL].iloc[-1] if TRIPS_COL in df and not df.empty else np.nan,
            })
    stats_df = pd.DataFrame(records)

    # summaries for the two checkpoints
    print("=== final hour (last day) statistics ===")
    print(stats_df.groupby("strategy")[['failed_last', 'pct_last']].describe())
    print()
    print("=== first-week (hour 168) statistics ===")
    print(stats_df.groupby("strategy")[['failed_week', 'pct_week']].describe())
    print()

    # distributions of events/trips at final hour (xpilot as ref)
    ref = stats_df[stats_df["strategy"] == "xpilot"]
    print("events distribution for xpilot (final hour):")
    print(ref["events_last"].describe())
    print()
    print("trips distribution for xpilot (final hour):")
    print(ref["trips"].describe())
    print()

    # paired t-tests on failed events comparing xpilot to others at both checkpoints
    xpilot_last = stats_df[stats_df["strategy"] == "xpilot"]["failed_last"].values
    xpilot_week = stats_df[stats_df["strategy"] == "xpilot"]["failed_week"].values
    for strat in STRATEGIES:
        if strat == "xpilot":
            continue
        other_last = stats_df[stats_df["strategy"] == strat]["failed_last"].values
        other_week = stats_df[stats_df["strategy"] == strat]["failed_week"].values
        # ensure equal length and drop nans pairwise
        mask = ~np.isnan(xpilot_last) & ~np.isnan(other_last)
        if np.any(mask):
            t, p = stats.ttest_rel(xpilot_last[mask], other_last[mask])
            print(f"paired t-test (failed) xpilot vs {strat} at last hour: t={t:.3f}, p={p:.3g}")
        maskw = ~np.isnan(xpilot_week) & ~np.isnan(other_week)
        if np.any(maskw):
            t, p = stats.ttest_rel(xpilot_week[maskw], other_week[maskw])
            print(f"paired t-test (failed) xpilot vs {strat} at week hour: t={t:.3f}, p={p:.3g}")

    # final conclusion based on means
    # print mean percentage lost at the two checkpoints
    mean_pct_last = stats_df.groupby("strategy")["pct_last"].mean()
    mean_pct_week = stats_df.groupby("strategy")["pct_week"].mean()
    print("\nAverage percent lost at final hour (by strategy):")
    print(mean_pct_last.round(3))
    print("\nAverage percent lost at week hour (by strategy):")
    print(mean_pct_week.round(3))

    mean_last = stats_df.groupby("strategy")["failed_last"].mean()
    best_last = mean_last.idxmin()
    print(f"\nLowest average failed events at final hour: {best_last} ({mean_last.min():.1f})")
    mean_week = stats_df.groupby("strategy")["failed_week"].mean()
    best_week = mean_week.idxmin()
    print(f"Lowest average failed events at week hour: {best_week} ({mean_week.min():.1f})")
    if best_last == "xpilot" and best_week == "xpilot":
        print("xpilot is best at both checkpoints based on mean failed events.")
    else:
        print("xpilot is not best at all checkpoints.")

if __name__ == "__main__":
    main()
