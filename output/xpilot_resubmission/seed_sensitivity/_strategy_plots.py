"""Quick script for comparing failed-event curves of different strategies.

This file reads one seed (seed 1) from a handful of CSV sets identified by
prefix ("xpilot", "pilot", "greedy_neighbours", "greedy"), trims each file
keeping only the columns listed in *keep*, and then plots a single line per
strategy showing how the number of failed events evolves over time.

Running the script will modify the CSVs in-place the first time (removing the
unwanted columns) so subsequent runs are much faster.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, date

# ---------- configuration ----------
DATA_DIR = Path(r"C:/Users/steffejb/OneDrive - NTNU/Work/GitHub/FOMO-sim/fomo/output/xpilot_resubmission/seed_sensitivity")
# these definitions are mostly vestigial now that preprocessing has
# produced hourly CSVs; we only need the time and value cols for plotting.
STRATEGIES = ["xpilot", "pilot", "greedy_neighbors", "greedy"]
# set SEED to a specific integer to plot only that seed, or None to load
# all seeds (1..30) and compute mean±sd envelope for each strategy.
SEED: int | None = None  # test aggregate across seeds
# when SEED is None you can still disable the shaded standard-deviation band
SHOW_SHADE = False

TIME_COL = "Time"
VALUE_COL = "Failed events"

# when aggregating across seeds we iterate this range
SEEDS = range(1, 31)

# ---------- helpers ----------

def load_hourly(prefix: str, seed: int) -> pd.DataFrame | None:
    """Load an hourly-processed CSV for given strategy prefix and seed."""
    filename = DATA_DIR / f"{prefix}_{seed}_hourly.csv"
    if not filename.exists():
        print(f"warning: hourly file {filename.name} not found")
        return None
    try:
        # hourly files written by preprocess use default comma delimiter
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"unable to read {filename.name}: {e}")
        return None
    if TIME_COL not in df.columns or VALUE_COL not in df.columns:
        print(f"{filename.name} missing required columns")
        return None
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors='coerce')
    return df


# ---------- main logic ----------

def main():
    data = {}
    if SEED is None:
        # aggregate across seeds
        for strat in STRATEGIES:
            frames = []
            for seed in SEEDS:
                df = load_hourly(strat, seed)
                if df is not None:
                    frames.append(df.set_index(TIME_COL)[VALUE_COL])
            if frames:
                combined = pd.concat(frames, axis=1)
                mean_series = combined.mean(axis=1)
                sd_series = combined.std(axis=1)
                data[strat] = (mean_series, sd_series)
    else:
        for strat in STRATEGIES:
            df = load_hourly(strat, SEED)
            if df is not None:
                data[strat] = df

    if not data:
        raise RuntimeError("no data loaded")

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, strat in enumerate(data):
        color = colors[i % len(colors)]
        if SEED is None:
            mean_series, sd_series = data[strat]
            ax.plot(mean_series.index, mean_series.values, label=strat, color=color)
            if SHOW_SHADE:
                ax.fill_between(mean_series.index,
                                mean_series - sd_series,
                                mean_series + sd_series,
                                color=color, alpha=0.2)
        else:
            df = data[strat]
            ax.plot(df[TIME_COL], df[VALUE_COL], label=strat, color=color)

    label = f"seed {SEED}" if SEED is not None else "average across seeds"
    ax.set_title(f"Failed events over time - {label}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Failed events")

    # use daily ticks with both weekday and hour
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
    ax.tick_params(axis="x", rotation=45)

    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
