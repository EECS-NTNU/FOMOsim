"""Generate strategy comparison plots for each city-week combination.

This script reads CSV files from the multi_city_policy_data folder and creates
plots comparing different policies for each city-week combination.

The plot code and generated PNGs are stored in a separate plotting folder
rather than alongside the raw CSV data.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from datetime import datetime
import re
import sys
from typing import Dict, Tuple, List

# ---------- configuration ----------
# The data folder is the parent of this plotting folder.
DATA_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_COL = "Time"
VALUE_COL = "Failed events"
SEEDS = range(1, 31)
NUM_SEEDS = 30

mpl.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 15,
    "legend.title_fontsize": 12,
})

# ---------- helpers ----------

def parse_filename(filename: str) -> Tuple[str, str, str, int] | None:
    """Parse filename to extract city, week, policy, and seed."""
    pattern = r"^([A-Z]{2})_(W\d+)_([a-z_]+)_.*?_\dv_(\d+)s_hourly\.csv$"
    match = re.match(pattern, filename)
    if match:
        city = match.group(1)
        week = match.group(2)
        policy = match.group(3)
        seed = int(match.group(4))
        return city, week, policy, seed
    return None


def load_csv(filepath: Path) -> pd.DataFrame | None:
    """Load a CSV file and convert Time column to datetime."""
    try:
        df = pd.read_csv(filepath)
        if TIME_COL not in df.columns or VALUE_COL not in df.columns:
            return None
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors='coerce')
        return df
    except Exception as e:
        print(f"Error loading {filepath.name}: {e}")
        return None


def get_city_week_keys() -> List[Tuple[str, str]]:
    """Extract all unique city-week combinations from filenames."""
    city_weeks = set()
    for csv_file in DATA_DIR.glob("*.csv"):
        parsed = parse_filename(csv_file.name)
        if parsed:
            city, week, policy, seed = parsed
            city_weeks.add((city, week))
    return sorted(list(city_weeks))


def load_data_for_city_week(city: str, week: str) -> Dict[str, Tuple[pd.Series, pd.Series]] | None:
    """Load all seeds for a given city-week combination and aggregate by policy."""
    policy_data = {}

    for csv_file in DATA_DIR.glob(f"{city}_{week}_*_hourly.csv"):
        parsed = parse_filename(csv_file.name)
        if not parsed:
            continue

        _, _, policy, seed = parsed
        df = load_csv(csv_file)
        if df is None:
            continue

        if policy not in policy_data:
            policy_data[policy] = []

        policy_data[policy].append(df.set_index(TIME_COL)[VALUE_COL])

    if not policy_data:
        return None

    aggregated = {}
    for policy, series_list in policy_data.items():
        if series_list:
            combined = pd.concat(series_list, axis=1)
            mean_series = combined.mean(axis=1)
            std_series = combined.std(axis=1)
            aggregated[policy] = (mean_series, std_series)

    return aggregated if aggregated else None


POLICY_LABELS = {
    "greedy": "GP",
    "greedy_neighbors": "GPNI",
    "pilot": "Kloimullner PILOT",
    "xpilot": "X-PILOT",
}


def plot_city_week(city: str, week: str, data: Dict[str, Tuple[pd.Series, pd.Series]]):
    """Create a plot for a city-week combination."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    policies = sorted(data.keys())

    for i, policy in enumerate(policies):
        color = colors[i % len(colors)]
        mean_series, std_series = data[policy]

        tuesday_data = mean_series[mean_series.index.dayofweek == 0]
        if len(tuesday_data) > 0:
            last_tuesday = tuesday_data.index[-1]
            mask = mean_series.index <= last_tuesday
            mean_series = mean_series[mask]
            std_series = std_series[mask]

        label = POLICY_LABELS.get(policy, policy)
        ax.plot(mean_series.index, mean_series.values, label=label, color=color, linewidth=2)

    ax.set_xlabel("Time", fontsize=13)
    ax.set_ylabel("Average # of accumulated failed events", fontsize=13)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a"))
    ax.tick_params(axis="x", rotation=45, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    fig.autofmt_xdate()
    plt.tight_layout()

    return fig


def create_congestion_bar_chart(city_weeks: List[Tuple[str, str]]):
    """Create a bar chart showing long_congestion vs starvation percentages for X-Pilot."""
    print("\nCreating congestion vs starvation bar chart...")
    bar_data = []
    labels = []

    for city, week in city_weeks:
        print(f"  Processing {city}_{week} for X-Pilot...")
        long_congestion_values = []
        starvation_values = []

        for csv_file in DATA_DIR.glob(f"{city}_{week}_xpilot_*_hourly.csv"):
            df = load_csv(csv_file)
            if df is None:
                continue

            tuesday_data = df[df[TIME_COL].dt.dayofweek == 1]
            if len(tuesday_data) > 0:
                last_tuesday_row = tuesday_data.iloc[-1]
                long_congestion_values.append(last_tuesday_row['long_congestion'])
                starvation_values.append(last_tuesday_row['starvation'])

        if long_congestion_values and starvation_values:
            avg_long_congestion = sum(long_congestion_values) / len(long_congestion_values)
            avg_starvation = sum(starvation_values) / len(starvation_values)
            total = avg_long_congestion + avg_starvation
            if total > 0:
                long_congestion_pct = (avg_long_congestion / total) * 100
                starvation_pct = (avg_starvation / total) * 100
                bar_data.append([long_congestion_pct, starvation_pct])
                labels.append(f"{city}_{week}")

    if not bar_data:
        print("  No data found for bar chart")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(labels))
    width = 0.35
    long_bars = ax.bar([i - width/2 for i in x], [d[0] for d in bar_data], width, label='Long roaming for locks', color='skyblue')
    starvation_bars = ax.bar([i + width/2 for i in x], [d[1] for d in bar_data], width, label='Starvation', color='lightcoral')

    for bars in [long_bars, starvation_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5, f"{height:.1f}", ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Percentage of failed events (%)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    output_file = OUTPUT_DIR / "xpilot_congestion_starvation_comparison.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved bar chart to {output_file.name}")
    plt.close(fig)


def main():
    print(f"Looking for CSV files in: {DATA_DIR}")
    print(f"Directory exists: {DATA_DIR.exists()}")
    all_csvs = list(DATA_DIR.glob("*.csv"))
    print(f"Total CSV files found: {len(all_csvs)}")
    if all_csvs:
        print("Sample files:")
        for f in all_csvs[:5]:
            print(f"  {f.name}")

    city_weeks = get_city_week_keys()
    if not city_weeks:
        print("\nNo data files matching the pattern found.")
        if all_csvs:
            print("Checking if filenames match the expected pattern...")
            for f in all_csvs[:3]:
                parsed = parse_filename(f.name)
                print(f"  {f.name} -> {parsed}")
        return

    print(f"Found {len(city_weeks)} city-week combinations:")
    for city, week in city_weeks:
        print(f"  {city}_{week}")

    print("\nGenerating plots...")
    for city, week in city_weeks:
        print(f"\n  Processing {city}_{week}...")
        try:
            data = load_data_for_city_week(city, week)
            if data:
                print(f"    Loaded {len(data)} policies")
                fig = plot_city_week(city, week, data)
                output_file = OUTPUT_DIR / f"{city}_{week}_strategy_comparison.png"
                fig.savefig(output_file, dpi=150, bbox_inches='tight')
                print(f"    ✓ Saved to {output_file.name}")
                plt.close(fig)
            else:
                print(f"    No data found")
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone! All plots saved.")
    create_congestion_bar_chart(city_weeks)


if __name__ == "__main__":
    main()
