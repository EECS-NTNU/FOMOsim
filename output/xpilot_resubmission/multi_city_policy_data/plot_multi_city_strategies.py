"""Generate strategy comparison plots for each city-week combination.

This script reads CSV files from the multi_city_policy_data folder and creates
plots comparing different policies for each city-week combination (BG_W35, NY_W31, OS_W31).

File naming convention:
    cityname_weeknumber_policy_0.13e_2.25k_numvehicles_seednumber.csv
    
For each city-week combination, the script:
1. Groups files by policy (greedy, greedy_neighbors, pilot, xpilot)
2. Aggregates data across all 30 seeds per policy
3. Creates a plot showing mean development with shaded std deviation band
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import re
import sys
from typing import Dict, Tuple, List

# ---------- configuration ----------
# Use the directory where this script is located
DATA_DIR = Path(__file__).resolve().parent

TIME_COL = "Time"
VALUE_COL = "Failed events"
SEEDS = range(1, 31)
NUM_SEEDS = 30

# ---------- helpers ----------

def parse_filename(filename: str) -> Tuple[str, str, str, int] | None:
    """Parse filename to extract city, week, policy, and seed.
    
    Expected format: cityname_weeknumber_policy_0.13e_2.25k_numvehicles_seednumber_hourly.csv
    Example: BG_W35_greedy_0.13e_2.25k_1v_10s_hourly.csv
    
    Returns:
        Tuple of (city, week, policy, seed) or None if parsing fails
    """
    # Pattern: CityWeek_policy_..._Vs_seed_hourly.csv
    # The middle irrelevant part can contain underscores, so we use .*? to skip it
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
    """Load all seeds for a given city-week combination and aggregate by policy.
    
    Returns:
        Dictionary mapping policy name to (mean_series, std_series) tuples,
        or None if no data found.
    """
    # Group files by policy
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
        
        # Extract the value column indexed by time
        policy_data[policy].append(df.set_index(TIME_COL)[VALUE_COL])
    
    if not policy_data:
        return None
    
    # Compute mean and std for each policy
    aggregated = {}
    for policy, series_list in policy_data.items():
        if series_list:
            combined = pd.concat(series_list, axis=1)
            mean_series = combined.mean(axis=1)
            std_series = combined.std(axis=1)
            aggregated[policy] = (mean_series, std_series)
    
    return aggregated if aggregated else None


def plot_city_week(city: str, week: str, data: Dict[str, Tuple[pd.Series, pd.Series]]):
    """Create a plot for a city-week combination."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Sort policies for consistent coloring
    policies = sorted(data.keys())
    
    for i, policy in enumerate(policies):
        color = colors[i % len(colors)]
        mean_series, std_series = data[policy]
        
        # Limit to Tuesday in next week - find the last Tuesday in the data
        tuesday_data = mean_series[mean_series.index.dayofweek == 0]  # Monday = 0Tuesday = 1
        if len(tuesday_data) > 0:
            last_tuesday = tuesday_data.index[-1]
            # Include data up to and including the last Tuesday
            mask = mean_series.index <= last_tuesday
            mean_series = mean_series[mask]
            std_series = std_series[mask]
        
        # Plot mean line
        ax.plot(mean_series.index, mean_series.values, label=policy, 
                color=color, linewidth=2)
        
        if False:
            # Plot shaded std deviation band
            ax.fill_between(mean_series.index,
                            mean_series - std_series,
                            mean_series + std_series,
                            color=color, alpha=0.2)
    
    ax.set_title(f"Failed events over time - {city} {week} (average across {NUM_SEEDS} seeds)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Failed events")
    
    # Use daily ticks with weekday only
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a"))
    ax.tick_params(axis="x", rotation=45)
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    
    return fig


def create_congestion_bar_chart(city_weeks: List[Tuple[str, str]]):
    """Create a bar chart showing long_congestion vs starvation percentages for X-Pilot strategy."""
    print("\nCreating congestion vs starvation bar chart...")
    
    # Collect data for each city_week
    bar_data = []
    labels = []
    
    for city, week in city_weeks:
        print(f"  Processing {city}_{week} for X-Pilot...")
        
        # Load all X-Pilot files for this city_week
        long_congestion_values = []
        starvation_values = []
        
        for csv_file in DATA_DIR.glob(f"{city}_{week}_xpilot_*_hourly.csv"):
            df = load_csv(csv_file)
            if df is None:
                continue
            
            # Find the last Tuesday in the data
            tuesday_data = df[df[TIME_COL].dt.dayofweek == 1]  # Tuesday = 1
            if len(tuesday_data) > 0:
                last_tuesday_row = tuesday_data.iloc[-1]
                long_congestion_values.append(last_tuesday_row['long_congestion'])
                starvation_values.append(last_tuesday_row['starvation'])
        
        if long_congestion_values and starvation_values:
            # Calculate averages
            avg_long_congestion = sum(long_congestion_values) / len(long_congestion_values)
            avg_starvation = sum(starvation_values) / len(starvation_values)
            total = avg_long_congestion + avg_starvation
            
            if total > 0:
                # Convert to percentages
                long_congestion_pct = (avg_long_congestion / total) * 100
                starvation_pct = (avg_starvation / total) * 100
                
                bar_data.append([long_congestion_pct, starvation_pct])
                labels.append(f"{city}_{week}")
    
    if not bar_data:
        print("  No data found for bar chart")
        return
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(labels))
    width = 0.35
    
    # Plot bars
    long_bars = ax.bar([i - width/2 for i in x], [d[0] for d in bar_data], 
                       width, label='Long roaming for locks', color='skyblue')
    starvation_bars = ax.bar([i + width/2 for i in x], [d[1] for d in bar_data], 
                            width, label='Starvation', color='lightcoral')
    
    # Add value labels on bars
    for bars in [long_bars, starvation_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                   f"{height:.1f}", ha='center', va='bottom', fontsize=9)
    
    # ax.set_xlabel('City_Week')
    ax.set_ylabel('Percentage of failed events (%)')
    # ax.set_title('X-Pilot: Long Roaming for Locks vs Starvation Distribution\n(Values at end of Tuesday in next week)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save the bar chart
    output_file = DATA_DIR / "xpilot_congestion_starvation_comparison.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved bar chart to {output_file.name}")
    plt.close(fig)


# ---------- main ----------

def main():
    """Load data and create plots for each city-week combination."""
    print(f"Looking for CSV files in: {DATA_DIR}")
    print(f"Directory exists: {DATA_DIR.exists()}")
    
    # Count all CSV files
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
                # Save figure
                output_file = DATA_DIR / f"{city}_{week}_strategy_comparison.png"
                fig.savefig(output_file, dpi=150, bbox_inches='tight')
                print(f"    ✓ Saved to {output_file.name}")
                plt.close(fig)  # Close to free memory
            else:
                print(f"    No data found")
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nDone! All plots saved.")
    
    # Create the additional bar chart for X-Pilot congestion vs starvation
    create_congestion_bar_chart(city_weeks)


if __name__ == "__main__":
    main()
