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
from scipy import stats

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

# Policy name mapping for display in plots
# Key: internal name from filename, Value: display name in plots
POLICY_DISPLAY_NAMES = {
    "greedy": "GP",
    "greedy_neighbors": "GPNI",
    "pilot": "Kloimullner PILOT",
    "xpilot": "X-PILOT"
}

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
    fig, ax = plt.subplots(figsize=(14, 7))
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
        
        # Get display name for policy
        display_name = POLICY_DISPLAY_NAMES.get(policy, policy)
        
        # Plot mean line
        ax.plot(mean_series.index, mean_series.values, label=display_name, 
                color=color, linewidth=3)
        
        if False:
            # Plot shaded std deviation band
            ax.fill_between(mean_series.index,
                            mean_series - std_series,
                            mean_series + std_series,
                            color=color, alpha=0.2)
    
    # ax.set_title(f"Failed events over time - {city} {week} (average across {NUM_SEEDS} seeds)")
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Average # of accumulated failed events", fontsize=14)
    
    # Use daily ticks with weekday only
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a"))
    ax.tick_params(axis="x", rotation=45, labelsize=12, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
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
        if city == "BG":
            print(f"  Skipping {city}_{week}")
            continue
        
        print(f"  Processing {city}_{week} for X-Pilot...")
        long_congestion_values = []
        starvation_values = []

        for csv_file in DATA_DIR.glob(f"{city}_{week}_xpilot_*_hourly.csv"):
            parsed = parse_filename(csv_file.name)
            if not parsed:
                continue
            _, _, parsed_policy, _ = parsed
            if parsed_policy != "xpilot":
                continue
            
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


def create_congestion_bar_chart_with_pilot(city_weeks: List[Tuple[str, str]]):
    """Create a bar chart comparing X-Pilot and Kloimullner PILOT percentages."""
    print("\nCreating combined X-Pilot and Kloimullner PILOT bar chart...")
    
    bar_data = []
    labels = []
    
    for city, week in city_weeks:
        if city == "BG":
            print(f"  Skipping {city}_{week}")
            continue
        
        print(f"  Processing {city}_{week} for X-Pilot and Kloimullner PILOT...")
        row = {
            "xpilot_long": None,
            "xpilot_starvation": None,
            "pilot_long": None,
            "pilot_starvation": None,
        }
        
        for policy_key in ["xpilot", "pilot"]:
            long_values = []
            starvation_values = []
            
            for csv_file in DATA_DIR.glob(f"{city}_{week}_{policy_key}_*_hourly.csv"):
                parsed = parse_filename(csv_file.name)
                if not parsed:
                    continue
                _, _, parsed_policy, _ = parsed
                if parsed_policy != policy_key:
                    continue
                
                df = load_csv(csv_file)
                if df is None:
                    continue
                
                monday_data = df[df[TIME_COL].dt.dayofweek == 0]  # Monday = 0
                if len(monday_data) > 0:
                    last_monday_row = monday_data.iloc[-1]
                    long_values.append(last_monday_row.get('long_congestion', 0))
                    starvation_values.append(last_monday_row.get('starvation', 0))
            
            if long_values and starvation_values:
                avg_long = sum(long_values) / len(long_values)
                avg_starve = sum(starvation_values) / len(starvation_values)
                total = avg_long + avg_starve
                if total > 0:
                    row[f"{policy_key}_long"] = (avg_long / total) * 100
                    row[f"{policy_key}_starvation"] = (avg_starve / total) * 100
        
        if all(value is not None for value in row.values()):
            bar_data.append(row)
            labels.append(f"{city}_{week}")
        else:
            print(f"    Skipping {city}_{week} because not all policies had complete data")
    
    if not bar_data:
        print("  No data found for combined bar chart")
        return
    
    sorted_indices = sorted(range(len(labels)), key=lambda i: (0 if labels[i].startswith("OS") else 1))
    bar_data = [bar_data[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    x = list(range(len(labels)))
    width = 0.16
    
    positions = {
        "xpilot_long": [i - 1.5 * width for i in x],
        "xpilot_starvation": [i - 0.5 * width for i in x],
        "pilot_long": [i + 0.5 * width for i in x],
        "pilot_starvation": [i + 1.5 * width for i in x],
    }
    
    colors = {"long": "skyblue", "starvation": "lightcoral"}
    hatches = {"xpilot": "", "pilot": "//"}
    labels_map = {
        "xpilot_long": "X-PILOT long roaming",
        "xpilot_starvation": "X-PILOT starvation",
        "pilot_long": "Kloimullner PILOT long roaming",
        "pilot_starvation": "Kloimullner PILOT starvation",
    }
    
    for key in ["xpilot_long", "xpilot_starvation", "pilot_long", "pilot_starvation"]:
        policy_key, metric = key.split("_")
        values = [row[key] for row in bar_data]
        bar = ax.bar(positions[key], values, width,
                     label=labels_map[key],
                     color=colors["long"] if metric == "long" else colors["starvation"],
                     hatch=hatches[policy_key], edgecolor='black', alpha=0.8)
        for rect, value in zip(bar, values):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.5,
                    f"{value:.1f}", ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Average percentage of failed events (%)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(fontsize=11, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_file = DATA_DIR / "xpilot_and_pilot_congestion_starvation_comparison.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved bar chart to {output_file.name}")
    plt.close(fig)


def perform_policy_comparison_tests(city_weeks: List[Tuple[str, str]]):
    """Perform paired t-tests comparing policies for Oslo and New York."""
    print("\n" + "="*80)
    print("POLICY COMPARISON - PAIRED T-TESTS")
    print("="*80)
    
    # Filter to OS and NY only
    filtered_cities = [(c, w) for c, w in city_weeks if c in ["OS", "NY"]]
    
    for city, week in filtered_cities:
        print(f"\n{city}_{week} Analysis:")
        print("-" * 80)
        
        # Load failed events at last Monday for each policy and seed
        # Store as dict: policy -> {seed: value}
        policy_values = {}  # policy -> {seed_num: value}
        
        for policy in POLICY_DISPLAY_NAMES.keys():
            seed_values = {}
            for csv_file in DATA_DIR.glob(f"{city}_{week}_{policy}_*_hourly.csv"):
                parsed = parse_filename(csv_file.name)
                if not parsed:
                    continue
                _, _, parsed_policy, seed = parsed
                
                # IMPORTANT: Verify parsed policy matches expected policy
                # Prevents greedy_neighbors files when looking for greedy
                if parsed_policy != policy:
                    continue
                
                df = load_csv(csv_file)
                if df is None:
                    continue
                
                # Get last Monday value
                monday_data = df[df[TIME_COL].dt.dayofweek == 0]  # Monday = 0
                if len(monday_data) > 0:
                    last_value = monday_data.iloc[-1][VALUE_COL]
                    seed_values[seed] = last_value
            
            if seed_values:
                policy_values[policy] = seed_values
        
        if len(policy_values) < 2:
            print("  Not enough policies with data")
            continue
                
        # Find common seeds across all policies
        all_seeds = set.intersection(*[set(values.keys()) for values in policy_values.values()])
        if len(all_seeds) < 2:
            print(f"  Insufficient common seeds for paired test (found {len(all_seeds)})")
            continue
        
        # Prepare summary stats using all available seeds
        print("\nSummary Statistics:")
        print(f"{'Policy':<25} {'Mean':>10} {'Std':>10} {'N':>5}")
        print("-" * 55)
        
        policy_list = sorted(policy_values.keys())
        policy_stats = {}
        
        for policy in policy_list:
            values = list(policy_values[policy].values())
            mean_val = sum(values) / len(values)
            std_val = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
            policy_stats[policy] = {"mean": mean_val, "std": std_val, "n": len(values)}
            
            display_name = POLICY_DISPLAY_NAMES.get(policy, policy)
            print(f"{display_name:<25} {mean_val:>10.1f} {std_val:>10.1f} {len(values):>5}")
        
        # Perform paired t-tests using only common seeds
        print(f"\nPaired T-Tests (using {len(all_seeds)} common seeds):")
        print("-" * 80)
        
        xpilot_values = policy_values.get("xpilot")
        if xpilot_values is None:
            print("  X-PILOT not found")
            continue
        
        print(f"{'Comparison':<30} {'t-statistic':>12} {'p-value':>12} {'Better':>20}")
        print("-" * 75)
        
        for policy in policy_list:
            if policy == "xpilot":
                continue
            
            # Create paired arrays using only common seeds
            xpilot_paired = [xpilot_values[s] for s in sorted(all_seeds)]
            other_paired = [policy_values[policy][s] for s in sorted(all_seeds)]
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(xpilot_paired, other_paired)
            
            # Determine which is better (lower is better)
            xpilot_mean = sum(xpilot_paired) / len(xpilot_paired)
            other_mean = sum(other_paired) / len(other_paired)
            
            if other_mean < xpilot_mean:
                better = f"{POLICY_DISPLAY_NAMES[policy]} BETTER"
            else:
                better = "X-PILOT BETTER"
            
            comp_name = f"{POLICY_DISPLAY_NAMES[policy]} vs X-PILOT"
            
            if p_value < 0.05:
                print(f"{comp_name:<30} {t_stat:>12.4f} {p_value:>12.4f} {better:>20} *")
            else:
                print(f"{comp_name:<30} {t_stat:>12.4f} {p_value:>12.4f} {'(no sig)':>20}")
        
        print("\n* p < 0.05 (significant difference)")
        print()



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
