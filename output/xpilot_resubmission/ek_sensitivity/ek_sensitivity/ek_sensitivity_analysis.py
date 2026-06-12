import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from scipy.ndimage import gaussian_filter

# Directory containing the CSV files (use repo-relative path)
DATA_DIR = Path(__file__).resolve().parent
print(f"Using DATA_DIR={DATA_DIR}")

# Columns to extract
FAILED_COL = "Failed events"
LONG_CONG_COL = "long_congestion"
STARVATION_COL = "starvation"

def parse_filename(filename):
    """Extract epsilon and kappa from filename like xpilot_0.00e_0.00k.csv"""
    parts = filename.split('_')
    epsilon = float(parts[1].replace('e', ''))
    kappa = float(parts[2].replace('k.csv', ''))
    return epsilon, kappa

def load_last_row(file_path):
    """Load the last row of the CSV file."""
    df = pd.read_csv(file_path, sep=';')
    return df.iloc[-1]

def main():
    # Find all xpilot CSV files
    files = glob.glob(str(DATA_DIR / "xpilot_*.csv"))
    if not files:
        print(f"No files found in {DATA_DIR} matching pattern 'xpilot_*.csv'")
        return
    
    # Data storage
    data = []
    
    for file in files:
        try:
            epsilon, kappa = parse_filename(Path(file).name)
        except Exception as e:
            print(f"Skipping file with unexpected name '{file}': {e}")
            continue

        try:
            last_row = load_last_row(file)
        except Exception as e:
            print(f"Skipping file due to read error '{file}': {e}")
            continue

        # Guard against missing columns in the CSV
        try:
            failed = last_row[FAILED_COL]
            long_cong = last_row[LONG_CONG_COL]
            starvation = last_row[STARVATION_COL]
        except Exception as e:
            print(f"Skipping file due to missing columns in '{file}': {e}")
            continue

        data.append({
            'epsilon': epsilon,
            'kappa': kappa,
            'failed_events': failed,
            'long_congestion': long_cong,
            'starvation': starvation
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    if df.empty:
        print("No valid data rows collected from files; aborting heatmap generation.")
        return
    
    # Pivot for heatmaps
    failed_pivot = df.pivot(index='epsilon', columns='kappa', values='failed_events')
    long_cong_pivot = df.pivot(index='epsilon', columns='kappa', values='long_congestion')
    starvation_pivot = df.pivot(index='epsilon', columns='kappa', values='starvation')

    # Plot and save separate heatmaps for each metric
    heatmaps = [
        (failed_pivot, 'Failed Events', 'ek_sensitivity_failed_events.pdf'),
        (long_cong_pivot, 'Long Congestion', 'ek_sensitivity_long_congestion.pdf'),
        (starvation_pivot, 'Starvation', 'ek_sensitivity_starvation.pdf'),
    ]

    for pivot, title, filename in heatmaps:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(pivot, ax=ax, cmap='viridis', annot=True, fmt='.0f', annot_kws={'fontsize':8})
        ax.set_title(title)
        ax.set_xlabel('Kappa')
        ax.set_ylabel('Epsilon')
        plt.tight_layout()
        outpath = Path(DATA_DIR) / filename
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved heatmap to {outpath}")
        plt.close(fig)

if __name__ == "__main__":
    main()