import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from scipy.ndimage import gaussian_filter

# Directory containing the CSV files
DATA_DIR = Path(r"C:/Users/steffejb/OneDrive - NTNU/Work/GitHub/FOMO-sim/fomo/output/xpilot_resubmission/ek_sensitivity/ek_sensitivity")

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
    
    # Data storage
    data = []
    
    for file in files:
        epsilon, kappa = parse_filename(Path(file).name)
        last_row = load_last_row(file)
        failed = last_row[FAILED_COL]
        long_cong = last_row[LONG_CONG_COL]
        starvation = last_row[STARVATION_COL]
        data.append({
            'epsilon': epsilon,
            'kappa': kappa,
            'failed_events': failed,
            'long_congestion': long_cong,
            'starvation': starvation
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Pivot for heatmaps
    failed_pivot = df.pivot(index='epsilon', columns='kappa', values='failed_events')
    long_cong_pivot = df.pivot(index='epsilon', columns='kappa', values='long_congestion')
    starvation_pivot = df.pivot(index='epsilon', columns='kappa', values='starvation')
    
    # Plot heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    sns.heatmap(failed_pivot, ax=axes[0], cmap='viridis', annot=True, fmt='.0f')
    axes[0].set_title('Failed Events')
    axes[0].set_xlabel('Kappa')
    axes[0].set_ylabel('Epsilon')
    
    sns.heatmap(long_cong_pivot, ax=axes[1], cmap='viridis', annot=True, fmt='.0f')
    axes[1].set_title('Long Congestion')
    axes[1].set_xlabel('Kappa')
    axes[1].set_ylabel('Epsilon')
    
    sns.heatmap(starvation_pivot, ax=axes[2], cmap='viridis', annot=True, fmt='.0f')
    axes[2].set_title('Starvation')
    axes[2].set_xlabel('Kappa')
    axes[2].set_ylabel('Epsilon')
    
    plt.tight_layout()
    plt.savefig(DATA_DIR / 'ek_sensitivity_heatmaps.png')
    plt.show()

if __name__ == "__main__":
    main()