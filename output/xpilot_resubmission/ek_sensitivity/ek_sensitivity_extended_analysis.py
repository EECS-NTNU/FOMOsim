import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import glob

# Directory with extended data (files include _2v_ and seed like 14s)
DATA_DIR = Path(__file__).resolve().parent / "data_extended"
OUTPUT_DIR = Path(__file__).resolve().parent
print(f"Using DATA_DIR={DATA_DIR}")

# Columns expected in the CSVs
FAILED_COL = "Failed events"
LONG_CONG_COL = "long_congestion"
STARVATION_COL = "starvation"


def parse_eps_kappa(filename: str):
    """Extract epsilon and kappa from filename using simple regex search."""
    # find first occurrence of number before 'e' and before 'k'
    me = re.search(r"([0-9]+(?:\.[0-9]+)?)e", filename)
    mk = re.search(r"([0-9]+(?:\.[0-9]+)?)k", filename)
    if not me or not mk:
        raise ValueError("Could not parse epsilon/kappa from filename")
    epsilon = float(me.group(1))
    kappa = float(mk.group(1))
    return epsilon, kappa


def load_last_row(path: Path):
    # try common separators: comma first, then semicolon
    try:
        return pd.read_csv(path).iloc[-1]
    except Exception:
        try:
            return pd.read_csv(path, sep=';').iloc[-1]
        except Exception:
            # last resort: read as raw and try to split
            txt = path.read_text()
            # fallback: construct a DataFrame from CSV with pandas default
            return pd.read_csv(path, engine='python').iloc[-1]


def main():
    files = list(DATA_DIR.glob("*.csv"))
    if not files:
        print(f"No CSV files found in {DATA_DIR}")
        return

    records = []
    for f in files:
        name = f.name
        # require the _2v_ pattern (as user indicated)
        if "_2v_" not in name:
            continue
        try:
            eps, kap = parse_eps_kappa(name)
        except Exception as e:
            print(f"Skipping {name}: {e}")
            continue

        try:
            last = load_last_row(f)
        except Exception as e:
            print(f"Skipping {name} (read error): {e}")
            continue

        # guard columns
        if FAILED_COL not in last or LONG_CONG_COL not in last or STARVATION_COL not in last:
            print(f"Skipping {name}: missing expected columns")
            continue

        records.append({
            'epsilon': eps,
            'kappa': kap,
            'failed_events': float(last[FAILED_COL]),
            'long_congestion': float(last[LONG_CONG_COL]),
            'starvation': float(last[STARVATION_COL])
        })

    if not records:
        print("No valid records collected; aborting")
        return

    df = pd.DataFrame(records)

    # Average across seeds for each (epsilon, kappa)
    df_avg = df.groupby(['epsilon', 'kappa'], as_index=False).mean()

    # pivot to matrices for heatmaps
    failed_pivot = df_avg.pivot(index='epsilon', columns='kappa', values='failed_events')
    long_pivot = df_avg.pivot(index='epsilon', columns='kappa', values='long_congestion')
    starv_pivot = df_avg.pivot(index='epsilon', columns='kappa', values='starvation')

    heatmaps = [
        (failed_pivot, 'Failed Events (avg across seeds)', OUTPUT_DIR / 'ek_sensitivity_extended_failed_events.pdf'),
        (long_pivot, 'Long Congestion (avg across seeds)', OUTPUT_DIR / 'ek_sensitivity_extended_long_congestion.pdf'),
        (starv_pivot, 'Starvation (avg across seeds)', OUTPUT_DIR / 'ek_sensitivity_extended_starvation.pdf'),
    ]

    for pivot, title, outpath in heatmaps:
        if pivot is None or pivot.empty:
            print(f"Skipping {title}: empty pivot")
            continue

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(pivot, ax=ax, cmap='viridis', annot=True, fmt='.0f', annot_kws={'fontsize':7})
        ax.set_title(title)
        ax.set_xlabel('Kappa')
        ax.set_ylabel('Epsilon')
        plt.tight_layout()
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {outpath}")


if __name__ == '__main__':
    main()
