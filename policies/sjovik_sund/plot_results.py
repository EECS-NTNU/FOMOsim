"""
plot_results.py  -  Print average service level table from results CSVs.

Usage
-----
    python policies/sjovik_sund/plot_results.py --folder policies/sjovik_sund/simulation_results/csv
    python policies/sjovik_sund/plot_results.py --folder policies/sjovik_sund/simulation_results/Benchmark_336_seed_42-52
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


_INSTANCE_RE = re.compile(r"_(TD_W\d+|OS_W\d+|EH_\w+)")

def short_name(stem: str) -> str:
    stem = stem.replace("_results", "")
    m = _INSTANCE_RE.search(stem)
    if m:
        return stem[: m.start()]
    m2 = re.search(r"_V\d+_D\d+h_", stem)
    if m2:
        return stem[: m2.start()]
    return stem


def load_results(folder: Path) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(folder.glob("*_results.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Warning: could not read {csv_path.name}: {e}")
            continue
        if "Service Level" not in df.columns:
            print(f"Warning: no 'Service Level' column in {csv_path.name}, skipping.")
            continue
        label = short_name(csv_path.stem)
        for sl in df["Service Level"]:
            rows.append({"experiment": label, "service_level": sl})
    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame, folder: Path) -> None:
    summary = (
        df.groupby("experiment")["service_level"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    col_w = max(summary["experiment"].str.len().max(), len("Experiment")) + 2
    print(f"\nService Level Summary — {folder.name}")
    print("=" * (col_w + 36))
    print(f"{'Experiment':<{col_w}} {'Mean':>8}  {'Std':>8}  {'n':>4}")
    print("-" * (col_w + 36))
    for _, row in summary.iterrows():
        std_str = f"{row['std']:.4f}" if not np.isnan(row["std"]) else "   N/A"
        print(f"{row['experiment']:<{col_w}} {row['mean']:>8.4f}  {std_str:>8}  {int(row['n']):>4}")
    print("=" * (col_w + 36))


def main() -> None:
    parser = argparse.ArgumentParser(description="Print service level table from results CSVs.")
    parser.add_argument(
        "--folder",
        type=str,
        default="policies/sjovik_sund/simulation_results/csv",
        help="Folder containing *_results.csv files.",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    df = load_results(folder)
    if df.empty:
        raise SystemExit("No valid results found.")

    print_table(df, folder)


if __name__ == "__main__":
    main()
