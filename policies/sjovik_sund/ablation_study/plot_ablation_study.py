import os
import sys
import re
import argparse
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
 
# Use LaTeX styling when the local machine has LaTeX installed; otherwise keep
# math labels working through Matplotlib's built-in mathtext renderer.
if shutil.which("latex"):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage[T1]{fontenc} \usepackage{mlmodern}"
    })
else:
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
    })
 
# --- PATHING ---
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))
 
# Matches:
#   exp_alpha_0.2
#   exp_alpha_0.2_20260414_134502
#   exp_alpha_0.2_bias_term_fsdiag10_20260414_134502
#   exp_sgd_0.2_20260414_134502
FOLDER_PATTERN = re.compile(r"^(.+)_(?:alpha|adam|sgd)_([0-9]+(?:\.[0-9]+)?)(?:_(.*))?$")
TIMESTAMP_SUFFIX_PATTERN = re.compile(r"(?:^|_)\d{8}_\d{6}(?:\s+copy)?$")
 
 
@dataclass(frozen=True)
class Run:
    exp_name: str
    base_name: str
    alpha: str
    path: Path
 
 
def _alpha_sort_key(alpha: str):
    try:
        return (0, float(alpha))
    except ValueError:
        return (1, alpha)
 
 
def _parse_run_folder(folder_name: str):
    m = FOLDER_PATTERN.match(folder_name)
    if not m:
        return None
 
    base_name, alpha_str, suffix = m.group(1), m.group(2), m.group(3) or ""
    suffix_without_timestamp = TIMESTAMP_SUFFIX_PATTERN.sub("", suffix).strip("_")
    exp_name = base_name
    if suffix_without_timestamp:
        exp_name = f"{base_name}_{suffix_without_timestamp}"
 
    return exp_name, base_name, alpha_str
 
 
def discover_runs(study_dir: Path, filter_alphas=None, filter_experiments=None):
    """
    Scan study_dir for folders matching {exp}_alpha_{alpha}[_{run_options}].
    Returns a list of Run entries so repeated experiment/alpha combinations are not dropped.
    """
    runs = []
    for d in sorted(study_dir.iterdir()):
        if not d.is_dir():
            continue
        parsed = _parse_run_folder(d.name)
        if not parsed:
            continue
        exp_name, base_name, alpha_str = parsed
        if filter_experiments and not {exp_name, base_name, d.name}.intersection(filter_experiments):
            continue
        if filter_alphas and alpha_str not in filter_alphas:
            continue
        runs.append(Run(exp_name=exp_name, base_name=base_name, alpha=alpha_str, path=d))
    return runs
 
 
def load_experiment(exp_dir: Path):
    """
    Load all seed CSVs in exp_dir while ignoring specific metadata/metric columns.
    """
    csv_files = sorted(exp_dir.glob("*_weights_evolution.csv"))
    if not csv_files:
        csv_files = sorted(exp_dir.glob("*.csv"))
    if not csv_files:
        return None
 
    # Define the columns you want to ignore
    cols_to_ignore = [
        "alpha", "alpha_initial", "epsilon", "epsilon_initial", "starvations",
        "long_congestions", "short_congestions", "total_trips", "bike_departures",
        "bike_arrivals", "total_onsite_repairs", "total_depot_pickups",
        "total_depot_deliveries", "total_depot_visits", "total_functional_pickups",
        "total_functional_deliveries", "broken_ratio_start_onsite",
        "broken_ratio_start_depot", "functional_ratio_start",
        "broken_ratio_end_onsite", "broken_ratio_end_depot", "functional_ratio_end",
        "new_breakdowns_onsite", "new_breakdowns_depot", "restored_onsite", "restored_depot"
    ]
 
    loaded_data = []
    found_seeds = []
    min_length = float("inf")
 
    for csv_file in csv_files:
        seed_match = re.search(r"seed(\d+)", csv_file.name)
        found_seeds.append(seed_match.group(1) if seed_match else "?")
 
        df = pd.read_csv(csv_file)
        if "episode" not in df.columns or "service_level" not in df.columns:
            df = pd.read_csv(csv_file, skiprows=[1])
        if "episode" not in df.columns or "service_level" not in df.columns:
            print(f"  Skipping {csv_file.name} — missing episode/service_level columns")
            continue
        
        # Drop the unwanted columns here
        df = df.drop(columns=cols_to_ignore, errors="ignore")
        df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
        df["service_level"] = pd.to_numeric(df["service_level"], errors="coerce")
        df = df.dropna(subset=["episode", "service_level"])
        if df.empty:
            print(f"  Skipping {csv_file.name} — no numeric episode/service_level rows")
            continue
 
        episodes = df["episode"].values
        service_levels = df["service_level"].values
 
        window = min(30, len(episodes))
        sl_smoothed = pd.Series(service_levels).rolling(window=window, min_periods=1).mean().values
        
        # w_vals now only contains the actual weight features
        feature_df = df.drop(columns=["episode", "service_level"])
        feature_df = feature_df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        w_vals = feature_df.values
 
        loaded_data.append((sl_smoothed, w_vals, episodes, list(feature_df.columns)))
        if len(episodes) < min_length:
            min_length = len(episodes)
 
    if not loaded_data:
        return None
 
    all_sls, all_weights, final_episodes = [], [], None
    feature_names = list(dict.fromkeys(feat for *_, features in loaded_data for feat in features))
    for sl_smoothed, w_vals, ep_vals, features in loaded_data:
        all_sls.append(sl_smoothed[:min_length])
        weight_df = pd.DataFrame(w_vals[:min_length, :], columns=features)
        all_weights.append(weight_df.reindex(columns=feature_names).values)
        final_episodes = ep_vals[:min_length]
 
    assert final_episodes is not None
    all_sls = np.array(all_sls)
    all_weights = np.array(all_weights, dtype=float)
 
    return (
        final_episodes,
        all_sls.mean(axis=0),
        all_sls.std(axis=0),
        np.nanmean(all_weights, axis=0),
        feature_names,
        found_seeds,
    )
 
 
def plot_ablation_comparison(filter_alphas=None, filter_experiments=None):
    study_dir = WORKSPACE_ROOT / "models/gamma95test"
    #study_dir = WORKSPACE_ROOT / "models/Convergencemethods/Rewardshaping"
    #study_dir = WORKSPACE_ROOT / "models/Convergencemethods/TDlambda"
    if not study_dir.exists():
        print(f"Error: Could not find ablation study directory at {study_dir}")
        return
 
    runs = discover_runs(study_dir, filter_alphas, filter_experiments)
 
    if not runs:
        print("No matching experiment folders found.")
        return
 
    runs_by_exp = defaultdict(list)
    for run in runs:
        runs_by_exp[run.exp_name].append(run)
 
    all_alphas = sorted({run.alpha for run in runs}, key=_alpha_sort_key)
    print(f"Found experiments : {sorted(runs_by_exp.keys())}")
    print(f"Found alphas      : {all_alphas}\n")
 
    # ── Shared style config ────────────────────────────────────────────────────
    # 20+ distinct colors inspired by the project palette
    COLORS = [
        "#344E41",  # Dark Spruce
        "#D1495B",  # Warm Red
        "#526ECA",  # Blue Accent
        "#EDAE49",  # Ochre Gold
        "#84A579",  # Sage Green
        "#7B3F5E",  # Plum
        "#4A90A4",  # Steel Blue
        "#C67C3E",  # Terracotta
        "#3D7068",  # Teal Green
        "#A44A3F",  # Brick Red
        "#9B6B9B",  # Dusty Purple
        "#2E5C8A",  # Navy Blue
        "#E8935A",  # Burnt Orange
        "#4C7C59",  # Forest Green
        "#F2C94C",  # Warm Yellow
        "#6B9E78",  # Medium Green
        "#5B7FA6",  # Slate Blue
        "#C4956A",  # Sand
        "#B05070",  # Raspberry
        "#3A7D7A",  # Deep Teal
        "#8B4513",  # Saddle Brown
        "#6C757D",  # Cool Grey
    ]
 
    exp_list  = sorted(runs_by_exp.keys())
    color_map = {exp: COLORS[i % len(COLORS)] for i, exp in enumerate(exp_list)}
 
    def get_color(exp, alpha=None):
        return color_map[exp]
 
    # ── 1. Per-experiment plots (one line per alpha) ────────────────────────────
    for exp_name, exp_runs in sorted(runs_by_exp.items()):
        fig, ax = plt.subplots(figsize=(12, 6))
        any_plotted = False
        alpha_counts = defaultdict(int)
        for run in exp_runs:
            alpha_counts[run.alpha] += 1
 
        for run in sorted(exp_runs, key=lambda r: (_alpha_sort_key(r.alpha), r.path.name)):
            alpha_label = run.alpha
            result = load_experiment(run.path)
            if result is None:
                print(f"  Skipping {run.path.name} — no usable CSV files found")
                continue
 
            episodes, sl_mean, sl_std, weights_mean, feature_names, found_seeds = result
            color = get_color(exp_name, run.alpha)
            run_suffix = f"  ({run.path.name})" if alpha_counts[run.alpha] > 1 else ""
 
            ax.plot(
                episodes, sl_mean,
                linestyle="-", linewidth=2.0, alpha=0.7,
                color=color,
                label=rf"$\alpha$={alpha_label}{run_suffix}  (peak SL={sl_mean.max():.4f}, final={sl_mean[-30:].mean():.4f})",
            )
            ax.fill_between(episodes, sl_mean - sl_std, sl_mean + sl_std, color=color, alpha=0.15)
            trend = np.poly1d(np.polyfit(episodes, sl_mean, 1))(episodes)
            ax.plot(episodes, trend, linestyle="--", linewidth=3.0, color=color, alpha=0.3)
            any_plotted = True
 
            # Weight evolution plot: each feature gets its own palette color
 
            # Weight evolution plot: each feature gets its own palette color
            fig_w, ax_w = plt.subplots(figsize=(12, 6))
            weight_colors = [COLORS[i % len(COLORS)] for i in range(len(feature_names))]
            for i, feat in enumerate(feature_names):
                ax_w.plot(episodes, weights_mean[:, i], linewidth=2, alpha=0.85,
                          color=weight_colors[i],
                          label=f"{feat} ({weights_mean[-1, i]:+.3f})")
            ax_w.set_title(
                rf"Weight evolution: {exp_name} ($\alpha$={alpha_label})",
                #f"Seeds: {', '.join(found_seeds)}",
                #f"Seeds: {', '.join(found_seeds)}",
                fontsize=13, fontweight="bold"
            )
            ax_w.set_xlabel("Episode", fontsize=12)
            ax_w.set_ylabel(r"Weight ($\theta$)", fontsize=12)
            ax_w.set_ylabel(r"Weight ($\theta$)", fontsize=12)
            ax_w.grid(True, alpha=0.3)
            ax_w.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=11)
            ax_w.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=11)
            plt.tight_layout()
            fig_w.savefig(run.path / f"{run.path.name}_mean_weights.png", dpi=200, bbox_inches="tight")
            plt.close(fig_w)
 
            print(f"  {run.path.name}  α={run.alpha}  seeds={found_seeds}  "
                  f"peak={sl_mean.max():.4f}  final={sl_mean[-30:].mean():.4f}")
 
        if not any_plotted:
            plt.close(fig)
            continue
 
        ax.set_title(
            f"Convergence: {exp_name}\n"
            f"(Solid = mean over seeds, shaded = ±1 std)",
            fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Training episode", fontsize=12)
        ax.set_ylabel("Service level (30-ep moving avg)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=11)
        ax.legend(loc="lower right", fontsize=11)
        fig.tight_layout()
 
        out = study_dir / f"convergence_{exp_name}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"  -> Saved {out.name}")
 
    # ── 2. Master service level comparison (all experiments × alphas) ───────────
    fig_master, ax_master = plt.subplots(figsize=(14, 7))
    any_master = False
 
    for exp_name, exp_runs in sorted(runs_by_exp.items()):
        alpha_counts = defaultdict(int)
        for run in exp_runs:
            alpha_counts[run.alpha] += 1
 
        for run in sorted(exp_runs, key=lambda r: (_alpha_sort_key(r.alpha), r.path.name)):
            alpha_label = run.alpha
            result = load_experiment(run.path)
            if result is None:
                continue
            episodes, sl_mean, sl_std, *_ = result
            color = get_color(exp_name)
            run_suffix = f"  ({run.path.name})" if alpha_counts[run.alpha] > 1 else ""
            ax_master.plot(
                episodes, sl_mean,
                linestyle="-", linewidth=2.0, alpha=1, color=color,
                label=rf"{exp_name}  $\alpha$={alpha_label}{run_suffix}  (peak={sl_mean.max():.4f})",
            )
            ax_master.fill_between(episodes, sl_mean - sl_std, sl_mean + sl_std, color=color, alpha=0.08)
            trend = np.poly1d(np.polyfit(episodes, sl_mean, 1))(episodes)
            ax_master.plot(episodes, trend, linestyle="--", linewidth=3.0, color=color, alpha=0.7)
            any_master = True
 
    if any_master:
        alpha_label = "_".join(all_alphas) if filter_alphas else "all"
        title_alphas = rf"$\alpha$ ∈ {{{', '.join(all_alphas)}}}" if filter_alphas else "All Alphas"
        ax_master.set_title(
            rf"Ablation study: Service level evolution across experiments",
            fontsize=13, fontweight="bold"
        )
        ax_master.set_xlabel("Training episode", fontsize=12)
        ax_master.set_ylabel(r"Service level ($30$-ep moving avg)", fontsize=12)
        ax_master.grid(True, alpha=0.3)
        ax_master.legend(loc="lower right", fontsize=11, ncol=2)
        ax_master.legend(loc="lower right", fontsize=11, ncol=2)
        fig_master.tight_layout()
        out = study_dir / f"ablation_comparison_alpha_{alpha_label}.png"
        fig_master.savefig(out, dpi=200)
        print(f"\nSaved master plot to: {out.name}")
    plt.close(fig_master)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot VFA ablation results across experiments and alphas")
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=str,
        default=None,
        metavar="A",
        help="Filter to specific alpha values (e.g. --alphas 0.2 0.5). Default: all discovered.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        type=str,
        default=None,
        metavar="NAME",
        help="Filter to specific experiments (e.g. --experiments SR V2_RC). Default: all discovered.",
    )
    args = parser.parse_args()
    plot_ablation_comparison(filter_alphas=args.alphas, filter_experiments=args.experiments)
 
 