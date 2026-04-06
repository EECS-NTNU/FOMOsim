import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_averaged_weights(alpha: str, files: list[Path], target_dir: Path):
    """Generates and saves an averaged weight convergence plot for a specific alpha."""
    all_weights = []
    episodes = None
    feature_names = None
    
    for csv_path in files:
        df = pd.read_csv(csv_path)
        if 'episode' not in df.columns:
            continue
            
        if episodes is None:
            episodes = df['episode'].values
        if feature_names is None:
            feature_names = df.drop(columns=['episode', 'service_level']).columns
            
        # Extract just the weights
        weights = df.drop(columns=['episode', 'service_level']).values
        all_weights.append(weights)

    if not all_weights:
        return

    # Calculate the mean across all seeds
    # Shape becomes: [num_episodes, num_features]
    all_weights = np.array(all_weights)
    mean_weights = np.mean(all_weights, axis=0)

    plt.figure(figsize=(14, 8))
    
    # Plot the averaged lines
    for i, column in enumerate(feature_names):
        plt.plot(episodes, mean_weights[:, i], linewidth=2, label=column)

    plt.title(f"Averaged VFA Weight (\u03B8) Convergence (\u03B1 = {alpha}) - {len(files)} Seeds", fontsize=16, fontweight='bold')
    plt.xlabel("Training Episode", fontsize=14)
    plt.ylabel("Mean Weight Value", fontsize=14)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = target_dir / f"vfa_mean_weights_alpha{alpha}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  -> Saved Averaged Weight Plot: {save_path.name}")
    plt.close()

def plot_averaged_alpha_comparison(grouped_runs: dict, target_dir: Path):
    """Generates a master plot comparing the averaged Service Levels across all alphas."""
    plt.figure(figsize=(12, 7))
    plotted_lines = 0

    # Sort alphas numerically so the legend makes sense (e.g., 0.001 -> 0.005 -> 0.01)
    sorted_alphas = sorted(grouped_runs.keys(), key=float)

    for alpha in sorted_alphas:
        files = grouped_runs[alpha]
        all_sls = []
        episodes = None
        
        for csv_path in files:
            df = pd.read_csv(csv_path)
            if 'episode' not in df.columns or 'service_level' not in df.columns:
                continue

            if episodes is None:
                episodes = df['episode'].values
            
            # Smooth the individual seed's RL service level
            window_size = min(10, len(episodes))
            sl_smoothed = df['service_level'].rolling(window=window_size, min_periods=1).mean().values
            all_sls.append(sl_smoothed)

        if not all_sls:
            continue
            
        all_sls = np.array(all_sls)
        mean_sl = np.mean(all_sls, axis=0)
        std_sl = np.std(all_sls, axis=0)
        
        # Plot mean line and shaded standard deviation
        line = plt.plot(episodes, mean_sl, linewidth=2.5, label=f"α = {alpha}")
        color = line[0].get_color()
        plt.fill_between(episodes, mean_sl - std_sl, mean_sl + std_sl, color=color, alpha=0.15)
        
        plotted_lines += 1

    if plotted_lines > 0:
        plt.title("Learning Rate (\u03B1) Robust Comparison (Mean \u00B1 1 Std Dev)", fontsize=16, fontweight='bold')
        plt.xlabel("Training Episode", fontsize=14)
        plt.ylabel(f"Service Level ({window_size}-ep Moving Avg)", fontsize=14)
        
        plt.legend(loc="lower right", title="Learning Rates", fontsize=12, title_fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        save_path = target_dir / "alpha_comparison_service_level_robust.png"
        plt.savefig(save_path, dpi=300)
        print(f"\n🚀 Success! Master Robust Alpha Comparison Plot saved to: {save_path.name}")
    else:
        print("\n⚠️ No valid runs found to compare.")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Batch plot robust, seed-averaged VFA results.")
    parser.add_argument(
        "--dir", 
        type=str, 
        default="policies/sjovik_sund/vfa/models", 
        help="Path to the directory containing the CSV files."
    )
    args = parser.parse_args()
    
    target_dir = Path(args.dir)
    if not target_dir.exists():
        print(f"Error: Directory {target_dir} does not exist.")
        return

    csv_files = list(target_dir.glob("*_weights_evolution.csv"))
    if not csv_files:
        print(f"No '*_weights_evolution.csv' files found in {target_dir}")
        return

    # 1. Group files by Alpha using regex
    # Matches patterns like "alpha0.1_seed" or "alpha0.05_seed"
    grouped_runs = {}
    for csv_file in csv_files:
        match = re.search(r"alpha([0-9\.]+)_seed", csv_file.name)
        if match:
            alpha_val = match.group(1)
            if alpha_val not in grouped_runs:
                grouped_runs[alpha_val] = []
            grouped_runs[alpha_val].append(csv_file)
        else:
            print(f"  [Warning] Could not parse alpha from filename: {csv_file.name}. Skipping.")

    print(f"Found {len(csv_files)} files across {len(grouped_runs)} distinct alpha values.")
    print("-" * 60)
    
    # 2. Plot averaged weights for each alpha group
    for alpha, files in grouped_runs.items():
        print(f"Processing \u03B1 = {alpha} ({len(files)} seeds)...")
        plot_averaged_weights(alpha, files, target_dir)
        
    print("-" * 60)
    
    # 3. Plot the master Service Level comparison
    plot_averaged_alpha_comparison(grouped_runs, target_dir)

if __name__ == "__main__":
    main()