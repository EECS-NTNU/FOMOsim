import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

ALPHA_MAP = {
    "20260405_163807": "0.1",
    "20260405_163812": "0.05",  
    "20260405_163901": "0.01",
    "20260405_163736": "0.005",
    "20260405_163919": "0.001",
}

def get_alpha_label(filename: str) -> str:
    """Checks the dictionary to see if we know the alpha for this file."""
    for timestamp, alpha in ALPHA_MAP.items():
        if timestamp in filename:
            return f" (Alpha: {alpha})"
    return ""  # If it's not in the map, just leave it blank

def plot_weight_evolution(csv_path: Path):
    """Generates and saves a convergence plot for a single CSV file."""
    df = pd.read_csv(csv_path)
    
    if 'episode' not in df.columns:
        print(f"  [Skip] {csv_path.name} is missing the 'episode' column.")
        return

    episodes = df['episode']
    plt.figure(figsize=(14, 8))
    
    features_plotted = 0
    for column in df.columns:
        if column not in ['episode', 'service_level']:
            plt.plot(episodes, df[column], linewidth=2, label=column)
            features_plotted += 1

    run_id = csv_path.stem.replace("_weights_evolution", "").replace("vfa_trained_", "")
    alpha_label = get_alpha_label(csv_path.name)

    plt.title(f"VFA Weight (\u03B8) Convergence: {run_id}{alpha_label}", fontsize=16, fontweight='bold')
    plt.xlabel("Training Episode", fontsize=14)
    plt.ylabel("Weight Value", fontsize=14)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = csv_path.with_suffix('.png')
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  -> Saved Weight Plot: {save_path.name}{alpha_label}")
    plt.close() 

def plot_alpha_comparison(csv_files: list[Path], target_dir: Path):
    """Generates a master plot comparing the Service Levels of all mapped Alphas."""
    plt.figure(figsize=(12, 7))
    plotted_lines = 0

    # Sort files so the legend order is consistent
    for csv_path in sorted(csv_files):
        alpha_label = get_alpha_label(csv_path.name)
        
        # Only plot it on the comparison graph if it is in our ALPHA_MAP
        if not alpha_label:
            continue
            
        df = pd.read_csv(csv_path)
        if 'episode' not in df.columns or 'service_level' not in df.columns:
            continue

        episodes = df['episode']
        
        # Apply a rolling mean to smooth the noisy RL service levels
        window_size = min(10, len(episodes))
        sl_smoothed = df['service_level'].rolling(window=window_size, min_periods=1).mean()
        
        # Clean up the label for the legend (e.g., "0.05" instead of "(Alpha: 0.05)")
        clean_alpha_val = alpha_label.replace(" (Alpha: ", "").replace(")", "")
        
        plt.plot(episodes, sl_smoothed, linewidth=2.5, alpha=0.9, label=f"α = {clean_alpha_val}")
        plotted_lines += 1

    if plotted_lines > 0:
        plt.title("Learning Rate (\u03B1) Comparison: Service Level Convergence", fontsize=16, fontweight='bold')
        plt.xlabel("Training Episode", fontsize=14)
        plt.ylabel(f"Service Level ({window_size}-ep Moving Avg)", fontsize=14)
        
        # Add legend and grid
        plt.legend(loc="lower right", title="Learning Rates", fontsize=12, title_fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Save the master plot
        save_path = target_dir / "alpha_comparison_service_level.png"
        plt.savefig(save_path, dpi=300)
        print(f"\n🚀 Success! Master Alpha Comparison Plot saved to: {save_path.name}")
    else:
        print("\n⚠️ No mapped alphas found to compare. Check your ALPHA_MAP!")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Batch plot VFA weight evolution CSVs.")
    parser.add_argument(
        "--dir", 
        type=str, 
        default="policies/sjovik_sund/vfa/models", 
        help="Path to the directory containing the CSV files."
    )
    args = parser.parse_args()
    
    target_dir = Path(args.dir)
    
    if not target_dir.exists():
        if Path("models").exists():
            target_dir = Path("models")
        else:
            print(f"Error: Directory {target_dir} does not exist.")
            return

    csv_files = list(target_dir.glob("*_weights_evolution.csv"))
    
    if not csv_files:
        print(f"No '*_weights_evolution.csv' files found in {target_dir}")
        return

    print(f"Found {len(csv_files)} weight evolution files. Generating plots...")
    print("-" * 50)
    
    # 1. Plot individual weight evolutions
    for csv_file in sorted(csv_files):
        plot_weight_evolution(csv_file)
        
    print("-" * 50)
    
    # 2. Plot the master Alpha comparison
    plot_alpha_comparison(csv_files, target_dir)

if __name__ == "__main__":
    main()