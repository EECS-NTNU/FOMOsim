'''import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re

# ─────────────────────────────────────────────────────────────────────────────
# Configuration & Settings
# ─────────────────────────────────────────────────────────────────────────────

# The parent directory containing all the seed folders
BASE_MODELS_DIR = Path("policies/sjovik_sund/sensitivity_analysis/models")

# String to match the grid search folders
GRID_SEARCH_PATTERN = "grid_search_*"

# Fallback list if running directly from an IDE instead of the terminal.
# Leave empty [] to plot ALL found configurations.
HARDCODED_TARGET_CONFIGS = [
    # "G0.99_Ts1.0_Te0.05",
]

# Moving average window to smooth out the noisy RL service levels
SMOOTHING_WINDOW = 10  

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Applies a simple moving average to smooth the learning curves."""
    if window_size < 2:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def extract_config_name(filename: str) -> str:
    """
    Extracts the config parameters from the filename.
    Matches: vfa_G0.99_Ts1.0_Te0.05_learning_curve.npy -> G0.99_Ts1.0_Te0.05
    """
    match = re.search(r"vfa_(.*)_learning_curve\.npy", filename)
    if match:
        return match.group(1)
    return "Unknown_Config"

# ─────────────────────────────────────────────────────────────────────────────
# Main Plotting Logic
# ─────────────────────────────────────────────────────────────────────────────

def plot_aggregated_learning_curves(target_configs: list):
    # Find all directories matching the grid search pattern across all seeds
    seed_dirs = list(BASE_MODELS_DIR.glob(GRID_SEARCH_PATTERN))
    
    if not seed_dirs:
        print(f"No directories found matching {BASE_MODELS_DIR / GRID_SEARCH_PATTERN}")
        return

    print(f"Found {len(seed_dirs)} seed directories.")
    
    if target_configs:
        print(f"Filtering for specific configurations: {target_configs}")
    else:
        print("No specific targets provided. Automatically plotting ALL configurations.")

    # Dictionary to aggregate curves: { "Config_Name": [array_seed1, array_seed2, ...] }
    learning_data = {}

    # 1. Gather all data across all seeds
    for seed_dir in seed_dirs:
        npy_files = list(seed_dir.glob("*_learning_curve.npy"))
        
        for npy_file in npy_files:
            config_name = extract_config_name(npy_file.name)
            
            # IF target_configs is NOT empty AND this config is NOT in the list, skip it.
            if target_configs and config_name not in target_configs:
                continue
                
            # Load the numpy array (1D array of service levels per episode)
            curve = np.load(npy_file)
            
            if config_name not in learning_data:
                learning_data[config_name] = []
            
            learning_data[config_name].append(curve)

    if not learning_data:
        print("No learning curve data found for the specified target configurations.")
        return

    # 2. Process and Plot
    plt.figure(figsize=(12, 7))
    
    for config_name, curves in learning_data.items():
        # Ensure all curves have the same length before stacking
        min_length = min(len(c) for c in curves)
        truncated_curves = np.array([c[:min_length] for c in curves])
        
        # Calculate Mean and Standard Deviation across seeds
        mean_curve = np.mean(truncated_curves, axis=0)
        std_curve = np.std(truncated_curves, axis=0)
        
        # Apply smoothing for better visual clarity
        smooth_mean = moving_average(mean_curve, SMOOTHING_WINDOW)
        smooth_std = moving_average(std_curve, SMOOTHING_WINDOW)
        
        # Adjust x-axis to account for the smoothing window offset
        episodes = np.arange(SMOOTHING_WINDOW - 1, min_length)
        
        # Plot the mean line
        line, = plt.plot(episodes, smooth_mean, label=f"{config_name} ({len(curves)} seeds)", linewidth=2)
        
        # Fill the standard deviation bounds
        plt.fill_between(
            episodes, 
            smooth_mean - smooth_std, 
            smooth_mean + smooth_std, 
            color=line.get_color(), 
            alpha=0.2
        )

    # 3. Graph Formatting
    plt.title(f"VFA Learning Curves across {len(seed_dirs)} Seeds", fontsize=16, fontweight='bold')
    plt.xlabel("Training Episode", fontsize=14)
    plt.ylabel("Service Level", fontsize=14)
    
    # Add a grid, legend, and layout tightening
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    
    # Save the plot in the base models directory
    save_path = BASE_MODELS_DIR / "aggregated_learning_curves.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved successfully to: {save_path}")
    
    # Show the plot interactively
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot aggregated learning curves from multiple seeds.")
    parser.add_argument(
        "--configs", 
        nargs="*", 
        default=HARDCODED_TARGET_CONFIGS,
        help="List of specific configurations to plot (e.g., G0.99_Ts1.0_Te0.05). Leave blank to plot all."
    )
    
    args = parser.parse_args()
    plot_aggregated_learning_curves(args.configs)'''
    
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# ─────────────────────────────────────────────────────────────────────────────
# Configuration & Settings
# ─────────────────────────────────────────────────────────────────────────────

# Automatically point to your actual models folder
DEFAULT_MODELS_DIR = Path("policies/sjovik_sund/vfa/models")

# Moving average window to smooth out the noisy RL service levels
SMOOTHING_WINDOW = 10  

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Applies a simple moving average to smooth the learning curves."""
    if window_size < 2:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# ─────────────────────────────────────────────────────────────────────────────
# Main Plotting Logic
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(models_dir: Path):
    # Find all .npy learning curve files in the specified directory
    npy_files = list(models_dir.glob("*_learning_curve.npy"))
    
    if not npy_files:
        print(f"No learning curve (.npy) files found in {models_dir}")
        print("Make sure you have run train_vfa.py and that the files are saved there.")
        return

    print(f"Found {len(npy_files)} learning curve(s) to plot.")

    plt.figure(figsize=(12, 7))
    
    for npy_file in sorted(npy_files):
        # Extract the timestamp/name from the file for the legend
        # e.g., "vfa_trained_20260404_151305_learning_curve.npy" -> "20260404_151305"
        name_parts = npy_file.stem.split('_')
        run_identifier = "_".join(name_parts[2:4]) if len(name_parts) >= 4 else npy_file.stem
        
        # Load the numpy array (1D array of service levels per episode)
        curve = np.load(npy_file)
        
        # Apply smoothing for better visual clarity
        smooth_mean = moving_average(curve, SMOOTHING_WINDOW)
        
        # Adjust x-axis to account for the smoothing window offset
        episodes = np.arange(SMOOTHING_WINDOW - 1, len(curve))
        
        # Plot the smoothed line
        plt.plot(episodes, smooth_mean, label=f"Run: {run_identifier}", linewidth=2)
        
        # Plot a faint, transparent version of the raw, noisy data in the background
        plt.plot(np.arange(len(curve)), curve, alpha=0.15, color='gray')

    # 3. Graph Formatting
    plt.title(f"VFA Learning Curves Comparison", fontsize=16, fontweight='bold')
    plt.xlabel("Training Episode", fontsize=14)
    plt.ylabel("Service Level", fontsize=14)
    
    # Add a grid, legend, and layout tightening
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    
    # Save the plot in the same directory as the models
    save_path = models_dir / "alpha_comparison_curves.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved successfully to: {save_path}")
    
    # Show the plot interactively
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot standalone learning curves from a directory.")
    parser.add_argument(
        "--dir", 
        type=str, 
        default=str(DEFAULT_MODELS_DIR),
        help="Path to the directory containing the .npy files."
    )
    
    args = parser.parse_args()
    
    # Fallback just in case the script is run from inside the vfa directory itself
    models_path = Path(args.dir)
    if not models_path.exists():
        fallback_path = Path("models")
        if fallback_path.exists():
            models_path = fallback_path
            print(f"Warning: {args.dir} not found. Falling back to local 'models/' directory.")
            
    plot_learning_curves(models_path)