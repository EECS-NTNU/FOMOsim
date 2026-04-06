import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path

def moving_average(data, window_size=5):
    """Calculates the moving average to smooth out the learning curves."""
    if len(data) < window_size:
        return data # Not enough data to smooth
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_learning_curves(folder_path, apply_smoothing=False, window_size=5):
    """
    Finds all learning curve .npy files in the target folder and plots them.
    """
    # Look for all .npy files in the specified directory
    search_pattern = Path(folder_path) / "*_learning_curve.npy"
    file_list = glob.glob(str(search_pattern))
    
    if not file_list:
        print(f"No '_learning_curve.npy' files found in {folder_path}!")
        return

    # Set up the plot aesthetics (Academic style)
    plt.figure(figsize=(12, 7))
    plt.title("Sensitivity Analysis: Service Level Convergence", fontsize=14, fontweight='bold')
    plt.xlabel("Training Episode", fontsize=12)
    plt.ylabel(f"Service Level{' (' + str(window_size) + '-ep Moving Avg)' if apply_smoothing else ''}", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    final_values = []

    # Load and plot each file
    for file_path in sorted(file_list):
        # Extract a clean label from the filename (e.g., 'vfa_G0.99_Ts0.1_Te0.001')
        filename = Path(file_path).name
        label = filename.replace("_learning_curve.npy", "")
        
        # Load the numpy array
        service_levels = np.load(file_path)
        
        # Apply smoothing if requested
        if apply_smoothing:
            plot_data = moving_average(service_levels, window_size)
            # Adjust x-axis so it aligns correctly with the episodes
            x_axis = range(window_size - 1, len(service_levels))
        else:
            plot_data = service_levels
            x_axis = range(len(service_levels))
            
        # Plot the curve
        plt.plot(x_axis, plot_data, label=label, linewidth=2, alpha=0.8)
        
        # --- NEW: Save the final value for ranking ---
        final_values.append((label, plot_data[-1]))
        
    # print all final values for transparency
    print("\nFinal Service Levels for Each Configuration:")
    for label, val in final_values:
        print(f"  {label}: {val:.4f}")

    # --- NEW: Print the Top 3 and Bottom 3 ---
    final_values.sort(key=lambda x: x[1], reverse=True)
    print("\n--- GRID SEARCH RESULTS (Ranked by final episode) ---")
    print("TOP 3 CONFIGURATIONS:")
    for label, val in final_values[:3]:
        print(f"  {label}: {val:.4f}")
        
    print("\nBOTTOM 3 CONFIGURATIONS (The Crashers):")
    for label, val in final_values[-3:]:
        print(f"  {label}: {val:.4f}")
    print("---------------------------------------------------\n")

    # Place legend outside the plot if there are many configurations
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)
    plt.tight_layout() # Ensures the legend doesn't get cut off
    
    save_path = "policies/sjovik_sund/sensitivity_analysis/sensitivity_analysis_learning_curves.png"
    
    plt.savefig(save_path, dpi=300)
    print(f"Saved learning curves plot to: {save_path}")
    
    # Show the plot
    plt.show()
    
    

if __name__ == "__main__":
    TARGET_FOLDER = "models/grid_search_20260401_0856" 
    
    print(f"Scanning for learning curves in: {TARGET_FOLDER}")
    
    # Set apply_smoothing=True if the raw lines are too chaotic
    plot_learning_curves(TARGET_FOLDER, apply_smoothing=True, window_size=5)