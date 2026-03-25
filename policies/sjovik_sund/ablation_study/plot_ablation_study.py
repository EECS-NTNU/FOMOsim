import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- BULLETPROOF PATHING ---
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))

def plot_ablation_comparison():
    study_dir = WORKSPACE_ROOT / "models" / "ablation_study"
    
    if not study_dir.exists():
        print(f"❌ Error: Could not find the ablation study directory at {study_dir}")
        return

    # Set up the master figure for the Service Level comparison
    fig_sl, ax_sl = plt.subplots(figsize=(12, 7))
    plotted_anything = False
    
    for exp_dir in sorted(study_dir.iterdir()):
        if not exp_dir.is_dir(): 
            continue
        
        # Load ALL runs for this specific experiment
        csv_files = list(exp_dir.glob("*_weights_evolution.csv"))
        if not csv_files:
            continue
        
        try:
            all_sls = []
            all_weights = []
            
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                episodes = df["episode"].values
                service_levels = df["service_level"].values
                
                # Smooth the service level for THIS specific run
                window_size = min(10, len(episodes))
                sl_smoothed = pd.Series(service_levels).rolling(window=window_size, min_periods=1).mean().values
                all_sls.append(sl_smoothed)
                
                # Extract weights
                w_df = df.drop(columns=["episode", "service_level"])
                all_weights.append(w_df.values)
            
            # Convert to numpy arrays for easy math 
            # Shape will be: [number_of_runs, number_of_episodes]
            all_sls = np.array(all_sls)
            
            # --- 1. ADD TO MASTER SERVICE LEVEL PLOT ---
            # Calculate the Mean and Standard Deviation across all runs
            sl_mean = all_sls.mean(axis=0)
            sl_std = all_sls.std(axis=0)
            
            # Plot the solid mean line
            line = ax_sl.plot(
                episodes, sl_mean, 
                linewidth=2.5, 
                label=f"{exp_dir.name} (Max Mean SL: {sl_mean.max():.3f})"
            )
            color = line[0].get_color() # Grab the color so the shading matches
            
            # Add the shaded confidence interval!
            ax_sl.fill_between(
                episodes, 
                sl_mean - sl_std, 
                sl_mean + sl_std, 
                color=color, 
                alpha=0.15 # 15% opacity so you can still see gridlines
            )
            plotted_anything = True
            
            # --- 2. GENERATE INDIVIDUAL WEIGHT EVOLUTION PLOT (Averaged) ---
            # Average the weights across all runs so the graph isn't too messy
            all_weights = np.array(all_weights)
            weights_mean = all_weights.mean(axis=0)
            feature_names = df.columns.drop(["episode", "service_level"])
            
            fig_w, ax_w = plt.subplots(figsize=(12, 6))
            
            for i, feature in enumerate(feature_names):
                ax_w.plot(
                    episodes, 
                    weights_mean[:, i], 
                    linewidth=2, 
                    alpha=0.8,
                    label=f"{feature} ({weights_mean[-1, i]:+.2f})"
                )
                
            ax_w.set_title(f"Mean Weight Evolution: {exp_dir.name} (Averaged over {len(csv_files)} runs)", fontsize=14, fontweight="bold")
            ax_w.set_xlabel("Training Episode", fontsize=12)
            ax_w.set_ylabel("Mean Weight Value (θ)", fontsize=12)
            ax_w.grid(True, alpha=0.3)
            
            # Move legend outside
            ax_w.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
            
            plt.tight_layout()
            
            # Save it
            weight_plot_path = exp_dir / f"{exp_dir.name}_mean_weights.png"
            fig_w.savefig(weight_plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig_w)
            
            print(f"✅ Loaded {exp_dir.name} ({len(csv_files)} runs averaged) & saved mean weight plot")
            
        except Exception as e:
            print(f"❌ Failed to process {exp_dir.name}: {e}")

    if not plotted_anything:
        print("❌ No data was plotted. Make sure your ablation study has generated CSV files!")
        return

    # --- 3. SAVE MASTER SERVICE LEVEL PLOT ---
    ax_sl.set_title("Feature Set Ablation Study: Service Level Convergence (Mean ± 1 Std Dev)", fontsize=14, fontweight="bold")
    ax_sl.set_xlabel("Training Episode", fontsize=12)
    ax_sl.set_ylabel("Service Level (10-ep Moving Avg)", fontsize=12)
    ax_sl.grid(True, alpha=0.3)
    ax_sl.legend(loc="lower right", fontsize=10)
    
    fig_sl.tight_layout()
    output_path = study_dir / "ablation_comparison_multi_run.png"
    fig_sl.savefig(output_path, dpi=300)
    plt.close(fig_sl)
    print(f"\n🚀 Success! Saved master comparison plot to: {output_path.name}")

if __name__ == "__main__":
    plot_ablation_comparison()