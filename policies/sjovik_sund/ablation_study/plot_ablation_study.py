import os
import sys
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
        
        csv_files = list(exp_dir.glob("*_weights_evolution.csv"))
        if not csv_files:
            continue
        
        try:
            df = pd.read_csv(csv_files[0])
            episodes = df["episode"].values
            service_levels = df["service_level"].values
            
            # --- 1. ADD TO MASTER SERVICE LEVEL PLOT ---
            window_size = min(10, len(episodes))
            sl_smoothed = pd.Series(service_levels).rolling(window=window_size, min_periods=1).mean()
            
            ax_sl.plot(
                episodes, sl_smoothed, 
                linewidth=2.5, 
                label=f"{exp_dir.name} (Max SL: {service_levels.max():.3f})"
            )
            plotted_anything = True
            
            # --- 2. GENERATE INDIVIDUAL WEIGHT EVOLUTION PLOT ---
            # Get all feature columns (exclude the tracking columns)
            feature_names = [col for col in df.columns if col not in ["episode", "service_level"]]
            
            fig_w, ax_w = plt.subplots(figsize=(12, 6))
            
            # Plot every feature in this specific configuration
            for feature in feature_names:
                ax_w.plot(
                    episodes, 
                    df[feature], 
                    linewidth=2, 
                    alpha=0.8,
                    label=f"{feature} ({df[feature].iloc[-1]:+.2f})" # Show final weight in legend
                )
                
            ax_w.set_title(f"Weight Evolution: {exp_dir.name}", fontsize=14, fontweight="bold")
            ax_w.set_xlabel("Training Episode", fontsize=12)
            ax_w.set_ylabel("Weight Value (θ)", fontsize=12)
            ax_w.grid(True, alpha=0.3)
            
            # Move legend outside the plot so it doesn't block the lines
            ax_w.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
            
            plt.tight_layout()
            
            # Save it directly inside the experiment's specific folder!
            weight_plot_path = exp_dir / f"{exp_dir.name}_weights.png"
            fig_w.savefig(weight_plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig_w) # Close figure to free memory
            
            print(f"✅ Loaded {exp_dir.name} & saved weight plot -> {weight_plot_path.name}")
            
        except Exception as e:
            print(f"❌ Failed to process {exp_dir.name}: {e}")

    if not plotted_anything:
        print("❌ No data was plotted. Make sure your ablation study has generated CSV files!")
        return

    # --- 3. SAVE MASTER SERVICE LEVEL PLOT ---
    ax_sl.set_title("Feature Set Ablation Study: Service Level Convergence", fontsize=14, fontweight="bold")
    ax_sl.set_xlabel("Training Episode", fontsize=12)
    ax_sl.set_ylabel("Service Level (10-ep Moving Avg)", fontsize=12)
    ax_sl.grid(True, alpha=0.3)
    ax_sl.legend(loc="lower right", fontsize=10)
    
    fig_sl.tight_layout()
    output_path = study_dir / "ablation_comparison.png"
    fig_sl.savefig(output_path, dpi=300)
    plt.close(fig_sl)
    print(f"\n🚀 Success! Saved master comparison plot to: {output_path.name}")

if __name__ == "__main__":
    plot_ablation_comparison()