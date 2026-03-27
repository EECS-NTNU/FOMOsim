import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- BULLETPROOF PATHING ---
# 1. Automatically find the FOMOsim root folder (assuming this script is in the root)
WORKSPACE_ROOT = Path(__file__).resolve().parent
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))

# 2. Build the absolute path to the models directory
MODELS_DIR = WORKSPACE_ROOT / "models"

def inspect_latest_model():
    if not MODELS_DIR.exists():
        print(f"Error: Directory does not exist -> {MODELS_DIR}")
        print(f"Make sure you have run train_vfa.py at least once to generate models!")
        return

    # Find all .pkl files in the directory
    pkl_files = list(MODELS_DIR.glob("*.pkl"))
    if not pkl_files:
        print(f"No .pkl files found in {MODELS_DIR}")
        return

    # Automatically grab the most recently saved .pkl file
    latest_pkl = max(pkl_files, key=lambda p: p.stat().st_mtime)
    npy_name = latest_pkl.name.replace(".pkl", "_learning_curve.npy")
    latest_npy = MODELS_DIR / npy_name

    print("==================================================")
    print(f" UNPACKING LATEST MODEL")
    print("==================================================")
    print(f"Model File: {latest_pkl.name}")
    print(f"Location:   {latest_pkl.parent}")
    
    # ---------------------------------------------------------
    # 1. Unpack the .pkl (Model Weights & Hyperparameters)
    # ---------------------------------------------------------
    with open(latest_pkl, "rb") as f:
        payload = pickle.load(f)
        
    print("\n--- HYPERPARAMETERS ---")
    print(f"Number of Features (n_features): {payload.get('n_features', 'N/A')}")
    print(f"Learning Rate (alpha)          : {payload.get('alpha', 'N/A')}")
    print(f"Discount Factor (gamma)        : {payload.get('gamma', 'N/A')}")
    
    print("\n--- LEARNED WEIGHTS (THETA) ---")
    theta = payload.get('theta', [])
    for i, weight in enumerate(theta):
        print(f"  Feature {i:02d}: {weight:+.6f}")

    # ---------------------------------------------------------
    # 2. Unpack the .npy (Learning Curve / Service Levels)
    # ---------------------------------------------------------
    print("\n==================================================")
    print(f" LEARNING CURVE DATA")
    print("==================================================")
    
    learning_curve = None
    if latest_npy.exists():
        learning_curve = np.load(latest_npy)
        print(f"\nTotal Training Episodes: {len(learning_curve)}")
        print(f"Best Service Level     : {np.max(learning_curve):.4f} (Achieved in Episode {np.argmax(learning_curve) + 1})")
        print(f"Final Service Level    : {learning_curve[-1]:.4f}")
        
        print("\n--- Quick Glance (First 5 Episodes) ---")
        for i, val in enumerate(learning_curve[:5]):
            print(f"  Episode {i+1:03d}: {val:.4f}")
            
        print("\n--- Quick Glance (Last 5 Episodes) ---")
        for i, val in enumerate(learning_curve[-5:]):
            ep_num = len(learning_curve) - 4 + i
            print(f"  Episode {ep_num:03d}: {val:.4f}")
    else:
        print(f"\n⚠️ No matching learning curve (.npy) file found for this model at: {latest_npy}")

    # ---------------------------------------------------------
    # 3. GENERATE PLOTS
    # ---------------------------------------------------------
    print("\n==================================================")
    print(f" GENERATING PLOTS")
    print("==================================================")

    # Plot 1: Learning Curve
    if learning_curve is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(learning_curve) + 1), learning_curve, marker='o', linestyle='-', color='b', alpha=0.7)
        plt.title(f"VFA Learning Curve\n({latest_npy.name})", fontsize=14)
        plt.xlabel("Training Episode", fontsize=12)
        plt.ylabel("Service Level", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        curve_plot_path = MODELS_DIR / latest_npy.name.replace(".npy", "_plot.png")
        plt.savefig(curve_plot_path, dpi=300)
        plt.close()
        print(f"✅ Saved Learning Curve Plot  -> {curve_plot_path.name}")

    # Plot 2: Feature Weights (Theta) Bar Chart
    if len(theta) > 0:
        # Try to pull the actual feature names from your code
        try:
            from policies.sjovik_sund.vfa.vfa_features import get_feature_names
            feature_names = get_feature_names(maintenance_enabled=True, shift_timing_enabled=False)
            if len(feature_names) != len(theta):
                feature_names = [f"Feature {i}" for i in range(len(theta))]
        except Exception:
            feature_names = [f"Feature {i}" for i in range(len(theta))]

        plt.figure(figsize=(12, max(8, len(theta) * 0.3)))
        
        sorted_indices = np.argsort(theta)
        sorted_theta = np.array(theta)[sorted_indices]
        sorted_names = np.array(feature_names)[sorted_indices]
        
        colors = ['#d62728' if val < 0 else '#2ca02c' for val in sorted_theta]
        
        plt.barh(sorted_names, sorted_theta, color=colors)
        plt.title(f"Learned VFA Feature Weights (Theta)\n({latest_pkl.name})", fontsize=14)
        plt.xlabel("Weight Value", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        weights_plot_path = MODELS_DIR / latest_pkl.name.replace(".pkl", "_weights_plot.png")
        plt.savefig(weights_plot_path, dpi=300)
        plt.close()
        print(f"✅ Saved Feature Weights Plot -> {weights_plot_path.name}")

if __name__ == "__main__":
    inspect_latest_model()