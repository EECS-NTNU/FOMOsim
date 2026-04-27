import pandas as pd
import matplotlib.pyplot as plt
import os

# Define your project directories
MODELS_DIR = "policies/sjovik_sund/NN/models/"
OUTPUT_DIR = os.path.join(MODELS_DIR, "hidden_layer_study/plots")

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map the raw training log files
files = {
    "Shallow Baseline [64]": "training_log_seed1000_arch_64_20260424_154535.csv",
    "Moderate Funnel [104, 52]": "training_log_seed1000_arch_104-52_20260425_115943.csv",
    "Extended Funnel [128, 64, 32]": "training_log_seed1000_arch_128-64-32_20260425_120130.csv",
    "Highly Complex [256, 128, 64, 32]": "training_log_seed1000_arch_256-128-64-32-2_20260425_120735.csv"
}

# Standard academic colors
colors = {
    "Shallow Baseline [64]": "#d62728",           # Red
    "Moderate Funnel [104, 52]": "#ff7f0e",       # Orange
    "Extended Funnel [128, 64, 32]": "#2ca02c",   # Green
    "Highly Complex [256, 128, 64, 32]": "#1f77b4"# Blue
}

# Smoothing window for the rolling average (adjust this if you want it smoother/noisier)
SMOOTHING_WINDOW = 20

# ---------------------------------------------------------
# 1. GENERATE AND SAVE THE TD LOSS PLOT
# ---------------------------------------------------------
fig_loss, ax_loss = plt.subplots(figsize=(8, 6))

for arch, filename in files.items():
    filepath = os.path.join(MODELS_DIR, filename)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        smoothed_loss = df['mean_loss'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
        ax_loss.plot(df['episode'], smoothed_loss, label=arch, color=colors[arch], linewidth=2)
    else:
        print(f"Warning: Could not find {filepath}")

ax_loss.set_title('TD Loss (Error) Over Time', fontsize=14, fontweight='bold')
ax_loss.set_xlabel('Offline Training Episode', fontsize=12)
ax_loss.set_ylabel('Mean TD Loss (Smoothed)', fontsize=12)
ax_loss.set_yscale('log') # Log scale beautifully shows exponential decay
ax_loss.grid(True, linestyle='--', alpha=0.7)
ax_loss.legend(loc='lower right')

plt.tight_layout()
loss_output_path = os.path.join(OUTPUT_DIR, 'td_loss_curve.pdf')
fig_loss.savefig(loss_output_path, format='pdf', dpi=300)
print(f"Saved TD Loss plot to: {loss_output_path}")
plt.close(fig_loss)


# ---------------------------------------------------------
# 2. GENERATE AND SAVE THE VALUE SPREAD PLOT
# ---------------------------------------------------------
fig_spread, ax_spread = plt.subplots(figsize=(8, 6))

for arch, filename in files.items():
    filepath = os.path.join(MODELS_DIR, filename)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        smoothed_spread = df['mean_value_spread'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
        ax_spread.plot(df['episode'], smoothed_spread, label=arch, color=colors[arch], linewidth=2)

ax_spread.set_title('Mean Value Spread (State Differentiation)', fontsize=14, fontweight='bold')
ax_spread.set_xlabel('Offline Training Episode', fontsize=12)
ax_spread.set_ylabel('Value Spread (Smoothed)', fontsize=12)
ax_spread.grid(True, linestyle='--', alpha=0.7)
ax_spread.legend(loc='lower right')

plt.tight_layout()
spread_output_path = os.path.join(OUTPUT_DIR, 'value_spread_curve.pdf')
fig_spread.savefig(spread_output_path, format='pdf', dpi=300)
print(f"Saved Value Spread plot to: {spread_output_path}")
plt.close(fig_spread)
