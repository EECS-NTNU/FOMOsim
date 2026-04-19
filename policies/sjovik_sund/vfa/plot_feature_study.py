import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def main():
    stats_path = "models/feature_study/feature_statistics.csv"
    corr_path = "models/feature_study/feature_correlations.csv"
    out_dir = "models/feature_study"

    if not os.path.exists(stats_path) or not os.path.exists(corr_path):
        print(f"Could not find the CSV files at {stats_path} or {corr_path}.")
        return

    # Load the data
    stats_df = pd.read_csv(stats_path, index_col=0)
    corr_df = pd.read_csv(corr_path, index_col=0)

    # Custom palette from Colorpallette.png:
    # Blue: #526eca, White: #ffffff, Red: #d1495b
    custom_cmap = LinearSegmentedColormap.from_list('custom_palette', ['#526eca', '#ffffff', '#d1495b'])

    # 1. Plot the Correlation Matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_df, annot=True, cmap=custom_cmap, fmt=".2f",
                linewidths=0.5, cbar_kws={'shrink': 0.8}, vmin=-1.0, vmax=1.0)
    plt.title("Feature Correlation Matrix", fontsize=16)
    plt.tight_layout()
    corr_out = os.path.join(out_dir, "feature_correlation_matrix.png")
    plt.savefig(corr_out, dpi=300)
    print(f"Saved correlation matrix to: {corr_out}")
    plt.close()

    # 2. Plot the Feature Statistics Table
    # Calculate an appropriate figure height based on the number of features
    fig, ax = plt.subplots(figsize=(10, len(stats_df) * 0.4 + 1))
    ax.axis('off')
    ax.axis('tight')

    # Round limits for cleaner display
    stats_df_rounded = stats_df.round(4)
    
    table = ax.table(cellText=stats_df_rounded.values,
                     rowLabels=stats_df_rounded.index,
                     colLabels=stats_df_rounded.columns,
                     cellLoc='center',
                     loc='center')
    
    # Styling the table
    table.scale(1, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    plt.title("Feature Statistics overview", fontsize=16)
    plt.tight_layout()
    stats_out = os.path.join(out_dir, "feature_statistics_table.png")
    plt.savefig(stats_out, dpi=300, bbox_inches='tight')
    print(f"Saved statistics table to: {stats_out}")
    plt.close()

if __name__ == "__main__":
    main()
