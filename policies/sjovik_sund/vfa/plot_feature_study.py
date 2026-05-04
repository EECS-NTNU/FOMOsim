import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Add the LaTeX configuration
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage[T1]{fontenc} \usepackage{mlmodern}"
})

def main():
    stats_path = "models/feature_study_results_1/feature_statistics.csv"
    corr_path = "models/feature_study_results_1/feature_correlations.csv"
    out_dir = "models/feature_study_results_1"

    if not os.path.exists(stats_path) or not os.path.exists(corr_path):
        print(f"Could not find the CSV files at {stats_path} or {corr_path}.")
        return

    # Load the data
    stats_df = pd.read_csv(stats_path, index_col=0)
    corr_df = pd.read_csv(corr_path, index_col=0)

    # Fix underscores in feature names for LaTeX compatibility
    stats_df.index = stats_df.index.str.replace('_', ' ')
    stats_df.columns = stats_df.columns.str.replace('_', ' ')
    corr_df.index = corr_df.index.str.replace('_', ' ')
    corr_df.columns = corr_df.columns.str.replace('_', ' ')

    # Custom palette from Colorpallette.png
    custom_cmap = LinearSegmentedColormap.from_list('custom_palette', ['#3758d8', '#ffffff', '#db3249'])

    # 1. Plot the Correlation Matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_df, annot=True, cmap=custom_cmap, fmt=".2f",
                linewidths=0.5, cbar_kws={'shrink': 0.8}, vmin=-1.0, vmax=1.0,
                annot_kws={"fontweight": "bold"})
    plt.title("Feature Correlation Matrix", fontsize=16, fontweight="bold")
    plt.tight_layout()
    corr_out = os.path.join(out_dir, "feature_correlation_matrix.svg")
    plt.savefig(corr_out, format="svg")
    print(f"Saved correlation matrix to {corr_out}")
    plt.close()

    # 2. Plot the Feature Statistics Table
    fig, ax = plt.subplots(figsize=(10, len(stats_df) * 0.4 + 1))
    ax.axis('off')
    ax.axis('tight')

    stats_df_rounded = stats_df.round(4)
    
    table = ax.table(cellText=stats_df_rounded.values, #type: ignore
                     rowLabels=stats_df_rounded.index, #type: ignore
                     colLabels=stats_df_rounded.columns, #type: ignore
                     cellLoc='center',
                     loc='center')
    
    table.scale(1, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    # Make all table text thicker
    for cell in table.get_celld().values():
        cell.get_text().set_fontweight("bold")

    plt.title("Feature Statistics overview", fontsize=16, fontweight="bold")
    plt.tight_layout()
    stats_out = os.path.join(out_dir, "feature_statistics_table.png")
    plt.savefig(stats_out, dpi=300, bbox_inches='tight')
    print(f"Saved statistics table to {stats_out}")
    plt.close()

if __name__ == "__main__":
    main()