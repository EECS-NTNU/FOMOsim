"""
Pareto Experiment Analysis and Visualization

This script processes the results from run_pareto_experiment.py and creates:
1. Pareto frontier plots
2. Statistical analysis tables
3. Time series comparisons
4. Sensitivity analysis

Author: Generated for FOMOsim thesis computational study
Date: 2025-11-23
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


def load_results(filename='pareto_experiment_results.csv'):
    """Load experiment results from CSV."""
    results_dir = Path('./policies/sjovik_sund/simulation_results/')
    filepath = results_dir / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}\nPlease run run_pareto_experiment.py first!")
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} experiment results from {filepath}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(f"r_M values: {sorted(df['r_M'].unique())}")
    return df


def create_summary_statistics(df):
    """Create summary statistics grouped by r_M."""
    
    # Aggregate by r_M (average across seeds)
    summary = df.groupby('r_M').agg({
        'total_service_cost': ['mean', 'std'],
        'starvations': ['mean', 'std'],
        'long_congestions': ['mean', 'std'],
        'short_congestions': ['mean', 'std'],
        'avg_criticality': ['mean', 'std'],
        'pct_critical_05': ['mean', 'std'],
        'pct_critical_07': ['mean', 'std'],
        'pct_critical_09': ['mean', 'std'],
        'service_level': ['mean', 'std'],
        'bike_pickups': ['mean', 'std'],
        'bike_deliveries': ['mean', 'std'],
        'solve_time_s': ['mean', 'std'],
    }).round(3)
    
    return summary


def plot_pareto_frontier(df, output_dir):
    """
    Plot the Pareto frontier: Service Cost vs. Maintenance Quality
    """
    # Aggregate by r_M
    agg_df = df.groupby('r_M').agg({
        'total_service_cost': ['mean', 'std'],
        'avg_criticality': ['mean', 'std'],
        'pct_critical_07': ['mean', 'std'],
    }).reset_index()
    
    agg_df.columns = ['r_M', 'service_cost_mean', 'service_cost_std', 
                      'criticality_mean', 'criticality_std',
                      'pct_critical_mean', 'pct_critical_std']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Service Cost vs. Average Criticality
    ax1.errorbar(agg_df['criticality_mean'], agg_df['service_cost_mean'],
                xerr=agg_df['criticality_std'], yerr=agg_df['service_cost_std'],
                fmt='o-', capsize=5, capthick=2, markersize=8, linewidth=2)
    
    # Annotate points with r_M values
    for _, row in agg_df.iterrows():
        ax1.annotate(f"r_M={row['r_M']:.2f}", 
                    xy=(row['criticality_mean'], row['service_cost_mean']),
                    xytext=(10, -5), textcoords='offset points',
                    fontsize=9, alpha=0.7)
    
    ax1.set_xlabel('Average Maintenance Criticality', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Service Cost\n(Starvations + Congestions)', fontsize=12, fontweight='bold')
    ax1.set_title('Pareto Frontier: Service Cost vs. Fleet Maintenance', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Service Cost vs. % Critical Bikes
    ax2.errorbar(agg_df['pct_critical_mean'], agg_df['service_cost_mean'],
                xerr=agg_df['pct_critical_std'], yerr=agg_df['service_cost_std'],
                fmt='s-', capsize=5, capthick=2, markersize=8, linewidth=2, color='darkorange')
    
    for _, row in agg_df.iterrows():
        ax2.annotate(f"r_M={row['r_M']:.2f}", 
                    xy=(row['pct_critical_mean'], row['service_cost_mean']),
                    xytext=(10, -5), textcoords='offset points',
                    fontsize=9, alpha=0.7)
    
    ax2.set_xlabel('% Bikes with Criticality > 0.7', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Service Cost', fontsize=12, fontweight='bold')
    ax2.set_title('Pareto Frontier: Service Cost vs. Critical Bikes', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = output_dir / 'pareto_frontier.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def plot_maintenance_impact(df, output_dir):
    """
    Plot how r_M affects various maintenance and service metrics.
    """
    agg_df = df.groupby('r_M').agg({
        'avg_criticality': 'mean',
        'pct_critical_05': 'mean',
        'pct_critical_07': 'mean',
        'pct_critical_09': 'mean',
        'starvations': 'mean',
        'long_congestions': 'mean',
        'short_congestions': 'mean',
    }).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Maintenance Criticality vs. r_M
    ax = axes[0, 0]
    ax.plot(agg_df['r_M'], agg_df['avg_criticality'], 'o-', linewidth=2, markersize=8, label='Average')
    ax.fill_between(agg_df['r_M'], 0, agg_df['avg_criticality'], alpha=0.3)
    ax.set_xlabel('Maintenance Reward Weight (r_M)', fontweight='bold')
    ax.set_ylabel('Average Maintenance Criticality', fontweight='bold')
    ax.set_title('Impact of r_M on Fleet Maintenance Level', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: % Critical Bikes by Threshold
    ax = axes[0, 1]
    ax.plot(agg_df['r_M'], agg_df['pct_critical_05'], 's-', linewidth=2, markersize=7, label='> 0.5')
    ax.plot(agg_df['r_M'], agg_df['pct_critical_07'], '^-', linewidth=2, markersize=7, label='> 0.7')
    ax.plot(agg_df['r_M'], agg_df['pct_critical_09'], 'o-', linewidth=2, markersize=7, label='> 0.9')
    ax.set_xlabel('Maintenance Reward Weight (r_M)', fontweight='bold')
    ax.set_ylabel('% of Bikes Above Threshold', fontweight='bold')
    ax.set_title('Critical Bikes by Criticality Threshold', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Service Quality Metrics
    ax = axes[1, 0]
    ax.plot(agg_df['r_M'], agg_df['starvations'], 'o-', linewidth=2, markersize=8, label='Starvations', color='red')
    ax.plot(agg_df['r_M'], agg_df['long_congestions'], 's-', linewidth=2, markersize=7, label='Long Congestions', color='orange')
    ax.plot(agg_df['r_M'], agg_df['short_congestions'], '^-', linewidth=2, markersize=7, label='Short Congestions', color='gold')
    ax.set_xlabel('Maintenance Reward Weight (r_M)', fontweight='bold')
    ax.set_ylabel('Number of Events', fontweight='bold')
    ax.set_title('Service Quality Metrics vs. r_M', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Total Service Cost
    ax = axes[1, 1]
    total_cost = agg_df['starvations'] + agg_df['long_congestions'] + agg_df['short_congestions']
    ax.plot(agg_df['r_M'], total_cost, 'D-', linewidth=3, markersize=8, color='darkred')
    ax.fill_between(agg_df['r_M'], 0, total_cost, alpha=0.3, color='red')
    ax.set_xlabel('Maintenance Reward Weight (r_M)', fontweight='bold')
    ax.set_ylabel('Total Service Cost', fontweight='bold')
    ax.set_title('Total Service Cost vs. r_M', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = output_dir / 'maintenance_impact.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def plot_tradeoff_analysis(df, output_dir):
    """
    Create detailed tradeoff analysis plots.
    """
    agg_df = df.groupby('r_M').agg({
        'total_service_cost': 'mean',
        'avg_criticality': 'mean',
        'bike_pickups': 'mean',
        'bike_deliveries': 'mean',
    }).reset_index()
    
    # Normalize metrics to [0, 1] for comparison
    for col in ['total_service_cost', 'avg_criticality', 'bike_pickups', 'bike_deliveries']:
        col_min = agg_df[col].min()
        col_max = agg_df[col].max()
        if col_max - col_min > 0:
            agg_df[f'{col}_norm'] = (agg_df[col] - col_min) / (col_max - col_min)
        else:
            agg_df[f'{col}_norm'] = 0.5
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Normalized metrics
    ax1.plot(agg_df['r_M'], agg_df['total_service_cost_norm'], 'o-', linewidth=2, markersize=8, label='Service Cost')
    ax1.plot(agg_df['r_M'], agg_df['avg_criticality_norm'], 's-', linewidth=2, markersize=8, label='Avg Criticality')
    ax1.plot(agg_df['r_M'], agg_df['bike_pickups_norm'], '^-', linewidth=2, markersize=8, label='Bike Pickups')
    ax1.set_xlabel('Maintenance Reward Weight (r_M)', fontweight='bold')
    ax1.set_ylabel('Normalized Value [0, 1]', fontweight='bold')
    ax1.set_title('Normalized Metrics Comparison', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([-0.05, 1.05])
    
    # Plot 2: Efficiency frontier (inverse criticality vs. service cost)
    # Lower is better for both, so this shows the Pareto-optimal solutions
    ax2.scatter(1.0 - agg_df['avg_criticality_norm'], 1.0 - agg_df['total_service_cost_norm'],
               s=200, c=agg_df['r_M'], cmap='viridis', edgecolors='black', linewidths=2, alpha=0.8)
    
    for _, row in agg_df.iterrows():
        ax2.annotate(f"{row['r_M']:.2f}",
                    xy=(1.0 - row['avg_criticality_norm'], 1.0 - row['total_service_cost_norm']),
                    fontsize=10, ha='center', va='center', fontweight='bold')
    
    ax2.set_xlabel('Maintenance Quality (1 - Criticality)', fontweight='bold')
    ax2.set_ylabel('Service Quality (1 - Cost)', fontweight='bold')
    ax2.set_title('Efficiency Frontier: Both Objectives Maximized', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('r_M Value', fontweight='bold')
    
    plt.tight_layout()
    filepath = output_dir / 'tradeoff_analysis.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def generate_latex_table(df, output_dir):
    """
    Generate a LaTeX table for thesis.
    """
    summary = df.groupby('r_M').agg({
        'total_service_cost': ['mean', 'std'],
        'starvations': 'mean',
        'long_congestions': 'mean',
        'avg_criticality': ['mean', 'std'],
        'pct_critical_07': ['mean', 'std'],
        'bike_pickups': 'mean',
        'bike_deliveries': 'mean',
    }).round(2)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary = summary.reset_index()
    
    latex_str = summary.to_latex(
        index=False,
        caption='Pareto Experiment Results: Impact of Maintenance Reward Weight',
        label='tab:pareto_results',
        column_format='c' + 'r' * (len(summary.columns) - 1),
        escape=False,
        float_format="%.2f"
    )
    
    filepath = output_dir / 'pareto_results_table.tex'
    with open(filepath, 'w') as f:
        f.write(latex_str)
    print(f"Saved: {filepath}")
    
    # Also save as CSV for easy viewing
    csv_filepath = output_dir / 'pareto_summary.csv'
    summary.to_csv(csv_filepath, index=False)
    print(f"Saved: {csv_filepath}")


def find_knee_point(df):
    """
    Find the "knee point" on the Pareto frontier - the best tradeoff.
    Uses distance to ideal point method.
    """
    agg_df = df.groupby('r_M').agg({
        'total_service_cost': 'mean',
        'avg_criticality': 'mean',
    }).reset_index()
    
    # Normalize to [0, 1]
    service_norm = (agg_df['total_service_cost'] - agg_df['total_service_cost'].min()) / \
                   (agg_df['total_service_cost'].max() - agg_df['total_service_cost'].min())
    criticality_norm = (agg_df['avg_criticality'] - agg_df['avg_criticality'].min()) / \
                       (agg_df['avg_criticality'].max() - agg_df['avg_criticality'].min())
    
    # Distance to ideal point (0, 0) - both objectives minimized
    distances = np.sqrt(service_norm**2 + criticality_norm**2)
    
    knee_idx = distances.idxmin()
    knee_r_M = agg_df.loc[knee_idx, 'r_M']
    
    return knee_r_M, agg_df.loc[knee_idx]


def main():
    """Main analysis function."""
    print("\n" + "="*80)
    print("PARETO EXPERIMENT ANALYSIS")
    print("="*80 + "\n")
    
    # Create output directory
    output_dir = Path('./policies/sjovik_sund/simulation_results/pareto_plots/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    try:
        df = load_results()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
    
    # Create summary statistics
    print("Creating summary statistics...")
    summary = create_summary_statistics(df)
    print("\nSummary Statistics (mean ± std):")
    print(summary)
    print()
    
    # Find knee point
    print("Finding knee point (optimal tradeoff)...")
    knee_r_M, knee_point = find_knee_point(df)
    print(f"\nRecommended r_M value: {knee_r_M:.3f}")
    print(f"  Service Cost: {knee_point['total_service_cost']:.1f}")
    print(f"  Avg Criticality: {knee_point['avg_criticality']:.3f}")
    print()
    
    # Generate plots
    print("Generating visualizations...")
    plot_pareto_frontier(df, output_dir)
    plot_maintenance_impact(df, output_dir)
    plot_tradeoff_analysis(df, output_dir)
    
    # Generate tables
    print("\nGenerating tables...")
    generate_latex_table(df, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print(f"All outputs saved to: {output_dir}")
    print("="*80 + "\n")
    
    print("Key Findings:")
    print(f"1. Optimal r_M (knee point): {knee_r_M:.3f}")
    print(f"2. Service cost range: {df.groupby('r_M')['total_service_cost'].mean().min():.1f} - {df.groupby('r_M')['total_service_cost'].mean().max():.1f}")
    print(f"3. Criticality range: {df.groupby('r_M')['avg_criticality'].mean().min():.3f} - {df.groupby('r_M')['avg_criticality'].mean().max():.3f}")
    
    # Improvement analysis
    no_maint = df[df['r_M'] == 0.0].groupby('r_M')['avg_criticality'].mean().values[0]
    with_maint = df[df['r_M'] == knee_r_M].groupby('r_M')['avg_criticality'].mean().values[0]
    improvement = 100.0 * (no_maint - with_maint) / no_maint if no_maint > 0 else 0
    print(f"4. Criticality reduction at optimal r_M: {improvement:.1f}%")


if __name__ == "__main__":
    main()

