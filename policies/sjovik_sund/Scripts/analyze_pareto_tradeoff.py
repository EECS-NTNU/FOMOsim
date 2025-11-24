#!/usr/bin/env python3
"""
Comprehensive Pareto analysis for maintenance vs service level tradeoff.

This script:
1. Loads the pareto_breakdowns.csv data
2. Correctly identifies the Pareto front
3. Creates multiple visualizations to understand tradeoffs
4. Provides recommendations on parameter settings

Usage: Run from repo root with venv active
    & .\.venv\Scripts\python.exe .\policies\sjovik_sund\Scripts\analyze_pareto_tradeoff.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def is_pareto_efficient(costs, return_mask=True):
    """
    Find the Pareto-efficient points.
    
    Args:
        costs: An (n_points, n_costs) array where we want to MINIMIZE all costs
        return_mask: True to return a mask, False to return indices
    
    Returns:
        A boolean mask or indices of Pareto-efficient points
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point with at least one cost <= this point, 
            # but remove points that are dominated (all costs >=)
            # A point dominates another if it's better in at least one objective 
            # and no worse in all others
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1) | \
                                          np.all(costs[is_efficient] <= c, axis=1)
    
    return is_efficient if return_mask else np.where(is_efficient)[0]


def load_and_clean_data(csv_path):
    """Load pareto data and remove invalid entries."""
    df = pd.read_csv(csv_path)
    
    # Remove rows with NaN values in key columns
    df = df.dropna(subset=['starv_contrib', 'cong_contrib', 'dev_contrib', 'maint_time'])
    
    # Calculate service total (sum of all service penalties - lower is better)
    df['service_total'] = df['starv_contrib'] + df['cong_contrib'] + df['dev_contrib']
    
    print(f"Loaded {len(df)} valid data points")
    print(f"\nService total range: [{df['service_total'].min():.2f}, {df['service_total'].max():.2f}]")
    print(f"Maintenance time range: [{df['maint_time'].min():.2f}, {df['maint_time'].max():.2f}] minutes")
    
    return df


def identify_pareto_front(df):
    """
    Identify Pareto-efficient solutions.
    
    We want to:
    - MINIMIZE service_total (less starvation, congestion, deviation)
    - MAXIMIZE maintenance_time (more maintenance is better for bike quality)
    
    For Pareto detection, we treat this as minimizing both:
    - service_total (as is)
    - negative_maintenance (i.e., minimize -maintenance_time = maximize maintenance_time)
    """
    # Create cost matrix: [service_total, -maint_time]
    # Both should be minimized for Pareto optimality
    costs = np.column_stack([
        df['service_total'].values,
        -df['maint_time'].values  # Negative because we want to maximize maintenance
    ])
    
    pareto_mask = is_pareto_efficient(costs)
    df['is_pareto'] = pareto_mask
    
    pareto_points = df[pareto_mask].copy()
    pareto_points = pareto_points.sort_values('service_total')
    
    print(f"\nFound {pareto_mask.sum()} Pareto-efficient points out of {len(df)} total points")
    
    return df, pareto_points


def plot_main_tradeoff(df, pareto_points, outdir):
    """Create the main service vs maintenance tradeoff plot."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot all points
    ax.scatter(df['service_total'], df['maint_time'], 
               c='lightgray', s=80, alpha=0.6, label='All solutions', zorder=2)
    
    # Plot Pareto front
    ax.scatter(pareto_points['service_total'], pareto_points['maint_time'],
               c='red', s=120, marker='D', edgecolors='darkred', linewidth=1.5,
               label='Pareto-efficient', zorder=3)
    
    # Connect Pareto points
    if len(pareto_points) > 1:
        ax.plot(pareto_points['service_total'], pareto_points['maint_time'],
                'r--', alpha=0.5, linewidth=2, zorder=1)
    
    ax.set_xlabel('Service Penalty (lower = better)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Maintenance Time (minutes, higher = better)', fontsize=12, fontweight='bold')
    ax.set_title('Pareto Front: Service Quality vs Maintenance Tradeoff', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add annotations for best/worst cases
    best_service = pareto_points.loc[pareto_points['service_total'].idxmin()]
    best_maintenance = pareto_points.loc[pareto_points['maint_time'].idxmax()]
    
    ax.annotate('Best Service\n(least maintenance)', 
                xy=(best_service['service_total'], best_service['maint_time']),
                xytext=(-60, -30), textcoords='offset points',
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
    
    ax.annotate('Most Maintenance\n(worse service)', 
                xy=(best_maintenance['service_total'], best_maintenance['maint_time']),
                xytext=(60, 30), textcoords='offset points',
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
    
    plt.tight_layout()
    outfile = os.path.join(outdir, 'pareto_analysis_main.png')
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    print(f"Saved: {outfile}")
    plt.close()


def plot_service_components_breakdown(df, pareto_points, outdir):
    """Show how starvation, congestion, and deviation contribute to service penalty."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    components = [
        ('starv_contrib', 'Starvation', 'Blues'),
        ('cong_contrib', 'Congestion', 'Oranges'),
        ('dev_contrib', 'Deviation', 'Greens')
    ]
    
    for ax, (col, title, cmap) in zip(axes, components):
        scatter = ax.scatter(df['service_total'], df['maint_time'],
                            c=df[col], cmap=cmap, s=100, alpha=0.7, edgecolors='gray', linewidth=0.5)
        
        # Highlight Pareto points
        ax.scatter(pareto_points['service_total'], pareto_points['maint_time'],
                  marker='D', s=150, edgecolors='red', facecolors='none', linewidth=2)
        
        ax.set_xlabel('Service Penalty', fontsize=10)
        ax.set_ylabel('Maintenance Time (min)', fontsize=10)
        ax.set_title(f'{title} Contribution', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(title, fontsize=9)
    
    plt.suptitle('Service Component Breakdown (Pareto points marked with red diamonds)', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    outfile = os.path.join(outdir, 'pareto_service_breakdown.png')
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    print(f"Saved: {outfile}")
    plt.close()


def plot_service_vs_components(df, pareto_points, outdir):
    """Plot individual service components vs maintenance time."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    components = [
        ('starv_contrib', 'Starvation Penalty', axes[0, 0]),
        ('cong_contrib', 'Congestion Penalty', axes[0, 1]),
        ('dev_contrib', 'Deviation Penalty', axes[1, 0]),
        ('service_total', 'Total Service Penalty', axes[1, 1])
    ]
    
    for col, title, ax in components:
        # All points
        ax.scatter(df['maint_time'], df[col],
                  c='lightgray', s=60, alpha=0.6, label='All solutions')
        
        # Pareto points
        ax.scatter(pareto_points['maint_time'], pareto_points[col],
                  c='red', s=100, marker='D', edgecolors='darkred', linewidth=1.5,
                  label='Pareto-efficient')
        
        ax.set_xlabel('Maintenance Time (minutes)', fontsize=10, fontweight='bold')
        ax.set_ylabel(title, fontsize=10, fontweight='bold')
        ax.set_title(f'{title} vs Maintenance', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('How Maintenance Affects Each Service Metric', 
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    outfile = os.path.join(outdir, 'pareto_maintenance_vs_components.png')
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    print(f"Saved: {outfile}")
    plt.close()


def create_pareto_summary_table(pareto_points, outdir):
    """Create a summary table of Pareto-efficient solutions."""
    summary = pareto_points[[
        'service_total', 'starv_contrib', 'cong_contrib', 'dev_contrib',
        'maint_time', 'maint_contrib', 'model_obj'
    ]].copy()
    
    summary = summary.sort_values('service_total')
    summary.index = range(1, len(summary) + 1)
    summary.index.name = 'Pareto_Point'
    
    # Round for readability
    summary = summary.round(3)
    
    # Save to CSV
    outfile = os.path.join(outdir, 'pareto_efficient_solutions.csv')
    summary.to_csv(outfile)
    print(f"\nSaved: {outfile}")
    
    # Print to console
    print("\n" + "="*80)
    print("PARETO-EFFICIENT SOLUTIONS")
    print("="*80)
    print(summary.to_string())
    print("="*80)
    
    return summary


def analyze_tradeoff_rate(pareto_points):
    """Analyze the rate of tradeoff between service and maintenance."""
    if len(pareto_points) < 2:
        print("\nNot enough Pareto points to analyze tradeoff rate")
        return
    
    print("\n" + "="*80)
    print("TRADEOFF ANALYSIS")
    print("="*80)
    
    pareto_sorted = pareto_points.sort_values('service_total')
    
    # Calculate marginal rates of substitution
    service_diff = pareto_sorted['service_total'].diff()
    maint_diff = pareto_sorted['maint_time'].diff()
    
    # Tradeoff rate: how much service penalty increases per minute of maintenance gained
    tradeoff_rate = service_diff / maint_diff
    
    print("\nMarginal Tradeoff Rates:")
    print("(How much service penalty increases per additional minute of maintenance)")
    print("-" * 80)
    
    for idx, (rate, serv, maint) in enumerate(zip(tradeoff_rate.dropna(), 
                                                    service_diff.dropna(), 
                                                    maint_diff.dropna())):
        print(f"  Step {idx+1}: +{maint:.2f} min maintenance → +{serv:.3f} service penalty")
        print(f"           Rate: {rate:.4f} penalty per minute")
        print()
    
    avg_rate = tradeoff_rate.dropna().mean()
    print(f"Average tradeoff rate: {avg_rate:.4f} penalty per minute of maintenance")
    print("="*80)


def provide_recommendations(df, pareto_points):
    """Provide actionable recommendations based on the analysis."""
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if len(pareto_points) == 0:
        print("\n⚠️  WARNING: No Pareto-efficient points found!")
        print("   This suggests all your solutions are dominated by others.")
        print("   Review your parameter settings and optimization constraints.")
        return
    
    # Find best balance point (using simple heuristics)
    # Normalize both objectives to [0, 1] scale
    service_norm = (pareto_points['service_total'] - df['service_total'].min()) / \
                   (df['service_total'].max() - df['service_total'].min())
    maint_norm = (pareto_points['maint_time'] - df['maint_time'].min()) / \
                 (df['maint_time'].max() - df['maint_time'].min())
    
    # Balance score: minimize service, maximize maintenance
    balance_score = service_norm - maint_norm  # Lower is better (low service, high maint)
    best_balance_idx = balance_score.idxmin()
    best_balance = pareto_points.loc[best_balance_idx]
    
    # Extremes
    best_service = pareto_points.loc[pareto_points['service_total'].idxmin()]
    most_maint = pareto_points.loc[pareto_points['maint_time'].idxmax()]
    
    print("\n1. BALANCED SOLUTION (recommended for most cases):")
    print(f"   Service penalty: {best_balance['service_total']:.2f}")
    print(f"   Maintenance time: {best_balance['maint_time']:.2f} minutes")
    print(f"   Breakdown: Starvation={best_balance['starv_contrib']:.3f}, "
          f"Congestion={best_balance['cong_contrib']:.3f}, "
          f"Deviation={best_balance['dev_contrib']:.3f}")
    
    print("\n2. BEST SERVICE QUALITY (minimize starvation/congestion):")
    print(f"   Service penalty: {best_service['service_total']:.2f}")
    print(f"   Maintenance time: {best_service['maint_time']:.2f} minutes")
    print(f"   → Use this if immediate service quality is priority")
    
    print("\n3. MAXIMUM MAINTENANCE (best long-term bike health):")
    print(f"   Service penalty: {most_maint['service_total']:.2f}")
    print(f"   Maintenance time: {most_maint['maint_time']:.2f} minutes")
    print(f"   → Use this if bike maintenance is priority over short-term service")
    
    # Check if there are dominated solutions far from Pareto front
    non_pareto = df[~df['is_pareto']]
    if len(non_pareto) > 0:
        worst = non_pareto.loc[non_pareto['service_total'].idxmax()]
        print(f"\n⚠️  Note: {len(non_pareto)} solutions are dominated (not on Pareto front)")
        print(f"   Worst solution: Service={worst['service_total']:.2f}, "
              f"Maintenance={worst['maint_time']:.2f} min")
        print(f"   → These parameter settings should be avoided!")
    
    print("\n" + "="*80)


def main():
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    policy_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(policy_dir, 'output')
    
    csv_path = os.path.join(output_dir, 'pareto_breakdowns.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        print("Run the pareto scan script first to generate data.")
        return
    
    print("="*80)
    print("PARETO FRONT ANALYSIS: Maintenance vs Service Level Tradeoff")
    print("="*80)
    
    # Load and process data
    df = load_and_clean_data(csv_path)
    
    # Identify Pareto front
    df, pareto_points = identify_pareto_front(df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_main_tradeoff(df, pareto_points, output_dir)
    plot_service_components_breakdown(df, pareto_points, output_dir)
    plot_service_vs_components(df, pareto_points, output_dir)
    
    # Create summary table
    summary = create_pareto_summary_table(pareto_points, output_dir)
    
    # Analyze tradeoff
    analyze_tradeoff_rate(pareto_points)
    
    # Provide recommendations
    provide_recommendations(df, pareto_points)
    
    print("\n✅ Analysis complete! Check the output folder for visualizations.")


if __name__ == '__main__':
    main()

