"""
Quick Results Viewer for Pareto Experiment

Displays a quick text summary of results without generating plots.
Useful for checking progress or results without waiting for plot generation.
"""

import pandas as pd
from pathlib import Path
import sys


def view_summary(filename='pareto_experiment_results.csv'):
    """Display quick summary of experiment results."""
    
    results_dir = Path('./policies/sjovik_sund/simulation_results/')
    filepath = results_dir / filename
    
    if not filepath.exists():
        print(f"ERROR: Results file not found: {filepath}")
        print("Please run run_pareto_experiment.py first!")
        return
    
    df = pd.read_csv(filepath)
    
    print("\n" + "="*80)
    print("PARETO EXPERIMENT RESULTS SUMMARY")
    print("="*80 + "\n")
    
    print(f"Results file: {filepath}")
    print(f"Total experiments: {len(df)}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(f"r_M values tested: {sorted(df['r_M'].unique())}")
    print(f"Duration: {df['duration_hours'].iloc[0]} hours")
    print()
    
    # Aggregate by r_M
    summary = df.groupby('r_M').agg({
        'total_service_cost': ['mean', 'std'],
        'starvations': 'mean',
        'long_congestions': 'mean',
        'short_congestions': 'mean',
        'avg_criticality': ['mean', 'std'],
        'pct_critical_07': ['mean', 'std'],
        'bike_pickups': 'mean',
        'bike_deliveries': 'mean',
        'service_level': 'mean',
    })
    
    print("="*80)
    print("RESULTS BY r_M (averaged across seeds)")
    print("="*80)
    print()
    
    # Display in a readable format
    print(f"{'r_M':<8} {'Service':<12} {'Criticality':<15} {'%Crit>0.7':<12} {'Pickups':<10} {'Deliveries':<10}")
    print(f"{'':8} {'Cost':<12} {'(avg)':<15} {'':12} {'':10} {'':10}")
    print("-"*80)
    
    for r_M in sorted(df['r_M'].unique()):
        row = summary.loc[r_M]
        service_cost = row[('total_service_cost', 'mean')]
        criticality = row[('avg_criticality', 'mean')]
        pct_critical = row[('pct_critical_07', 'mean')]
        pickups = row[('bike_pickups', 'mean')]
        deliveries = row[('bike_deliveries', 'mean')]
        
        print(f"{r_M:<8.3f} {service_cost:<12.1f} {criticality:<15.3f} {pct_critical:<12.1f} {pickups:<10.0f} {deliveries:<10.0f}")
    
    print()
    print("="*80)
    print("KEY FINDINGS")
    print("="*80)
    print()
    
    # Find best and worst
    avg_by_r_M = df.groupby('r_M').agg({
        'total_service_cost': 'mean',
        'avg_criticality': 'mean',
        'pct_critical_07': 'mean',
    })
    
    # Best service (lowest cost)
    best_service_r_M = avg_by_r_M['total_service_cost'].idxmin()
    best_service_cost = avg_by_r_M.loc[best_service_r_M, 'total_service_cost']
    
    # Best maintenance (lowest criticality)
    best_maint_r_M = avg_by_r_M['avg_criticality'].idxmin()
    best_maint_crit = avg_by_r_M.loc[best_maint_r_M, 'avg_criticality']
    
    # No maintenance baseline
    baseline_r_M = df['r_M'].min()
    baseline_service = avg_by_r_M.loc[baseline_r_M, 'total_service_cost']
    baseline_crit = avg_by_r_M.loc[baseline_r_M, 'avg_criticality']
    
    print(f"Baseline (r_M={baseline_r_M:.3f}):")
    print(f"  Service Cost: {baseline_service:.1f}")
    print(f"  Avg Criticality: {baseline_crit:.3f}")
    print()
    
    print(f"Best Service Quality (r_M={best_service_r_M:.3f}):")
    print(f"  Service Cost: {best_service_cost:.1f} ({100*(best_service_cost-baseline_service)/max(baseline_service,1):.1f}% vs baseline)")
    print()
    
    print(f"Best Maintenance (r_M={best_maint_r_M:.3f}):")
    print(f"  Avg Criticality: {best_maint_crit:.3f} ({100*(best_maint_crit-baseline_crit)/max(baseline_crit,0.01):.1f}% vs baseline)")
    print()
    
    # Tradeoff analysis
    print("Tradeoff Analysis:")
    service_range = avg_by_r_M['total_service_cost'].max() - avg_by_r_M['total_service_cost'].min()
    crit_range = avg_by_r_M['avg_criticality'].max() - avg_by_r_M['avg_criticality'].min()
    
    if service_range > 0 and crit_range > 0:
        print(f"  Service Cost varies by: {service_range:.1f} ({100*service_range/avg_by_r_M['total_service_cost'].mean():.1f}% of mean)")
        print(f"  Criticality varies by: {crit_range:.3f} ({100*crit_range/avg_by_r_M['avg_criticality'].mean():.1f}% of mean)")
    
    # Find knee point (simple version - minimum normalized distance)
    service_norm = (avg_by_r_M['total_service_cost'] - avg_by_r_M['total_service_cost'].min()) / \
                   max((avg_by_r_M['total_service_cost'].max() - avg_by_r_M['total_service_cost'].min()), 1)
    crit_norm = (avg_by_r_M['avg_criticality'] - avg_by_r_M['avg_criticality'].min()) / \
                max((avg_by_r_M['avg_criticality'].max() - avg_by_r_M['avg_criticality'].min()), 0.01)
    
    distance = (service_norm**2 + crit_norm**2)**0.5
    knee_r_M = distance.idxmin()
    
    print()
    print(f"Recommended r_M (knee point): {knee_r_M:.3f}")
    print(f"  Service Cost: {avg_by_r_M.loc[knee_r_M, 'total_service_cost']:.1f}")
    print(f"  Avg Criticality: {avg_by_r_M.loc[knee_r_M, 'avg_criticality']:.3f}")
    print(f"  % Critical (>0.7): {avg_by_r_M.loc[knee_r_M, 'pct_critical_07']:.1f}%")
    
    print()
    print("="*80)
    print("\nNext step: Run 'python analyze_pareto_results.py' to generate visualizations")
    print()


if __name__ == "__main__":
    # Check if user specified a different filename
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'pareto_experiment_results.csv'
    
    view_summary(filename)

