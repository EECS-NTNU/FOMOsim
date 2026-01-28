#!/usr/bin/env python3
"""
Load `subproblem_objective_breakdown_*.json` files and compute/plot a Pareto front
showing the tradeoff between maintenance and service-level penalties.

Generates:
- `pareto_service_vs_maintenance.png` (service total vs maintenance time)
- `pareto_service_components.png` (scatter of individual service terms)

Usage: run from repository root with the project's venv activated.
    & .\.venv\Scripts\python.exe .\scripts\pareto_from_jsons.py

"""
import glob
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_breakdowns(path_pattern='subproblem_objective_breakdown_*.json'):
    rows = []
    for fn in sorted(glob.glob(path_pattern)):
        try:
            with open(fn, 'r') as fh:
                j = json.load(fh)
            ts = j.get('timestamp')
            mo = j.get('model_obj')
            terms = j.get('terms', {})
            starv = terms.get('starvation', {}).get('contribution', None)
            cong  = terms.get('congestion', {}).get('contribution', None)
            dev   = terms.get('deviation', {}).get('contribution', None)
            maint_contrib = terms.get('maintenance_time', {}).get('contribution', None)
            maint_time = terms.get('maintenance_time', {}).get('sum', None)
            rows.append({
                'file': os.path.basename(fn),
                'timestamp': ts,
                'model_obj': mo,
                'starv_contrib': starv,
                'cong_contrib': cong,
                'dev_contrib': dev,
                'maint_contrib': maint_contrib,
                'maint_time': maint_time
            })
        except Exception as e:
            print('Failed to read', fn, e)
    return pd.DataFrame(rows)


def is_pareto_min(points):
    """
    Find Pareto-efficient points (minimization).
    A point is Pareto-efficient if no other point is better in all objectives.
    """
    N = points.shape[0]
    is_efficient = np.ones(N, dtype=bool)
    
    for i in range(N):
        if not is_efficient[i]:
            continue
        
        # Point i is dominated if there exists another point that is:
        # - Better (strictly less) in at least one objective, AND
        # - Not worse (<=) in all other objectives
        # 
        # We check if point i dominates any other points
        # A point j is dominated by i if: points[i] <= points[j] element-wise AND
        # points[i] < points[j] in at least one dimension
        
        # Mark all points dominated by point i as inefficient
        dominated = np.all(points[is_efficient] >= points[i], axis=1) & \
                   np.any(points[is_efficient] > points[i], axis=1)
        
        # Update efficiency mask
        efficiency_indices = np.where(is_efficient)[0]
        efficiency_indices_dominated = efficiency_indices[dominated]
        is_efficient[efficiency_indices_dominated] = False
    
    return is_efficient


def plot_service_vs_maintenance(df, outfn='pareto_service_vs_maintenance.png', outdir=None):
    # Build service total (lower better) and use maintenance time (higher better)
    df = df.dropna(subset=['starv_contrib', 'cong_contrib', 'dev_contrib', 'maint_time'])
    df['service_total'] = df['starv_contrib'] + df['cong_contrib'] + df['dev_contrib']
    # For Pareto minimization we want to minimize service_total and minimize (-maint_time)
    pts = np.vstack([df['service_total'].to_numpy(), (-df['maint_time'].to_numpy())]).T
    pareto_mask = is_pareto_min(pts)
    df['pareto'] = pareto_mask

    plt.figure(figsize=(7,6))
    plt.scatter(df['service_total'], df['maint_time'], c='gray', s=30, label='all points')
    pareto = df[df['pareto']].sort_values('service_total')
    plt.scatter(pareto['service_total'], pareto['maint_time'], c='red', s=60, label='Pareto front')
    plt.plot(pareto['service_total'], pareto['maint_time'], '-r', alpha=0.6)
    plt.xlabel('Service total (weighted contributions, lower better)')
    plt.ylabel('Maintenance time (minutes, higher better)')
    plt.title('Tradeoff: Service level vs Maintenance')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        outfn = os.path.join(outdir, outfn)
    plt.savefig(outfn, dpi=200)
    print('Saved', outfn)


def plot_service_components(df, outfn='pareto_service_components.png', outdir=None):
    df = df.dropna(subset=['starv_contrib', 'cong_contrib', 'dev_contrib'])
    plt.figure(figsize=(8,5))
    plt.scatter(df['starv_contrib'], df['cong_contrib'], c='gray', s=30, label='points')
    plt.xlabel('Starvation contribution')
    plt.ylabel('Congestion contribution')
    plt.title('Service components scatter (color = deviation)')
    sc = plt.scatter(df['starv_contrib'], df['cong_contrib'], c=df['dev_contrib'], cmap='viridis', s=40)
    plt.colorbar(sc, label='Deviation contribution')
    plt.grid(True)
    plt.tight_layout()
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        outfn = os.path.join(outdir, outfn)
    plt.savefig(outfn, dpi=200)
    print('Saved', outfn)


def main(outdir=None):
    df = load_breakdowns()
    if df.empty:
        print('No breakdown JSON files found in current directory.')
        return
    print('Loaded', len(df), 'breakdown files')
    # Save a CSV copy of parsed breakdowns for convenience
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        csvfn = os.path.join(outdir, 'pareto_breakdowns.csv')
    else:
        csvfn = 'pareto_breakdowns.csv'
    df.to_csv(csvfn, index=False)
    print('Wrote parsed breakdowns to', csvfn)
    plot_service_vs_maintenance(df, outdir=outdir)
    plot_service_components(df, outdir=outdir)


if __name__ == '__main__':
    # Default output folder inside the sjovik_sund policy output folder
    default_out = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    outdir = default_out
    main(outdir=outdir)
