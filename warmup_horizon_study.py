"""
Warmup horizon study: run GreedyMaintenancePolicy over increasing horizons
to measure broken bike accumulation. Helps determine optimal warmup period.

Usage:
    python warmup_horizon_study.py
    python warmup_horizon_study.py --seeds 1000 1001 1002 --instance TD_W34_old
"""
import os
import sys
import csv
import argparse
from pathlib import Path
from datetime import datetime

WORKSPACE_ROOT = Path(__file__).parent
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, "")

from policies.sjovik_sund.run_simulation_ingvild import run_simulation, SimulationConfig
from policies.greedy_policy_maintenance import GreedyMaintenancePolicy
from sim.bike_degradation_modeling.steady_state_odometer import iter_unique_bikes

# Horizons in hours: 2d, 1w, 2w, 4w, 8w
HORIZONS_HOURS = [48, 168, 336, 672, 1344]
HORIZON_LABELS = ["2d", "1w", "2w", "4w", "8w"]


def count_broken_bikes(state):
    """Count bikes by damage status at end of simulation."""
    total = onsite = depot = healthy = 0
    for bike in iter_unique_bikes(state):
        total += 1
        status = getattr(bike, "damage_status", None)
        if status == "onsite":
            onsite += 1
        elif status == "depot":
            depot += 1
        else:
            healthy += 1
    broken = onsite + depot
    return {
        "total": total,
        "healthy": healthy,
        "broken": broken,
        "onsite": onsite,
        "depot": depot,
        "broken_frac": broken / max(total, 1),
    }


def run_horizon_study(seeds, instance_name, num_vehicles=1):
    config = SimulationConfig()
    results = []

    for horizon_h, label in zip(HORIZONS_HOURS, HORIZON_LABELS):
        seed_results = []
        for seed in seeds:
            print(f"  horizon={label} ({horizon_h}h) | seed={seed} ...", flush=True)
            simulator = run_simulation(
                seed=seed,
                policy=GreedyMaintenancePolicy(),
                duration=horizon_h,
                num_vehicles=num_vehicles,
                instance_name=instance_name,
                config=config,
                run_logger=None,
                verbose=False,
                odometer_stats_path="none",  # fresh bikes, no warm-start
            )
            counts = count_broken_bikes(simulator.state)
            seed_results.append(counts)
            print(
                f"    broken={counts['broken']}/{counts['total']} "
                f"({100*counts['broken_frac']:.1f}%)  "
                f"onsite={counts['onsite']}  depot={counts['depot']}"
            )

        # Average over seeds
        n = len(seed_results)
        avg = {k: sum(r[k] for r in seed_results) / n for k in seed_results[0]}
        results.append({
            "horizon_label": label,
            "horizon_hours": horizon_h,
            "seeds": len(seeds),
            **{f"avg_{k}": round(v, 4) for k, v in avg.items()},
        })

    return results


def print_table(results):
    print(f"\n{'Horizon':<8} {'Hours':>6} {'Broken':>8} {'Frac':>8} {'Onsite':>8} {'Depot':>8}")
    print("-" * 50)
    for r in results:
        print(
            f"{r['horizon_label']:<8} {r['horizon_hours']:>6} "
            f"{r['avg_broken']:>8.1f} {r['avg_broken_frac']:>8.3f} "
            f"{r['avg_onsite']:>8.1f} {r['avg_depot']:>8.1f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000])
    parser.add_argument("--instance", type=str, default="TD_W34_old")
    parser.add_argument("--vehicles", type=int, default=1)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    print(f"\nWarmup horizon study | instance={args.instance} | seeds={args.seeds}")
    print(f"Horizons: {HORIZON_LABELS}\n")

    results = run_horizon_study(args.seeds, args.instance, args.vehicles)
    print_table(results)

    # Save CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else Path(f"warmup_horizon_study_{timestamp}.csv")
    fieldnames = list(results[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {out_path}")
