"""Plot and summarize operational scenario experiments from run_logs.

Example
-------
python policies/sjovik_sund/plot_operational_scenarios.py \
    --run-dir run_logs/run_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(exist_ok=True)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="notebook")


def _read_many(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def load_scenario_logs(run_dir: Path) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    data = {
        "results": _read_many(sorted(run_dir.rglob("results.csv"))),
        "hourly": _read_many(sorted(run_dir.rglob("hourly_metrics.csv"))),
        "daily": _read_many(sorted(run_dir.rglob("daily_metrics.csv"))),
        "decisions": _read_many(sorted(run_dir.rglob("decisions.csv"))),
    }
    return normalize_logs(data)


def normalize_logs(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    decisions = data["decisions"]
    if not decisions.empty:
        def series(name: str, default=0):
            if name in decisions.columns:
                return decisions[name]
            return pd.Series(default, index=decisions.index)

        for col in [
            "functional_pickups",
            "functional_deliveries",
            "onsite_repairs",
            "depot_pickups",
            "depot_deliveries",
            "load_from_queue",
            "action_duration_min",
            "travel_time_min",
        ]:
            if col in decisions.columns:
                decisions[col] = pd.to_numeric(decisions[col], errors="coerce").fillna(0)
        if "num_rebalancing_actions" not in decisions.columns:
            decisions["num_rebalancing_actions"] = (
                series("functional_pickups") + series("functional_deliveries") > 0
            ).astype(int)
        if "num_maintenance_actions" not in decisions.columns:
            decisions["num_maintenance_actions"] = (
                series("onsite_repairs")
                + series("depot_pickups")
                + series("depot_deliveries")
                + series("load_from_queue")
                > 0
            ).astype(int)
        if "num_depot_trips" not in decisions.columns:
            current = series("current_station_id", "").astype(str)
            next_station = series("next_station_id", "").astype(str)
            decisions["num_depot_trips"] = (next_station.str.startswith("D") & ~current.str.startswith("D")).astype(int)
        if "vehicle_time_min" not in decisions.columns:
            decisions["vehicle_time_min"] = series("action_duration_min") + series("travel_time_min")
        if "action_type" not in decisions.columns:
            decisions["action_type"] = "unknown"
        if "vehicle_active" not in decisions.columns:
            decisions["vehicle_active"] = True

    hourly = data["hourly"]
    if not hourly.empty and "broken_bikes_total" not in hourly.columns:
        hourly["broken_bikes_total"] = pd.NA
    return data


def _scenario_col(df: pd.DataFrame) -> str:
    return "scenario_name" if "scenario_name" in df.columns else "exp_name"


def write_summary_tables(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    results = data["results"]
    decisions = data["decisions"]
    hourly = data["hourly"]

    if not results.empty:
        scenario = _scenario_col(results)
        result_summary = (
            results.groupby(scenario, dropna=False)
            .agg(
                seeds=("seed", "nunique"),
                num_service_vehicles=("num_service_vehicles", "mean"),
                service_level_mean=("service_level", "mean"),
                service_level_std=("service_level", "std"),
                starvations_mean=("starvations", "mean"),
                congestions_mean=("congestions", "mean"),
                total_trips_mean=("total_trips", "mean"),
                runtime_s_mean=("total_runtime_s", "mean"),
            )
            .reset_index()
        )
        result_summary.to_csv(out_dir / "scenario_results_summary.csv", index=False)

    if not decisions.empty:
        scenario = _scenario_col(decisions)
        action_summary = (
            decisions.groupby([scenario, "vehicle_id"], dropna=False)
            .agg(
                decisions=("action_type", "size"),
                rebalancing_actions=("num_rebalancing_actions", "sum"),
                maintenance_actions=("num_maintenance_actions", "sum"),
                depot_trips=("num_depot_trips", "sum"),
                vehicle_time_min=("vehicle_time_min", "sum"),
                active_decisions=("vehicle_active", lambda s: s.astype(str).str.lower().isin(["true", "1", "yes"]).sum()),
            )
            .reset_index()
        )
        action_summary.to_csv(out_dir / "scenario_vehicle_action_summary.csv", index=False)

    if not hourly.empty:
        scenario = _scenario_col(hourly)
        hourly_summary = (
            hourly.groupby([scenario, "seed"], dropna=False)
            .agg(
                avg_broken_bikes=("broken_bikes_total", "mean"),
                max_broken_bikes=("broken_bikes_total", "max"),
                avg_damaged_fraction=(
                    "damaged_fraction_onsite",
                    lambda s: s.mean(),
                ),
                total_breakdowns=("total_breakdowns", "sum"),
            )
            .reset_index()
        )
        hourly_summary.to_csv(out_dir / "scenario_backlog_summary.csv", index=False)


def make_plots(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    results = data["results"]
    decisions = data["decisions"]
    hourly = data["hourly"]

    if not results.empty:
        scenario = _scenario_col(results)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=results, x=scenario, y="service_level", errorbar="se", ax=ax)
        ax.set_title("Service Level by Scenario")
        ax.set_xlabel("")
        ax.set_ylabel("Service level")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(out_dir / "service_level_by_scenario.png", dpi=300)
        plt.close(fig)

    if not decisions.empty:
        scenario = _scenario_col(decisions)
        action_counts = (
            decisions.groupby(scenario, dropna=False)
            .agg(
                rebalancing=("num_rebalancing_actions", "sum"),
                maintenance=("num_maintenance_actions", "sum"),
                depot_trips=("num_depot_trips", "sum"),
            )
            .reset_index()
            .melt(id_vars=scenario, var_name="action_group", value_name="count")
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=action_counts, x=scenario, y="count", hue="action_group", ax=ax)
        ax.set_title("Operational Actions by Scenario")
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(out_dir / "actions_by_scenario.png", dpi=300)
        plt.close(fig)

        vehicle_util = (
            decisions.groupby([scenario, "vehicle_id"], dropna=False)["vehicle_time_min"]
            .sum()
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=vehicle_util, x=scenario, y="vehicle_time_min", hue="vehicle_id", ax=ax)
        ax.set_title("Vehicle Utilization Proxy by Scenario")
        ax.set_xlabel("")
        ax.set_ylabel("Operation + travel minutes")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(out_dir / "vehicle_utilization_by_scenario.png", dpi=300)
        plt.close(fig)

    if not hourly.empty:
        scenario = _scenario_col(hourly)
        if "absolute_hour" not in hourly.columns:
            hourly["absolute_hour"] = hourly["day"] * 24 + hourly["hour"]
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.lineplot(data=hourly, x="absolute_hour", y="broken_bikes_total", hue=scenario, errorbar="se", ax=ax)
        ax.set_title("Broken Bikes Over Time")
        ax.set_xlabel("Simulation hour")
        ax.set_ylabel("Broken bikes")
        fig.tight_layout()
        fig.savefig(out_dir / "broken_bikes_over_time.png", dpi=300)
        plt.close(fig)

    if not results.empty and not hourly.empty:
        scenario_r = _scenario_col(results)
        scenario_h = _scenario_col(hourly)
        backlog = (
            hourly.groupby([scenario_h, "seed"], dropna=False)["broken_bikes_total"]
            .mean()
            .reset_index(name="avg_broken_bikes")
        )
        tradeoff = results.merge(
            backlog,
            left_on=[scenario_r, "seed"],
            right_on=[scenario_h, "seed"],
            how="inner",
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=tradeoff, x="avg_broken_bikes", y="service_level", hue=scenario_r, s=70, ax=ax)
        ax.set_title("Service Level Versus Maintenance Backlog")
        ax.set_xlabel("Average broken bikes")
        ax.set_ylabel("Service level")
        fig.tight_layout()
        fig.savefig(out_dir / "service_vs_backlog.png", dpi=300)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot operational scenario run_logs.")
    parser.add_argument("--run-dir", required=True, type=Path, help="RunLogger directory containing scenario folders.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for tables and plots.")
    args = parser.parse_args()

    out_dir = args.out_dir or args.run_dir / "scenario_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_scenario_logs(args.run_dir)
    for name, df in data.items():
        print(f"{name:9s}: {len(df):,} rows")

    write_summary_tables(data, out_dir)
    make_plots(data, out_dir)
    print(f"Wrote scenario analysis to {out_dir}")


if __name__ == "__main__":
    main()
