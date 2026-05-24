#!/usr/bin/env python3
"""
Create station demand summaries and heat maps for selected FOMO instances.

Default:
    python policies/sjovik_sund/demand_heatmap.py

The script reads demand directly from the instance files. That is the cleanest
source for comparing cities because maintenance, degradation, and routing
policies only affect whether demand can be served, not where requested trips
are generated.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "fomosim_matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/private/tmp") / "fomosim_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))
DEFAULT_INSTANCES = ("TD_W34", "BG_W35", "OS_W31")
DEFAULT_OUTPUT_DIR = Path("policies/sjovik_sund/output/demand_heatmaps")
PRESSURE_CMAP = LinearSegmentedColormap.from_list(
    "station_pressure",
    [
        (0.00, "#03045e"),
        (0.18, "#0833b8"),
        (0.34, "#1f8dff"),
        (0.48, "#35f2a7"),
        (0.66, "#f7ff4a"),
        (0.82, "#ff9d2e"),
        (1.00, "#e31a1c"),
    ],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise and plot station-level demand for FOMO instances."
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        default=list(DEFAULT_INSTANCES),
        help="Instance names without .json.gz. Default: TD_W34 BG_W35 OS_W31",
    )
    parser.add_argument(
        "--instances-dir",
        type=Path,
        default=Path("instances"),
        help="Directory containing instance .json.gz and .png files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV and PNG outputs are written.",
    )
    parser.add_argument(
        "--start-day",
        type=int,
        default=0,
        help="Start day in the instance demand matrix, Monday-like index 0-6.",
    )
    parser.add_argument(
        "--start-hour",
        type=int,
        default=0,
        help="Start hour of day, 0-23.",
    )
    parser.add_argument(
        "--duration-hours",
        type=int,
        default=7 * 24,
        help="Number of hours to aggregate. Default: full 7-day demand profile.",
    )
    parser.add_argument(
        "--mode",
        choices=("expected", "simulated"),
        default="expected",
        help=(
            "expected sums hourly intensities; simulated draws seeded Poisson "
            "requests matching GenerateBikeTrips."
        ),
    )
    parser.add_argument(
        "--demand-source",
        choices=("instance", "simulator"),
        default="instance",
        help=(
            "instance reads demand matrices directly; simulator runs FOMOsim "
            "and records realized trip requests."
        ),
    )
    parser.add_argument(
        "--policy",
        choices=("greedy",),
        default="greedy",
        help="Policy used with --demand-source simulator.",
    )
    parser.add_argument(
        "--vehicles",
        type=int,
        default=1,
        help="Number of service vehicles used with --demand-source simulator.",
    )
    parser.add_argument(
        "--target-state",
        choices=("half_capacity", "equal_prob"),
        default="half_capacity",
        help="Target state used with --demand-source simulator.",
    )
    parser.add_argument(
        "--enable-component-degradation",
        action="store_true",
        help=(
            "Keep component degradation enabled in simulator mode. By default "
            "the script disables it for demand-only runs."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Seeds used only in --mode simulated. Multiple seeds are averaged.",
    )
    parser.add_argument(
        "--bike-demand-factor",
        type=float,
        default=2.0,
        help=(
            "Multiplier for departure demand. The simulator currently creates "
            "2 * round(Poisson(lambda)) bike requests each hour, so the default "
            "is 2.0 to match that behavior."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=("departure_demand", "arrival_demand", "endpoint_demand", "net_demand"),
        default="departure_demand",
        help="Station metric used for the heat map color and marker size.",
    )
    parser.add_argument(
        "--scale",
        choices=("linear", "log"),
        default="linear",
        help="Color scale for the heat map.",
    )
    parser.add_argument(
        "--no-map-background",
        action="store_true",
        help="Plot stations on blank longitude/latitude axes instead of instance PNG maps.",
    )
    parser.add_argument(
        "--show-axes",
        action="store_true",
        help="Show longitude/latitude axes, labels, grids, and subplot titles.",
    )
    parser.add_argument(
        "--style",
        choices=("glow", "markers"),
        default="glow",
        help="Use a smoothed glowing pressure field or station markers.",
    )
    parser.add_argument(
        "--color-by-demand",
        dest="color_by_demand",
        action="store_true",
        default=True,
        help="Use a heat gradient for marker color.",
    )
    parser.add_argument(
        "--single-color",
        dest="color_by_demand",
        action="store_false",
        help="Use one solid marker color and show demand by marker size only.",
    )
    parser.add_argument(
        "--show-scale",
        action="store_true",
        help="Show the colorbar or marker-size scale on the generated images.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution.",
    )
    return parser.parse_args()


def load_instance(instances_dir: Path, instance_name: str) -> dict:
    path = instances_dir / f"{instance_name}.json.gz"
    if not path.exists():
        raise FileNotFoundError(f"Could not find instance file: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def selected_hours(start_day: int, start_hour: int, duration_hours: int) -> Iterable[tuple[int, int]]:
    start = (start_day % 7) * 24 + (start_hour % 24)
    for offset in range(duration_hours):
        absolute_hour = start + offset
        yield (absolute_hour // 24) % 7, absolute_hour % 24


def station_id_from_index(station_counter: int) -> str:
    return f"S{station_counter}"


def expected_station_demand(
    station: dict,
    hours: list[tuple[int, int]],
    bike_demand_factor: float,
) -> dict[str, float]:
    departure = 0.0
    arrival = 0.0
    for day, hour in hours:
        departure += bike_demand_factor * float(station["leave_intensities"][day][hour])
        arrival += float(station["arrive_intensities"][day][hour])
    return {
        "departure_demand": departure,
        "arrival_demand": arrival,
        "endpoint_demand": departure + arrival,
        "net_demand": departure - arrival,
    }


def simulate_instance_demand(
    stations: list[dict],
    hours: list[tuple[int, int]],
    seeds: list[int],
    bike_demand_factor: float,
    instance_offset: int,
) -> dict[int, dict[str, float]]:
    demand_by_station = {
        station_idx: {"departure_demand": [], "arrival_demand": []}
        for station_idx, station in enumerate(stations)
        if not station.get("is_depot", False)
    }

    for seed in seeds:
        rng = np.random.default_rng(seed + 20000 + instance_offset)
        seed_departures = {station_idx: 0.0 for station_idx in demand_by_station}
        seed_arrivals = {station_idx: 0.0 for station_idx in demand_by_station}

        for hour_idx, (day, hour) in enumerate(hours):
            start_minute = hour_idx * 60
            for station_idx, station in enumerate(stations):
                if station_idx not in demand_by_station:
                    continue
                leave_lambda = float(station["leave_intensities"][day][hour])
                requests = bike_demand_factor * round(rng.poisson(leave_lambda))
                seed_departures[station_idx] += requests

                # GenerateBikeTrips also draws concrete departure times after
                # the count. Consume those random numbers to keep the stream in
                # the same state as the simulator for following stations.
                rng.integers(start_minute, start_minute + 60, int(requests))

            for station_idx, station in enumerate(stations):
                if station_idx not in demand_by_station:
                    continue
                arrive_lambda = float(station["arrive_intensities"][day][hour])
                arrivals = round(rng.poisson(arrive_lambda))
                seed_arrivals[station_idx] += arrivals
                rng.integers(start_minute, start_minute + 60, int(arrivals))

        for station_idx in demand_by_station:
            demand_by_station[station_idx]["departure_demand"].append(seed_departures[station_idx])
            demand_by_station[station_idx]["arrival_demand"].append(seed_arrivals[station_idx])

    averaged = {}
    for station_idx, samples in demand_by_station.items():
        departure = float(np.mean(samples["departure_demand"]))
        arrival = float(np.mean(samples["arrival_demand"]))
        averaged[station_idx] = {
            "departure_demand": departure,
            "arrival_demand": arrival,
            "endpoint_demand": departure + arrival,
            "net_demand": departure - arrival,
        }
    return averaged


def build_station_summary(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    instances_dir = (WORKSPACE_ROOT / args.instances_dir).resolve()
    hours = list(selected_hours(args.start_day, args.start_hour, args.duration_hours))

    for instance_idx, instance_name in enumerate(args.instances):
        data = load_instance(instances_dir, instance_name)
        simulated_demands = {}
        if args.mode == "simulated":
            simulated_demands = simulate_instance_demand(
                data["stations"],
                hours,
                seeds=args.seeds,
                bike_demand_factor=args.bike_demand_factor,
                instance_offset=instance_idx * 100000,
            )

        station_counter = 0
        for raw_station_idx, raw_station in enumerate(data["stations"]):
            if raw_station.get("is_depot", False):
                continue

            if args.mode == "expected":
                demand = expected_station_demand(
                    raw_station,
                    hours,
                    bike_demand_factor=args.bike_demand_factor,
                )
            else:
                demand = simulated_demands[raw_station_idx]

            lat, lon = raw_station["location"]
            rows.append({
                "instance": instance_name,
                "city": data.get("city", instance_name),
                "station_id": station_id_from_index(station_counter),
                "raw_station_id": raw_station.get("id", station_counter),
                "lat": lat,
                "lon": lon,
                "capacity": raw_station.get("capacity", np.nan),
                "initial_bikes": raw_station.get("num_bikes", np.nan),
                **demand,
            })
            station_counter += 1

    return pd.DataFrame(rows)


def _patch_component_degradation(enabled: bool):
    import importlib

    import settings

    bike_module = importlib.import_module("sim.Bike")
    bike_arrival_module = importlib.import_module("sim.events.BikeArrival")
    bike_departure_module = importlib.import_module("sim.events.BikeDeparture")

    modules = [settings, bike_module, bike_arrival_module, bike_departure_module]
    previous = {
        module: getattr(module, "ENABLE_COMPONENT_FAILURES", None)
        for module in modules
        if hasattr(module, "ENABLE_COMPONENT_FAILURES")
    }
    for module in previous:
        setattr(module, "ENABLE_COMPONENT_FAILURES", enabled)
    return previous


def _restore_component_degradation(previous: dict) -> None:
    for module, value in previous.items():
        setattr(module, "ENABLE_COMPONENT_FAILURES", value)


class DemandRecordingSimulatorMixin:
    def log_trip_request(
        self,
        time,
        station_id,
        success=True,
        failure_reason=None,
        did_roam=False,
        arrival_station_id=None,
        travel_time=None,
        bike_id=None,
        bike_criticality=None,
    ):
        day = int(time // (24 * 60))
        hour = int((time % (24 * 60)) // 60)
        minute = int(time % 60)
        self.trip_requests.append({
            "time_minutes": time,
            "day": day,
            "hour": hour,
            "minute": minute,
            "station_id": station_id,
            "success": bool(success),
            "failure_reason": failure_reason if not success else None,
            "did_roam": bool(did_roam),
            "arrival_station_id": arrival_station_id if success else None,
            "travel_time": travel_time if success else None,
            "bike_id": bike_id if success else None,
        })

    def log_bike_movement(self, *args, **kwargs):
        pass

    def log_component_failure(self, *args, **kwargs):
        pass


def make_recording_simulator_class():
    import sim

    class DemandRecordingSimulator(DemandRecordingSimulatorMixin, sim.Simulator):
        def __init__(self, *args, **kwargs):
            self.trip_requests = []
            super().__init__(*args, **kwargs)

    return DemandRecordingSimulator


def _target_state_from_name(name: str):
    import target_state

    if name == "equal_prob":
        return target_state.EqualProbTargetState()
    return target_state.HalfCapacityTargetState()


def run_simulator_demand_for_instance(
    args: argparse.Namespace,
    instance_name: str,
    instance_idx: int,
) -> pd.DataFrame:
    import demand
    import init_state
    from policies.greedy_policy import GreedyPolicy

    instances_dir = (WORKSPACE_ROOT / args.instances_dir).resolve()
    instance_data = load_instance(instances_dir, instance_name)
    state_template = init_state.read_initial_state(str(instances_dir / instance_name))
    target = _target_state_from_name(args.target_state)
    simulator_cls = make_recording_simulator_class()

    rows = []
    station_meta = {}
    for station in state_template.get_stations():
        station_meta[station.id] = {
            "lat": station.lat,
            "lon": station.lon,
            "capacity": station.capacity,
            "initial_bikes": station.number_of_bikes(),
        }

    previous_degradation = _patch_component_degradation(args.enable_component_degradation)
    try:
        for seed in args.seeds:
            state = init_state.read_initial_state(str(instances_dir / instance_name))
            state.set_seed(seed + instance_idx * 100000)

            policies = [
                GreedyPolicy(service_hours=(0, 24))
                for _ in range(max(args.vehicles, 0))
            ]
            if policies:
                state.set_sb_vehicles(policies)

            for vehicle_idx, vehicle in enumerate(state.get_vehicles()):
                station_ids = sorted(state.get_station_ids(), key=lambda sid: int(sid[1:]))
                vehicle.location = state.locations[station_ids[vehicle_idx % len(station_ids)]]

            simulator = simulator_cls(
                initial_state=state,
                target_state=target,
                demand=demand.Demand(),
                start_time=(args.start_day % 7) * 24 * 60 + (args.start_hour % 24) * 60,
                duration=args.duration_hours * 60,
                cluster=True,
                verbose=False,
            )
            simulator.run()

            if simulator.trip_requests:
                request_df = pd.DataFrame(simulator.trip_requests)
                counts = request_df.groupby("station_id").agg(
                    departure_demand=("station_id", "count"),
                    served_demand=("success", "sum"),
                )
            else:
                counts = pd.DataFrame(columns=["departure_demand", "served_demand"])

            for station_id, meta in station_meta.items():
                departure = float(counts.loc[station_id, "departure_demand"]) if station_id in counts.index else 0.0
                served = float(counts.loc[station_id, "served_demand"]) if station_id in counts.index else 0.0
                rows.append({
                    "instance": instance_name,
                    "city": instance_data.get("city", instance_name),
                    "seed": seed,
                    "station_id": station_id,
                    "raw_station_id": int(station_id[1:]),
                    "lat": meta["lat"],
                    "lon": meta["lon"],
                    "capacity": meta["capacity"],
                    "initial_bikes": meta["initial_bikes"],
                    "departure_demand": departure,
                    "arrival_demand": 0.0,
                    "endpoint_demand": departure,
                    "net_demand": departure,
                    "served_demand": served,
                    "unserved_demand": departure - served,
                })
    finally:
        _restore_component_degradation(previous_degradation)

    seed_df = pd.DataFrame(rows)
    if seed_df.empty:
        return seed_df

    value_cols = [
        "departure_demand",
        "arrival_demand",
        "endpoint_demand",
        "net_demand",
        "served_demand",
        "unserved_demand",
    ]
    index_cols = [
        "instance",
        "city",
        "station_id",
        "raw_station_id",
        "lat",
        "lon",
        "capacity",
        "initial_bikes",
    ]
    return seed_df.groupby(index_cols, as_index=False)[value_cols].mean()


def build_simulator_station_summary(args: argparse.Namespace) -> pd.DataFrame:
    all_rows = []
    for instance_idx, instance_name in enumerate(args.instances):
        print(
            f"Running simulator demand recording for {instance_name} "
            f"({args.policy}, vehicles={args.vehicles}, seeds={args.seeds})"
        )
        all_rows.append(run_simulator_demand_for_instance(args, instance_name, instance_idx))
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def build_city_summary(station_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = station_df.groupby(["instance", "city"], as_index=False)
    summary = grouped.agg(
        stations=("station_id", "count"),
        total_capacity=("capacity", "sum"),
        initial_bikes=("initial_bikes", "sum"),
        departure_demand=("departure_demand", "sum"),
        arrival_demand=("arrival_demand", "sum"),
        endpoint_demand=("endpoint_demand", "sum"),
        net_demand=("net_demand", "sum"),
        mean_station_metric=(metric, "mean"),
        median_station_metric=(metric, "median"),
        max_station_metric=(metric, "max"),
    )
    summary["demand_per_station"] = summary["departure_demand"] / summary["stations"]
    summary["demand_per_capacity"] = summary["departure_demand"] / summary["total_capacity"]
    return summary


def metric_values_for_plot(station_df: pd.DataFrame, metric: str) -> np.ndarray:
    values = station_df[metric].to_numpy(dtype=float)
    if metric == "net_demand":
        return np.abs(values)
    return values


def marker_sizes(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    upper = np.nanpercentile(values, 95)
    if not np.isfinite(upper) or upper <= 0:
        return np.full(values.shape, 35.0)
    scaled = np.clip(values / upper, 0.05, 1.0)
    return 25.0 + 185.0 * np.sqrt(scaled)


def marker_size_for_value(value: float, values: np.ndarray) -> float:
    upper = np.nanpercentile(values, 95)
    if not np.isfinite(upper) or upper <= 0:
        return 35.0
    scaled = np.clip(value / upper, 0.05, 1.0)
    return float(25.0 + 185.0 * np.sqrt(scaled))


def add_size_legend(fig, axes, values: np.ndarray, label: str) -> None:
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return

    legend_values = np.unique(np.percentile(positive, [35, 70, 95]).round(0))
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            label=f"{value:.0f}",
            markerfacecolor="#d9481e",
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=np.sqrt(marker_size_for_value(value, values)),
            alpha=0.78,
        )
        for value in legend_values
    ]
    fig.legend(
        handles=handles,
        title=label,
        loc="center right",
        bbox_to_anchor=(0.985, 0.5),
        frameon=False,
        borderaxespad=0.0,
    )


def set_station_extent(ax, instance_rows: pd.DataFrame) -> None:
    lon_pad = max((instance_rows["lon"].max() - instance_rows["lon"].min()) * 0.08, 0.005)
    lat_pad = max((instance_rows["lat"].max() - instance_rows["lat"].min()) * 0.08, 0.005)
    ax.set_xlim(instance_rows["lon"].min() - lon_pad, instance_rows["lon"].max() + lon_pad)
    ax.set_ylim(instance_rows["lat"].min() - lat_pad, instance_rows["lat"].max() + lat_pad)


def demand_glow_grid(
    instance_rows: pd.DataFrame,
    metric: str,
    extent: tuple[float, float, float, float],
    resolution: int = 360,
) -> np.ndarray:
    west, east, south, north = extent
    xs = np.linspace(west, east, resolution)
    ys = np.linspace(south, north, resolution)
    xx, yy = np.meshgrid(xs, ys)

    width = max(east - west, 1e-9)
    height = max(north - south, 1e-9)
    sigma_x = width * 0.017
    sigma_y = height * 0.017

    values = metric_values_for_plot(instance_rows, metric)
    upper = np.nanpercentile(values[np.isfinite(values)], 96) if values.size else 1.0
    if not np.isfinite(upper) or upper <= 0:
        upper = 1.0
    grid = np.zeros_like(xx, dtype=float)
    for lon, lat, value in zip(instance_rows["lon"], instance_rows["lat"], values):
        if not np.isfinite(value) or value <= 0:
            continue
        # Compress station-to-station variation a little so medium-demand
        # stations still produce visible blue/green halos while peaks get cores.
        weight = np.sqrt(value / upper) * upper
        dx2 = ((xx - lon) / sigma_x) ** 2
        dy2 = ((yy - lat) / sigma_y) ** 2
        grid += weight * np.exp(-0.5 * (dx2 + dy2))

    return grid


def glow_alpha(grid: np.ndarray, global_max: float) -> np.ndarray:
    if global_max <= 0:
        return np.zeros_like(grid)
    intensity = np.clip(grid / global_max, 0.0, 1.0)
    alpha = 0.02 + 0.78 * np.power(intensity, 0.55)
    alpha[intensity <= 0.04] = 0.0
    return alpha


def output_run_label(args: argparse.Namespace) -> str:
    if args.demand_source == "simulator":
        return f"simulator_{args.policy}"
    return args.mode


def get_color_norm(values: np.ndarray, scale: str):
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    if scale == "log":
        positive = finite_values[finite_values > 0]
        if positive.size == 0:
            return Normalize(vmin=0.0, vmax=1.0)
        return LogNorm(vmin=max(float(positive.min()), 1e-6), vmax=float(positive.max()))
    vmin = float(finite_values.min())
    vmax = float(finite_values.max())
    if vmin == vmax:
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def add_map_background(ax, instances_dir: Path, instance_name: str, instance_rows: pd.DataFrame) -> None:
    json_data = load_instance(instances_dir, instance_name)
    map_name = json_data.get("map")
    bbox = json_data.get("map_boundingbox")
    map_path = instances_dir / map_name if map_name else None

    if map_path is not None and map_path.exists() and bbox is not None:
        west, east, south, north = bbox
        img = plt.imread(map_path)
        ax.imshow(img, extent=[west, east, south, north], aspect="auto", zorder=0)
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
    else:
        set_station_extent(ax, instance_rows)


def plot_combined_heatmap(
    station_df: pd.DataFrame,
    city_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Path:
    output_dir = (WORKSPACE_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    instances_dir = (WORKSPACE_ROOT / args.instances_dir).resolve()

    values = metric_values_for_plot(station_df, args.metric)
    norm = get_color_norm(values, args.scale)

    fig, axes = plt.subplots(
        1,
        len(args.instances),
        figsize=(4.1 * len(args.instances), 4.5),
        constrained_layout=False,
        gridspec_kw={"wspace": 0.18},
    )
    if len(args.instances) == 1:
        axes = [axes]

    scatter = None
    glow_grids = []
    glow_payloads = []
    for ax, instance_name in zip(axes, args.instances):
        instance_rows = station_df[station_df["instance"] == instance_name].copy()
        instance_values = metric_values_for_plot(instance_rows, args.metric)
        sizes = marker_sizes(instance_values)

        if not args.no_map_background:
            add_map_background(ax, instances_dir, instance_name, instance_rows)
        else:
            set_station_extent(ax, instance_rows)
        ax.set_box_aspect(1)

        if args.style == "glow":
            west, east = ax.get_xlim()
            south, north = ax.get_ylim()
            extent = (west, east, south, north)
            grid = demand_glow_grid(instance_rows, args.metric, extent)
            glow_grids.append(grid)
            glow_payloads.append((ax, grid, extent))
        else:
            scatter_kwargs = {
                "x": instance_rows["lon"],
                "y": instance_rows["lat"],
                "s": sizes,
                "alpha": 0.78,
                "edgecolor": "white",
                "linewidth": 0.35,
                "zorder": 2,
            }
            if args.color_by_demand:
                scatter_kwargs.update({"c": instance_values, "cmap": PRESSURE_CMAP, "norm": norm})
            else:
                scatter_kwargs.update({"color": "#d9481e"})
            scatter = ax.scatter(**scatter_kwargs)

        city_row = city_df[city_df["instance"] == instance_name].iloc[0]
        if args.show_axes:
            ax.set_title(
                f"{city_row['city']} ({instance_name})\n"
                f"{int(city_row['stations'])} stations, "
                f"{city_row['departure_demand']:.0f} departures",
                fontsize=11,
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.tick_params(labelsize=8)
            ax.grid(color="white", alpha=0.25, linewidth=0.5)
        else:
            ax.set_axis_off()

    label = args.metric.replace("_", " ")
    if args.metric == "net_demand":
        label = "absolute net demand"

    if args.style == "glow" and args.show_scale:
        global_glow_max = max((float(np.nanmax(grid)) for grid in glow_grids), default=0.0)
        glow_norm = Normalize(vmin=0.0, vmax=global_glow_max if global_glow_max > 0 else 1.0)
        for ax, grid, extent in glow_payloads:
            ax.imshow(
                grid,
                extent=extent,
                origin="lower",
                aspect="auto",
                cmap=PRESSURE_CMAP,
                norm=glow_norm,
                alpha=glow_alpha(grid, global_glow_max),
                interpolation="bilinear",
                zorder=2,
            )
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
        mappable = ScalarMappable(norm=glow_norm, cmap=PRESSURE_CMAP)
        mappable.set_array([])
        cbar_ax = fig.add_axes([0.24, 0.91, 0.52, 0.035])
        cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(f"{label} pressure", labelpad=4)
        cbar.ax.xaxis.set_label_position("top")
        cbar.ax.xaxis.set_ticks_position("bottom")
        cbar.ax.tick_params(labelsize=8)
    elif args.style == "glow":
        global_glow_max = max((float(np.nanmax(grid)) for grid in glow_grids), default=0.0)
        glow_norm = Normalize(vmin=0.0, vmax=global_glow_max if global_glow_max > 0 else 1.0)
        for ax, grid, extent in glow_payloads:
            ax.imshow(
                grid,
                extent=extent,
                origin="lower",
                aspect="auto",
                cmap=PRESSURE_CMAP,
                norm=glow_norm,
                alpha=glow_alpha(grid, global_glow_max),
                interpolation="bilinear",
                zorder=2,
            )
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
    elif args.color_by_demand and args.show_scale:
        assert scatter is not None
        cbar_ax = fig.add_axes([0.24, 0.91, 0.52, 0.035])
        cbar = fig.colorbar(scatter, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(label, labelpad=4)
        cbar.ax.xaxis.set_label_position("top")
        cbar.ax.xaxis.set_ticks_position("bottom")
    elif args.show_scale:
        add_size_legend(fig, axes, values, label)
    if args.show_axes:
        fig.suptitle(
            f"Station demand heat map ({output_run_label(args)}, {args.duration_hours} h)",
            fontsize=14,
        )
        fig.subplots_adjust(left=0.06, right=0.94, bottom=0.12, top=0.78, wspace=0.28)
    else:
        top = 0.83 if args.show_scale else 0.96
        fig.subplots_adjust(left=0.025, right=0.975, bottom=0.055, top=top, wspace=0.26)

    output_path = output_dir / f"combined_{args.metric}_{output_run_label(args)}_{args.scale}.png"
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)
    return output_path


def plot_individual_heatmaps(
    station_df: pd.DataFrame,
    args: argparse.Namespace,
) -> list[Path]:
    output_dir = (WORKSPACE_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    instances_dir = (WORKSPACE_ROOT / args.instances_dir).resolve()

    output_paths = []
    values = metric_values_for_plot(station_df, args.metric)
    norm = get_color_norm(values, args.scale)

    for instance_name in args.instances:
        instance_rows = station_df[station_df["instance"] == instance_name].copy()
        instance_values = metric_values_for_plot(instance_rows, args.metric)

        fig, ax = plt.subplots(figsize=(7.0, 7.0), constrained_layout=True)
        if not args.no_map_background:
            add_map_background(ax, instances_dir, instance_name, instance_rows)
        else:
            set_station_extent(ax, instance_rows)
        ax.set_box_aspect(1)

        scatter = None
        grid = None
        extent = None
        if args.style == "glow":
            west, east = ax.get_xlim()
            south, north = ax.get_ylim()
            extent = (west, east, south, north)
            grid = demand_glow_grid(instance_rows, args.metric, extent)
            glow_max = float(np.nanmax(grid)) if grid.size else 0.0
            glow_norm = Normalize(vmin=0.0, vmax=glow_max if glow_max > 0 else 1.0)
            ax.imshow(
                grid,
                extent=extent,
                origin="lower",
                aspect="auto",
                cmap=PRESSURE_CMAP,
                norm=glow_norm,
                alpha=glow_alpha(grid, glow_max),
                interpolation="bilinear",
                zorder=2,
            )
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
        else:
            scatter_kwargs = {
                "x": instance_rows["lon"],
                "y": instance_rows["lat"],
                "s": marker_sizes(instance_values),
                "alpha": 0.78,
                "edgecolor": "white",
                "linewidth": 0.35,
                "zorder": 2,
            }
            if args.color_by_demand:
                scatter_kwargs.update({"c": instance_values, "cmap": PRESSURE_CMAP, "norm": norm})
            else:
                scatter_kwargs.update({"color": "#d9481e"})
            scatter = ax.scatter(**scatter_kwargs)

        if args.show_axes:
            city = instance_rows["city"].iloc[0]
            ax.set_title(f"{city} ({instance_name}) station demand", fontsize=13)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(color="white", alpha=0.25, linewidth=0.5)
        else:
            ax.set_axis_off()

        if args.style == "glow" and args.show_scale:
            mappable = ScalarMappable(norm=glow_norm, cmap=PRESSURE_CMAP)
            mappable.set_array([])
            cbar_ax = fig.add_axes([0.24, 0.91, 0.52, 0.035])
            cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
            cbar.set_label(f"{args.metric.replace('_', ' ')} pressure", labelpad=4)
            cbar.ax.xaxis.set_label_position("top")
            cbar.ax.xaxis.set_ticks_position("bottom")
            cbar.ax.tick_params(labelsize=8)
        elif args.color_by_demand and args.show_scale:
            cbar_ax = fig.add_axes([0.24, 0.91, 0.52, 0.035])
            cbar = fig.colorbar(scatter, cax=cbar_ax, orientation="horizontal")
            cbar.set_label(args.metric.replace("_", " "), labelpad=4)
            cbar.ax.xaxis.set_label_position("top")
            cbar.ax.xaxis.set_ticks_position("bottom")
        elif args.show_scale:
            add_size_legend(fig, [ax], instance_values, args.metric.replace("_", " "))

        output_path = output_dir / f"{instance_name}_{args.metric}_{output_run_label(args)}_{args.scale}.png"
        fig.savefig(output_path, dpi=args.dpi)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def main() -> int:
    args = parse_args()
    output_dir = (WORKSPACE_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.demand_source == "simulator":
        station_df = build_simulator_station_summary(args)
    else:
        station_df = build_station_summary(args)
    output_suffix = output_run_label(args)

    city_df = build_city_summary(station_df, args.metric)

    station_csv = output_dir / f"station_demand_{output_suffix}.csv"
    city_csv = output_dir / f"city_demand_{output_suffix}.csv"
    station_df.to_csv(station_csv, index=False)
    city_df.to_csv(city_csv, index=False)

    combined_png = plot_combined_heatmap(station_df, city_df, args)
    individual_pngs = plot_individual_heatmaps(station_df, args)

    print("Demand heat map outputs written:")
    print(f"  Station CSV : {station_csv}")
    print(f"  City CSV    : {city_csv}")
    print(f"  Combined PNG: {combined_png}")
    for path in individual_pngs:
        print(f"  City PNG    : {path}")
    print()
    print(city_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
