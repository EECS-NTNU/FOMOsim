"""Utilities for steady-state component odometer initialization."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def iter_unique_bikes(state) -> Iterable[Any]:
    """Yield every bike once, including bikes in repair queues."""
    seen = set()

    def maybe_yield(bike):
        bike_id = getattr(bike, "bike_id", None)
        if bike_id is None or bike_id in seen:
            return None
        seen.add(bike_id)
        return bike

    for bike in getattr(state, "get_all_bikes", lambda: [])():
        yielded = maybe_yield(bike)
        if yielded is not None:
            yield yielded

    for depot in getattr(state, "get_depots", lambda: [])():
        for bike in getattr(depot, "fixed_queue", {}).values():
            yielded = maybe_yield(bike)
            if yielded is not None:
                yield yielded
        for _, bikes in getattr(depot, "in_repair", []):
            for bike in bikes:
                yielded = maybe_yield(bike)
                if yielded is not None:
                    yield yielded

    for station in getattr(state, "get_stations", lambda: [])():
        for queue_item in getattr(station, "onsite_repair_queue", []):
            if len(queue_item) >= 2:
                yielded = maybe_yield(queue_item[1])
                if yielded is not None:
                    yield yielded


def collect_component_odometer_samples(state, seed: int | None = None) -> list[dict[str, Any]]:
    """Return one row per bike/component odometer in the current state."""
    rows = []
    for bike in iter_unique_bikes(state):
        component_odometers = getattr(bike, "component_odometers", {}) or {}
        for component_type, odometer in component_odometers.items():
            rows.append(
                {
                    "seed": seed,
                    "bike_id": getattr(bike, "bike_id", None),
                    "component_type": component_type,
                    "odometer_km": float(odometer),
                    "total_distance_km": float(getattr(bike, "total_distance_km", 0.0) or 0.0),
                    "damage_status": getattr(bike, "damage_status", None),
                    "is_available": bool(getattr(bike, "is_available", True)),
                }
            )
    return rows


def summarize_component_odometers(state, seed: int | None = None) -> list[dict[str, Any]]:
    """Summarize final-state component odometers by component category."""
    by_component = defaultdict(list)
    for row in collect_component_odometer_samples(state, seed=seed):
        by_component[row["component_type"]].append(row["odometer_km"])

    summaries = []
    for component_type, values in sorted(by_component.items()):
        arr = np.asarray(values, dtype=float)
        summaries.append(
            {
                "component_type": component_type,
                "seed": seed,
                "count": int(arr.size),
                "min_odometer": float(np.min(arr)),
                "mean_odometer": float(np.mean(arr)),
                "max_odometer": float(np.max(arr)),
                "std_odometer": float(np.std(arr, ddof=0)),
                "p05_odometer": float(np.percentile(arr, 5)),
                "p50_odometer": float(np.percentile(arr, 50)),
                "p95_odometer": float(np.percentile(arr, 95)),
            }
        )
    return summaries


def aggregate_seed_summaries(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average per-seed component summaries into one initialization table."""
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[row["component_type"]].append(row)

    aggregated = []
    for component_type, rows in sorted(grouped.items()):
        total_count = sum(int(row.get("count", 0) or 0) for row in rows)
        aggregated.append(
            {
                "component_type": component_type,
                "num_seeds": len(rows),
                "avg_count": total_count / len(rows),
                "avg_min": _mean(rows, "min_odometer"),
                "avg_mean": _mean(rows, "mean_odometer"),
                "avg_max": _mean(rows, "max_odometer"),
                "avg_std": _mean(rows, "std_odometer"),
                "avg_p05": _mean(rows, "p05_odometer"),
                "avg_p50": _mean(rows, "p50_odometer"),
                "avg_p95": _mean(rows, "p95_odometer"),
            }
        )
    return aggregated


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV, creating the parent directory if needed."""
    if not rows:
        raise ValueError(f"No rows to write to {path}")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_component_odometer_stats(path: str | Path) -> dict[str, dict[str, float]]:
    """Load either per-seed or aggregated odometer stats by component type."""
    stats = {}
    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            component_type = row["component_type"]
            stats[component_type] = {
                "min": _first_float(row, "avg_min", "min_odometer"),
                "mean": _first_float(row, "avg_mean", "mean_odometer"),
                "max": _first_float(row, "avg_max", "max_odometer"),
                "std": _first_float(row, "avg_std", "std_odometer", default=0.0),
                "p05": _first_float(row, "avg_p05", "p05_odometer", "avg_min", "min_odometer"),
                "p95": _first_float(row, "avg_p95", "p95_odometer", "avg_max", "max_odometer"),
            }
    return stats


def apply_component_odometer_initialization(
    state,
    stats_path: str | Path,
    rng=None,
    method: str = "triangular",
    bounds: str = "p05-p95",
) -> int:
    """Initialize component odometers from steady-state stats.

    Returns the number of bike objects initialized.
    """
    stats = load_component_odometer_stats(stats_path)
    rng = rng if rng is not None else np.random.default_rng()
    initialized = 0

    for bike in iter_unique_bikes(state):
        component_odometers = getattr(bike, "component_odometers", None)
        if not component_odometers:
            continue

        sampled_values = []
        for component_type in component_odometers:
            if component_type not in stats:
                continue
            value = sample_component_odometer(
                stats[component_type],
                rng=rng,
                method=method,
                bounds=bounds,
            )
            component_odometers[component_type] = value
            sampled_values.append(value)

        if sampled_values:
            bike.total_distance_km = max(
                float(getattr(bike, "total_distance_km", 0.0) or 0.0),
                max(sampled_values),
            )
            initialized += 1

    return initialized


def sample_component_odometer(
    stat: dict[str, float],
    rng,
    method: str = "triangular",
    bounds: str = "p05-p95",
) -> float:
    """Sample one bounded component odometer value."""
    lower_key, upper_key = ("p05", "p95") if bounds == "p05-p95" else ("min", "max")
    lower = max(0.0, float(stat.get(lower_key, stat["min"])))
    upper = max(lower, float(stat.get(upper_key, stat["max"])))
    mean = min(max(float(stat["mean"]), lower), upper)

    if upper == lower:
        return lower

    if method == "uniform":
        return float(rng.uniform(lower, upper))

    if method == "truncated-normal":
        std = float(stat.get("std", 0.0) or 0.0)
        if std <= 0:
            std = max((upper - lower) / 4.0, 1e-9)
        for _ in range(100):
            value = float(rng.normal(mean, std))
            if lower <= value <= upper:
                return value
        return float(np.clip(value, lower, upper))

    if method == "triangular":
        return float(rng.triangular(lower, mean, upper))

    raise ValueError(
        f"Unknown odometer sampling method '{method}'. "
        "Expected one of: triangular, uniform, truncated-normal."
    )


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _first_float(row: dict[str, Any], *keys: str, default: float | None = None) -> float:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    if default is not None:
        return default
    raise KeyError(f"None of these fields were present with values: {keys}")
