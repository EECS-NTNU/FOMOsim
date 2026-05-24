"""Add a central depot to a station-based instance file.

Default use for Oslo:

    python add_depot_to_instance.py --input instances/OS_W31.json.gz --in-place

The simulator names depot locations sequentially as D0, D1, ... when loading the
JSON. The numeric ``id`` field in the JSON is kept only as source metadata.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import geopy.distance
import numpy as np

from settings import ESCOOTER_SPEED, VEHICLE_SPEED, DEFAULT_DEPOT_CAPACITY


def read_instance(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def write_instance(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def zero_week_profile() -> list[list[float]]:
    return [[0.0] * 24 for _ in range(7)]


def central_location(stations: list[dict]) -> list[float]:
    station_locations = np.array(
        [station["location"] for station in stations if not station.get("is_depot", False)],
        dtype=float,
    )
    if station_locations.size == 0:
        raise ValueError("Cannot place depot because the instance contains no regular stations.")
    return [float(station_locations[:, 0].mean()), float(station_locations[:, 1].mean())]


def travel_minutes(from_location: list[float], to_location: list[float], speed_kmh: float) -> float:
    if from_location == to_location:
        return 0.0
    distance_km = geopy.distance.distance(tuple(from_location), tuple(to_location)).km
    return float((distance_km / speed_kmh) * 60.0)


def expand_square_matrix(matrix, stations: list[dict], depot_location: list[float], speed_kmh: float):
    """Append depot row/column when the matrix is a square station matrix."""
    if matrix is None:
        return None
    n = len(stations)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError(f"Expected a {n}x{n} matrix, got inconsistent dimensions.")

    expanded = [list(row) for row in matrix]
    depot_col = [
        travel_minutes(station["location"], depot_location, speed_kmh)
        for station in stations
    ]
    for row, value in zip(expanded, depot_col):
        row.append(value)

    depot_row = [
        travel_minutes(depot_location, station["location"], speed_kmh)
        for station in stations
    ]
    depot_row.append(0.0)
    expanded.append(depot_row)
    return expanded


def expand_stdev_matrix(matrix, stations: list[dict]):
    if matrix is None:
        return None
    n = len(stations)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError(f"Expected a {n}x{n} stdev matrix, got inconsistent dimensions.")
    expanded = [list(row) + [0.0] for row in matrix]
    expanded.append([0.0] * (n + 1))
    return expanded


def add_central_depot(data: dict, capacity: int, depot_capacity: int, force: bool = False) -> dict:
    stations = list(data["stations"])
    existing = [station for station in stations if station.get("is_depot", False)]
    if existing and not force:
        print(f"Instance already has {len(existing)} depot(s); no depot added.")
        return data

    regular_stations = [station for station in stations if not station.get("is_depot", False)]
    depot_location = central_location(regular_stations)

    depot = {
        "id": len(stations),
        "original_id": "central_depot",
        "location": depot_location,
        "is_depot": True,
        "capacity": int(capacity),
        "depot_capacity": int(depot_capacity),
        "num_bikes": 0,
        "leave_intensities": zero_week_profile(),
        "arrive_intensities": zero_week_profile(),
        "leave_intensities_stdev": zero_week_profile(),
        "arrive_intensities_stdev": zero_week_profile(),
        "move_probabilities": [[[] for _ in range(24)] for _ in range(7)],
    }

    data = dict(data)
    data["stations"] = stations + [depot]

    data["traveltime"] = expand_square_matrix(
        data.get("traveltime"), stations, depot_location, ESCOOTER_SPEED
    )
    data["traveltime_stdev"] = expand_stdev_matrix(data.get("traveltime_stdev"), stations)
    data["traveltime_vehicle"] = expand_square_matrix(
        data.get("traveltime_vehicle"), stations, depot_location, VEHICLE_SPEED
    )
    data["traveltime_vehicle_stdev"] = expand_stdev_matrix(
        data.get("traveltime_vehicle_stdev"), stations
    )

    print("Added depot:")
    print(f"  JSON id          : {depot['id']}")
    print("  simulator id     : D0")
    print(f"  location         : {depot_location}")
    print(f"  capacity         : {capacity}")
    print(f"  depot capacity   : {depot_capacity}")
    print(f"  total locations  : {len(data['stations'])}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add a central depot to an instance JSON file.")
    parser.add_argument("--input", type=Path, default=Path("instances/OS_W31.json.gz"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--in-place", action="store_true", help="Overwrite the input file.")
    parser.add_argument("--force", action="store_true", help="Add a depot even if one already exists.")
    parser.add_argument("--capacity", type=int, default=100)
    parser.add_argument("--depot-capacity", type=int, default=DEFAULT_DEPOT_CAPACITY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is None and not args.in_place:
        stem = args.input.name.removesuffix(".json.gz").removesuffix(".json")
        args.output = args.input.with_name(f"{stem}_with_depot.json.gz")
    elif args.in_place:
        args.output = args.input

    print(f"Loading {args.input}...")
    data = read_instance(args.input)
    original_count = len(data["stations"])
    data = add_central_depot(
        data,
        capacity=args.capacity,
        depot_capacity=args.depot_capacity,
        force=args.force,
    )

    print(f"Writing {args.output}...")
    write_instance(args.output, data)
    print(f"Done. Stations/locations: {original_count} -> {len(data['stations'])}.")


if __name__ == "__main__":
    main()
