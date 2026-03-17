"""
Operational debug logging for vehicle decisions and actions.

This module is intentionally lightweight and print-based so it can be used
inside the simulation loop without external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class OperationalLogger:
    """Structured operational logger for simulator events and actions."""

    enabled: bool = False
    include_bike_ids: bool = True
    prefix: str = "[OPS]"

    def _fmt_time(self, minutes: float) -> str:
        total = int(minutes)
        day = total // (24 * 60)
        rem = total % (24 * 60)
        hour = rem // 60
        minute = rem % 60
        return f"day={day} {hour:02d}:{minute:02d} (t={minutes:.2f})"

    def _emit(self, message: str) -> None:
        if self.enabled:
            print(f"{self.prefix} {message}")

    def _fmt_ids(self, ids: Optional[Iterable]) -> str:
        if ids is None:
            return "[]"
        ids_list = list(ids)
        if not self.include_bike_ids:
            return f"count={len(ids_list)}"
        return str(ids_list)

    def log_arrival(self, time: float, vehicle_id: str, station_id: str) -> None:
        self._emit(
            f"ARRIVAL {self._fmt_time(time)} vehicle={vehicle_id} station={station_id}"
        )

    def log_decision_trigger(self, time: float, vehicle_id: str, station_id: str) -> None:
        self._emit(
            f"DECISION_TRIGGER {self._fmt_time(time)} vehicle={vehicle_id} station={station_id}"
        )

    def log_action_selected(
        self,
        time: float,
        vehicle_id: str,
        station_id: str,
        next_station: str,
        pick_ups,
        deliveries,
        battery_swaps,
        maintenance_time: float,
    ) -> None:
        self._emit(
            "ACTION_SELECTED "
            f"{self._fmt_time(time)} vehicle={vehicle_id} station={station_id} "
            f"next={next_station} pickups={self._fmt_ids(pick_ups)} "
            f"deliveries={self._fmt_ids(deliveries)} swaps={self._fmt_ids(battery_swaps)} "
            f"maintenance_min={maintenance_time:.2f}"
        )

    def log_bike_pickup(self, time: float, vehicle_id: str, station_id: str, bike_id: str) -> None:
        self._emit(
            f"PICKUP {self._fmt_time(time)} vehicle={vehicle_id} station={station_id} bike={bike_id}"
        )

    def log_bike_dropoff(self, time: float, vehicle_id: str, station_id: str, bike_id: str) -> None:
        self._emit(
            f"DROPOFF {self._fmt_time(time)} vehicle={vehicle_id} station={station_id} bike={bike_id}"
        )

    def log_depot_dropoff(self, time: float, vehicle_id: str, depot_id: str, num_bikes: int, bike_ids=None) -> None:
        """Log bikes dropped off at depot for repair."""
        bikes_str = self._fmt_ids(bike_ids) if bike_ids else f"count={num_bikes}"
        self._emit(
            f"DEPOT_DROPOFF [REPAIR] {self._fmt_time(time)} vehicle={vehicle_id} depot={depot_id} bikes={bikes_str}"
        )

    def log_battery_swap(self, time: float, vehicle_id: str, station_id: str, bike_id: str) -> None:
        self._emit(
            f"BATTERY_SWAP {self._fmt_time(time)} vehicle={vehicle_id} station={station_id} bike={bike_id}"
        )

    def log_depot_battery_refill(
        self,
        time: float,
        vehicle_id: str,
        station_id: str,
        swapped_count: int,
    ) -> None:
        self._emit(
            f"DEPOT_BATTERY_REFILL {self._fmt_time(time)} vehicle={vehicle_id} "
            f"station={station_id} swapped={swapped_count}"
        )

    def log_maintenance(self, time: float, vehicle_id: str, station_id: str, maintenance_time: float) -> None:
        self._emit(
            f"MAINTENANCE {self._fmt_time(time)} vehicle={vehicle_id} station={station_id} "
            f"maintenance_min={maintenance_time:.2f}"
        )

    def log_inventory_update(
        self,
        time: float,
        station_id: str,
        station_before: int,
        station_after: int,
        vehicle_id: str,
        vehicle_load_before: int,
        vehicle_load_after: int,
    ) -> None:
        self._emit(
            "INVENTORY_UPDATE "
            f"{self._fmt_time(time)} station={station_id} bikes_before={station_before} "
            f"bikes_after={station_after} vehicle={vehicle_id} load_before={vehicle_load_before} "
            f"load_after={vehicle_load_after}"
        )

    def log_departure(
        self,
        time: float,
        vehicle_id: str,
        origin_id: str,
        destination_id: str,
        travel_time: float,
        action_time: float,
        expected_arrival: float,
    ) -> None:
        self._emit(
            "DEPARTURE "
            f"{self._fmt_time(time)} vehicle={vehicle_id} from={origin_id} to={destination_id} "
            f"travel_min={travel_time:.2f} action_min={action_time:.2f} "
            f"eta={self._fmt_time(expected_arrival)}"
        )
