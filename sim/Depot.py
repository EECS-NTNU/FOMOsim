from sim.Station import Station
from sim.Scooter import Scooter
from settings import *


class Depot(Station):
    """
    Class for depot. Contains method for updating the state of the depot.
    """

    def __init__(
        self,
        depot_id: int,
        main_depot = False,
        scooters: [Scooter] = [],
        leave_intensity_per_iteration=None,
        arrive_intensity_per_iteration=None,
        center_location=None,
        move_probabilities=None,
        average_number_of_scooters=None,
        target_state=None,
        capacity=DEFAULT_STATION_CAPACITY,
        original_id = None,
        charging_station = None,
    ):
        super().__init__(
            depot_id, scooters, leave_intensity_per_iteration, arrive_intensity_per_iteration,
            center_location, move_probabilities, average_number_of_scooters, target_state,
            capacity, original_id, charging_station
        )

        self.battery_inventory = MAIN_DEPOT_CAPACITY if main_depot else SMALL_DEPOT_CAPACITY
        self.time = 0
        self.charging = []

    def swap_battery_inventory(self, time, number_of_battery_to_change) -> int:
        self.battery_inventory += self.get_delta_capacity(time)
        self.time = time

        if number_of_battery_to_change > self.battery_inventory:
            raise ValueError(
                f"Depot has only {self.battery_inventory} batteries available. "
                f"Vehicle tried to swap {number_of_battery_to_change}"
            )

        self.battery_inventory -= number_of_battery_to_change

        self.charging.append((time, number_of_battery_to_change))

        return (
            round(number_of_battery_to_change * SWAP_TIME_PER_BATTERY)
            + CONSTANT_DEPOT_DURATION
        )

    def get_available_battery_swaps(self, time):
        return self.battery_inventory + self.get_delta_capacity(time, update_charging=False)

    def get_delta_capacity(self, time, update_charging=True):
        delta_capacity = 0
        time_filter = (
            lambda filter_time, filter_charging_start_time: filter_time
            > filter_charging_start_time + CHARGE_TIME_PER_BATTERY
        )
        for i, (charging_start_time, number_of_batteries) in enumerate(self.charging):
            if time_filter(time, charging_start_time):
                delta_capacity += number_of_batteries
        if update_charging:
            self.charging = [
                (charging_start_time, number_of_batteries)
                for charging_start_time, number_of_batteries in self.charging
                if not time_filter(time, charging_start_time)
            ]
        return delta_capacity

    def __str__(self):
        return f"Depot {self.id}"

    def __repr__(self):
        return f"<Depot, id: {self.id}, cap: {self.battery_inventory}>"
