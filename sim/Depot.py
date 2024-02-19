import copy

from sim.Station import Station
from settings import *


class Depot(Station):
    """
    Class for depot. Contains method for updating the state of the depot.
    """

    def __init__(
        self,
        depot_id: int,
        depot_capacity = DEFAULT_DEPOT_CAPACITY,
        bikes = [],
        leave_intensities=None,
        arrive_intensities=None,
        leave_intensities_stdev=None,
        arrive_intensities_stdev=None,
        center_location=None,
        move_probabilities=None,
        average_number_of_bikes=None,
        target_state=None,
        capacity=None,
        original_id = None,
        charging_station = None,
    ):
        super().__init__(
            depot_id, bikes, leave_intensities, leave_intensities_stdev, arrive_intensities, arrive_intensities_stdev,
            center_location, move_probabilities, average_number_of_bikes, target_state,
            capacity, original_id, charging_station
        )

        self.depot_capacity = depot_capacity
        self.battery_inventory = depot_capacity
        self.time = 0
        self.charging = []

    def sloppycopy(self, *args):
        return Depot(
            self.id,
            self.depot_capacity,
            list(copy.deepcopy(self.bikes).values()),
            leave_intensities=self.leave_intensities,
            arrive_intensities=self.arrive_intensities,
            center_location=self.get_location(),
            move_probabilities=self.move_probabilities,
            average_number_of_bikes=self.average_number_of_bikes,
            target_state=self.target_state,
            capacity=self.capacity,
            original_id=self.original_id,
            charging_station=self.charging_station,
        )

    def swap_battery_inventory(self, time, number_of_battery_to_change) -> int:
        """
        Method to perform a swapping of batteries at depot. Raises error if inventory is not capable.
        Adds the batteries to charging list, and reduces the battery inventory.

        Returns the time it takes to persom battery swaps at the depot.
        """
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
        """
        Returns the number of batteries that are fully charged.
        """
        return self.battery_inventory + self.get_delta_capacity(time, update_charging=False)

    def get_delta_capacity(self, time, update_charging=True):
        """
        Method that updates the charging list. 
        Removes the timestamp and amount of the batteries that are fully charged (if update = True), and counts the batteries that are now fully charged.

        Returns the amount of batteries that are now fully charged.
        """
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
        return f"Depot   {self.id:2d}: Arrive {self.get_arrive_intensity(0, 8):4.2f} Leave {self.get_leave_intensity(0, 8):4.2f} Ideal {self.get_target_state(0, 8):3d} Bikes {len(self.bikes):3d} Cap {self.depot_capacity} Inv {self.battery_inventory}"

    def __repr__(self):
        return f"<Depot, id: {self.id}, cap: {self.battery_inventory}>"
