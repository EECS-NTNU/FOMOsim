import copy

from sim.Station import Station
from settings import *


class Depot(Station):
    """
    Class for depot. Contains method for updating the state of the depot.
    """

    def __init__(
        self,
        depot_id,
        is_station_based,
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
        capacity= DEFAULT_DEPOT_CAPACITY,
        original_id = None,
        charging_station = None,
    ):
        super().__init__(
            depot_id, bikes, leave_intensities, leave_intensities_stdev, arrive_intensities, arrive_intensities_stdev,
            center_location, move_probabilities, average_number_of_bikes, target_state,
            capacity, original_id, charging_station
        )

        self.is_station_based = is_station_based
        self.depot_capacity = depot_capacity
        self.battery_inventory = depot_capacity
        self.time = 0
        self.charging = []
        
        # ── Repair queue management ────────────────────────────────────────
        # fixed_queue: bikes finished repair, ready to be picked up by vehicles
        self.fixed_queue = 0
        # in_repair: list of (ready_timestamp, num_bikes) tuples
        # Bikes enter in_repair immediately, exit to fixed_queue when ready_timestamp <= current_time
        self.in_repair = []

    def sloppycopy(self, *args):
        return Depot(
            self.id,
            self.depot_capacity,
            self.is_station_based,
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

    def is_depot(self):
        return True

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
                f"Depot has only {self.battery_inventory} batteries available."
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
    
    # ── Repair queue management ────────────────────────────────────────────
    
    def receive_bikes_for_repair(self, num_bikes: int, current_time: float, repair_duration_minutes: float = 1440.0) -> None:
        """
        Receive broken bikes at the depot for off-site repair.
        
        When a vehicle drops off broken bikes (depot_cargo), they are added
        to the in_repair queue and will become available for pickup after
        repair_duration_minutes.
        
        Args:
            num_bikes                : number of bikes being dropped off for repair
            current_time             : current simulation time (minutes)
            repair_duration_minutes  : how long repair takes (default 24h = 1440 min)
        """
        if num_bikes <= 0:
            return
        
        ready_time = current_time + repair_duration_minutes
        self.in_repair.append((ready_time, num_bikes))
    
    def tick_repair_queue(self, current_time: float) -> int:
        """
        Process the repair queue: move finished bikes from in_repair to fixed_queue.
        
        Call this at each simulator tick to progress repairs across time.
        
        Args:
            current_time : current simulation time (minutes)
        
        Returns:
            bikes_completed : number of bikes that completed repair this tick
        """
        bikes_completed = 0
        remaining_in_repair = []
        
        for ready_time, num_bikes in self.in_repair:
            if current_time >= ready_time:
                # These bikes are done repairing
                bikes_completed += num_bikes
                self.fixed_queue += num_bikes
            else:
                # Still repairing
                remaining_in_repair.append((ready_time, num_bikes))
        
        self.in_repair = remaining_in_repair
        return bikes_completed
    
    def remove_bikes_from_queue(self, num_bikes: int) -> int:
        """
        Vehicle picks up repaired bikes from fixed_queue.
        
        Args:
            num_bikes : how many bikes the vehicle wants to pick up
        
        Returns:
            bikes_loaded : actual number loaded (≤ num_bikes, ≤ fixed_queue)
        """
        bikes_loaded = min(num_bikes, self.fixed_queue)
        self.fixed_queue -= bikes_loaded
        return bikes_loaded
    
    def get_repair_queue_status(self) -> tuple:
        """
        Returns (num_in_repair, num_in_fixed_queue).
        """
        total_in_repair = sum(count for _, count in self.in_repair)
        return total_in_repair, self.fixed_queue

    def __str__(self):
        return f"Depot   {self.id}: Arrive {self.get_arrive_intensity(0, 8):4.2f} Leave {self.get_leave_intensity(0, 8):4.2f} Ideal {self.get_target_state(0, 8)} Bikes {len(self.bikes):3d} Cap {self.depot_capacity} Inv {self.battery_inventory}"

    def __repr__(self):
        return f"<Depot, id: {self.id}, cap: {self.battery_inventory}>"
