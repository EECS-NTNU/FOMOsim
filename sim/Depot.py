import copy

from sim.Station import Station
from settings import *
from sim.bike_degradation_modeling.bike_component_maintenance_model import ComponentMaintenanceManager


class Depot(Station):
    """
    Class for depot. Contains method for updating the state of the depot.
    """

    def __init__(
        self,
        depot_id,
        is_station_based,
        depot_capacity = DEFAULT_DEPOT_CAPACITY,
        bikes = None,
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
        
        if bikes is None:
            bikes = []
            
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
        #              keyed by bike_id, same layout as self.bikes
        self.fixed_queue = {}
        # in_repair: list of (ready_timestamp, bikes_list) tuples
        # Bikes enter in_repair immediately, exit to fixed_queue when ready_timestamp <= current_time
        self.in_repair = []

    '''def sloppycopy(self, *args):
        return Depot(
            self.id,
            self.is_station_based,
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
        )'''
        
    def sloppycopy(self, bike_map=None, *args):
        if bike_map is None:
            bike_map = {}
        cloned_bikes = []
        for bike in self.bikes.values():
            if bike.bike_id not in bike_map:
                bike_map[bike.bike_id] = copy.copy(bike)
            cloned_bikes.append(bike_map[bike.bike_id])

        new_depot = Depot(
            self.id,
            self.is_station_based,
            self.depot_capacity,
            cloned_bikes,
            leave_intensities=self.leave_intensities,
            arrive_intensities=self.arrive_intensities,
            leave_intensities_stdev=self.leave_intensities_stdev,
            arrive_intensities_stdev=self.arrive_intensities_stdev,
            center_location=self.get_location(),
            move_probabilities=self.move_probabilities,
            average_number_of_bikes=self.average_number_of_bikes,
            target_state=self.target_state,
            capacity=self.capacity,
            original_id=self.original_id,
            charging_station=self.charging_station,
        )
        
        # Safely copy the repair queues!
        new_depot.fixed_queue = {}
        for bid, bike in self.fixed_queue.items():
            if bike.bike_id not in bike_map:
                bike_map[bike.bike_id] = copy.copy(bike)
            new_depot.fixed_queue[bid] = bike_map[bike.bike_id]

        new_depot.in_repair = []
        for ready_time, bikes in self.in_repair:
            cloned_queue_bikes = []
            for bike in bikes:
                if bike.bike_id not in bike_map:
                    bike_map[bike.bike_id] = copy.copy(bike)
                cloned_queue_bikes.append(bike_map[bike.bike_id])
            new_depot.in_repair.append((ready_time, cloned_queue_bikes))

        new_depot.battery_inventory = self.battery_inventory
        new_depot.time = self.time
        new_depot.charging = list(self.charging)
        
        return new_depot

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
    
    def receive_bikes_for_repair(self, bikes: list, current_time: float, repair_duration_minutes: float = 1440.0) -> None:
        """
        Receive broken bikes at the depot for off-site repair.
        
        Args:
            bikes                    : list of bike objects being dropped off for repair
            current_time             : current simulation time (minutes)
            repair_duration_minutes  : how long repair takes (default 24h = 1440 min)
        """
        if not bikes:
            return
        
        ready_time = current_time + repair_duration_minutes
        self.in_repair.append((ready_time, bikes))

        for b in bikes:
            print(f"[DEPOT REPAIR QUEUED] t={current_time:.1f} | Bike {b.bike_id} will be ready for pickup at t={ready_time:.1f}")
    
    
    def tick_repair_queue(self, current_time: float) -> int:
        bikes_completed = 0
        remaining_in_repair = []
        
        for ready_time, bikes_list in self.in_repair:
            if current_time >= ready_time:

                # These bikes are done repairing. 
                for bike in bikes_list:
                    category_to_fix = getattr(bike, 'pending_failure_category', None) or getattr(bike, 'last_failure_category', None)
                    
                    if category_to_fix:
                        ComponentMaintenanceManager.repair_component(
                            bike, 
                            category_to_fix, 
                            repair_type="depot", 
                            verbose=True
                        )
                    
                    # Clear all flags so it is fully functional
                    ComponentMaintenanceManager.clear_damage_status(bike)
                    
                    # Move to fixed_queue, ready for vehicle pickup
                    self.fixed_queue[bike.bike_id] = bike

                    print(f"[DEPOT REPAIR DONE] t={current_time:.1f} | Bike {bike.bike_id} finished 24h repair for '{category_to_fix}'. Moved to fixed_queue. (Status: {bike.damage_status})")

                bikes_completed += len(bikes_list)
            else:
                remaining_in_repair.append((ready_time, bikes_list))
        
        self.in_repair = remaining_in_repair
        return bikes_completed
    
    '''def remove_bikes_from_queue(self, num_bikes: int) -> list:
        """
        Vehicle picks up repaired bikes from fixed_queue.
        
        Returns:
            bikes_loaded : list of bike objects loaded onto the vehicle
        """
        bikes_loaded = []
        keys = list(self.fixed_queue.keys())[:num_bikes]
        for key in keys:
            bikes_loaded.append(self.fixed_queue.pop(key))
        return bikes_loaded
    
    def get_repair_queue_status(self) -> tuple:
        """
        Returns (num_in_repair, num_in_fixed_queue).
        """
        total_in_repair = sum(len(bikes) for _, bikes in self.in_repair)
        return total_in_repair, len(self.fixed_queue)'''
        
    def get_bike_from_id(self, bike_id):
        """Override to also search the fixed_queue for freshly repaired bikes."""
        if bike_id in self.bikes:
            return self.bikes[bike_id]
        if hasattr(self, 'fixed_queue') and bike_id in self.fixed_queue:
            return self.fixed_queue[bike_id]
        # Fallback to parent just in case
        return super().get_bike_from_id(bike_id)

    def remove_bike(self, bike):
        """Override to allow removing freshly repaired bikes directly from the fixed_queue."""
        if bike.bike_id in self.bikes:
            del self.bikes[bike.bike_id]
        elif hasattr(self, 'fixed_queue') and bike.bike_id in self.fixed_queue:
            del self.fixed_queue[bike.bike_id]
        else:
            # Fallback to parent
            super().remove_bike(bike)

    def __str__(self):
        return f"Depot   {self.id}: Arrive {self.get_arrive_intensity(0, 8):4.2f} Leave {self.get_leave_intensity(0, 8):4.2f} Ideal {self.get_target_state(0, 8)} Bikes {len(self.bikes):3d} Cap {self.depot_capacity} Inv {self.battery_inventory}"

    def __repr__(self):
        return f"<Depot, id: {self.id}, cap: {self.battery_inventory}>"
