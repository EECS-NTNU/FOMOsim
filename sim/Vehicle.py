from typing import Union
from sim.Depot import Depot
from sim.Station import Station
from settings import *

class Vehicle:
    """
    Class for vehicle state. Keeps track of current location and inventory including a service route log.
    """

    def __init__(
        self,
        vehicle_id,
        start_location,
        policy,
        battery_inventory_capacity,
        bike_inventory_capacity,
        is_station_based
    ):
        
        self.id = vehicle_id
        self.battery_inventory = battery_inventory_capacity
        self.battery_inventory_capacity = battery_inventory_capacity
        self.bike_inventory = {}
        self.bike_inventory_capacity = bike_inventory_capacity
        self.location = start_location
        self.policy = policy
        self.is_station_based = is_station_based
        self.eta = 0
        self.handling_time = MINUTES_PER_ACTION
        self.parking_time = MINUTES_CONSTANT_PER_ACTION
        self.cluster = None
        self.NUMBER = 77


    def change_battery(self, bike):
        if self.battery_inventory <= 0:
            raise ValueError(
                "Can't change battery when the vehicle's battery inventory is empty"
            )
        else:
            self.battery_inventory -= 1
            bike.swap_battery()
            return True

    def get_bike_inventory(self):
        return list(self.bike_inventory.values())
    
    def get_sb_bike_inventory(self):
        return [bike for bike in self.bike_inventory.values() if bike.is_station_based]
    
    def get_ff_bike_inventory(self):
        return [bike for bike in self.bike_inventory.values() if not bike.is_station_based]
    
    def get_available_ff_bikes(self):
        return [bike for bike in self.bike_inventory.values() if bike.usable() and not bike.is_station_based]

    def pick_up(self, bike):
        if len(self.bike_inventory) + 1 > self.bike_inventory_capacity:
            raise ValueError("Can't pick up an bike when the vehicle is full")
        else:
            self.bike_inventory[bike.bike_id] = bike
            if bike.hasBattery():
                if bike.battery < self.policy.swap_threshold and self.battery_inventory > 0:
                    self.change_battery(bike)
            bike.remove_location()

    def drop_off(self, bike_id):
        bike = self.bike_inventory[bike_id]
        del self.bike_inventory[bike_id]
        return bike

    def add_battery_inventory(self, number_of_batteries):
        if (
            number_of_batteries + self.battery_inventory
            > self.battery_inventory_capacity
        ):
            raise ValueError(
                f"Adding {number_of_batteries} exceeds the vehicles capacity ({self.battery_inventory_capacity})."
                f"Current battery inventory: {self.battery_inventory}"
            )
        else:
            self.battery_inventory += number_of_batteries

    def __repr__(self):
        return (
            f"<Vehicle {self.id} at {self.location.id}, {len(self.bike_inventory)} bikes,"
            f" {self.battery_inventory} batteries>"
        )

    def is_at_depot(self):
        return isinstance(self.location, Depot)

    def get_max_number_of_swaps(self):
        return (
            min(
                len(self.location.bikes), 
                self.battery_inventory,
                len(self.location.get_swappable_bikes()),
            )
            if not self.is_at_depot()
            else 0
        )

    def get_flat_batteries(self):
        return self.battery_inventory_capacity - self.battery_inventory
