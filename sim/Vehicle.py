from typing import Union
from sim.Depot import Depot
from sim.Station import Station
from sim.Scooter import Scooter
import settings


class Vehicle:
    """
    Class for vehicle state. Keeps track of current location and inventory including a service route log.
    """

    def __init__(
        self,
        vehicle_id: int,
        start_location: Union[Station, Depot],
        battery_inventory_capacity: int,
        scooter_inventory_capacity: int,
    ):
        
        self.id = vehicle_id
        self.vehicle_id = self.id
        self.battery_inventory = battery_inventory_capacity
        self.battery_inventory_capacity = battery_inventory_capacity
        self.scooter_inventory = {}
        self.scooter_inventory_capacity = scooter_inventory_capacity
        self.location = start_location
        self.eta = 0
        self.speed = VEHICLE_SPEED  #imported from settings
        self.handling_time = HANDLING_TIME
        self.parking_time = PARKING_TIME

    def change_battery(self, scooter: Scooter):
        if self.battery_inventory <= 0:
            raise ValueError(
                "Can't change battery when the vehicle's battery inventory is empty"
            )
        else:
            self.battery_inventory -= 1
            scooter.swap_battery()
            return True

    def get_scooter_inventory(self):
        return list(self.scooter_inventory.values())

    def pick_up(self, scooter: Scooter):
        if len(self.scooter_inventory) + 1 > self.scooter_inventory_capacity:
            raise ValueError("Can't pick up an scooter when the vehicle is full")
        else:
            self.scooter_inventory[scooter.id] = scooter
            if scooter.hasBattery():
                if scooter.battery < 70:
                    self.change_battery(scooter)
            scooter.remove_location()

    def drop_off(self, scooter_id: int):
        scooter = self.scooter_inventory[scooter_id]
        del self.scooter_inventory[scooter_id]
        return scooter

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
            f"<Vehicle {self.id} at {self.location.id}, {len(self.scooter_inventory)} scooters,"
            f" {self.battery_inventory} batteries>"
        )

    def is_at_depot(self):
        return isinstance(self.location, Depot)

    def get_max_number_of_swaps(self):
        return (
            min(
                min(len(self.location.scooters), self.battery_inventory),
                len(self.location.get_swappable_scooters()),
            )
            if not self.is_at_depot()
            else 0
        )

    def flat_batteries(self):
        return self.battery_inventory_capacity - self.battery_inventory
