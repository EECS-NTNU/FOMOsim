from typing import Union
from classes.Depot import Depot
from classes.Cluster import Cluster
from classes.Scooter import Scooter


class Vehicle:
    """
    Class for vehicle state. Keeps track of current location and inventory including a service route log.
    """

    def __init__(
        self,
        vehicle_id: int,
        start_location: Union[Cluster, Depot],
        battery_inventory_capacity: int,
        scooter_inventory_capacity: int,
    ):
        self.id = vehicle_id
        self.battery_inventory = battery_inventory_capacity
        self.battery_inventory_capacity = battery_inventory_capacity
        self.scooter_inventory = []
        self.scooter_inventory_capacity = scooter_inventory_capacity
        self.service_route = []
        self.current_location = start_location

    def change_battery(self, scooter: Scooter):
        if self.battery_inventory <= 0:
            raise ValueError(
                "Can't change battery when the vehicle's battery inventory is empty"
            )
        else:
            self.battery_inventory -= 1
            scooter.swap_battery()
            return True

    def pick_up(self, scooter: Scooter):
        if len(self.scooter_inventory) + 1 > self.scooter_inventory_capacity:
            raise ValueError("Can't pick up an scooter when the vehicle is full")
        else:
            self.scooter_inventory.append(scooter)
            if scooter.battery < 70:
                self.change_battery(scooter)
            scooter.remove_location()

    def drop_off(self, scooter_id: int):
        if scooter_id not in map(
            lambda inventory_scooter: inventory_scooter.id, self.scooter_inventory
        ):
            raise ValueError(
                "Can't deliver a scooter that isn't in the vehicle inventory"
            )

        scooter = next(
            (scooter for scooter in self.scooter_inventory if scooter.id == scooter_id),
            None,
        )
        self.scooter_inventory.remove(scooter)
        return scooter

    def set_current_location(self, location: Cluster, action):
        self.service_route.append((self.current_location, action))
        self.current_location = location

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

    def get_route(self):
        return self.service_route

    def __repr__(self):
        return (
            f"<Vehicle {self.id} at {self.current_location.id}, {len(self.scooter_inventory)} scooters,"
            f" {self.battery_inventory} batteries>"
        )

    def is_at_depot(self):
        return isinstance(self.current_location, Depot)

    def get_max_number_of_swaps(self):
        return (
            min(
                min(len(self.current_location.scooters), self.battery_inventory),
                len(self.current_location.get_swappable_scooters()),
            )
            if not self.is_at_depot()
            else 0
        )

    def flat_batteries(self):
        return self.battery_inventory_capacity - self.battery_inventory
