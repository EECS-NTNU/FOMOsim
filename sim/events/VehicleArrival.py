import time

import sim
from sim.events.Event import Event
import copy


class VehicleArrival(Event):
    """
    Event where the main decision is done. A vehicle arrives to a station and need to determine what to do.
    Different policies can be applied depending on the policy object in the world object.
    """

    def __init__(self, arrival_time: int, vehicle: sim.Vehicle):
        super().__init__(arrival_time)
        self.vehicle = vehicle

    def perform(self, world) -> None:
        """
        :param world: world object
        """

        world_time = world.time

        super().perform(world)

        arrival_time = 0
        
        # find the best action from the current world state
        action = self.vehicle.policy.get_action(world, self.vehicle)
        
        if isinstance(action, tuple):
            action, _ = action

        # Record current location of vehicle to compute action time
        arrival_station_id = self.vehicle.location.location_id

        self.vehicle.cluster = action.cluster

        # perform the best action on the state and send vehicle to new location
        refill_time = world.state.do_action(action, self.vehicle, world_time)

        action_time = (
            action.get_action_time(
                world.state.get_vehicle_travel_time(arrival_station_id, action.next_location)
            )
            + refill_time
        )

        driving_time = world.state.get_vehicle_travel_time(arrival_station_id, action.next_location)

        # Compute the arrival time for the Vehicle arrival event created by the action
        arrival_time += self.time + action_time + driving_time

        # Add a new Vehicle Arrival event for the next station arrival to the world event_queue
        world.add_event(VehicleArrival(arrival_time, self.vehicle))

        self.vehicle.eta = arrival_time

    def __repr__(self):
        return f"<{self.__class__.__name__} at time {self.time} to location {self.vehicle.location.location_id}>"
