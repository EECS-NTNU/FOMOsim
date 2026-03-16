import time

import sim
from sim.events.Event import Event
import copy


class VehicleArrival(Event):
    """
    Event where the main decision is done. A vehicle arrives to a station and need to determine what to do.
    Different policies can be applied depending on the policy object in the simul object.
    """

    def __init__(self, arrival_time: int, vehicle: sim.Vehicle):
        super().__init__(arrival_time)
        self.vehicle = vehicle

    def perform(self, simul) -> None:
        """
        :param simul: simul object
        """
        simul_time = simul.state.time;

        operation_logger = getattr(simul, "operation_logger", None)

        super().perform(simul)

        if operation_logger and operation_logger.enabled:
            operation_logger.log_arrival(self.time, self.vehicle.id, self.vehicle.location.id)
            operation_logger.log_decision_trigger(self.time, self.vehicle.id, self.vehicle.location.id)
        
        arrival_time = 0
        # find the best action from the current simul state
        action = self.vehicle.policy.get_action(simul.state, self.vehicle)
        if isinstance(action, tuple):
            action, _ = action

        if operation_logger and operation_logger.enabled:
            operation_logger.log_action_selected(
                time=self.time,
                vehicle_id=self.vehicle.id,
                station_id=self.vehicle.location.id,
                next_station=action.next_location,
                pick_ups=action.pick_ups,
                deliveries=action.delivery_bikes,
                battery_swaps=action.battery_swaps,
                maintenance_time=float(getattr(action, "maintenance_time", 0.0)),
            )

        # Record current location of vehicle to compute action time
        arrival_station_id = self.vehicle.location.id

        # perform the best action on the state and send vehicle to new location
        refill_time = simul.state.do_action(action, self.vehicle, simul_time)

        action_time = (
            action.get_action_time(
                simul.state.get_vehicle_travel_time(arrival_station_id, action.next_location)
            )
            + refill_time
        )

        driving_time = simul.state.get_vehicle_travel_time(arrival_station_id, action.next_location)

        # Compute the arrival time for the Vehicle arrival event created by the action
        arrival_time += self.time + action_time

        # Add a new Vehicle Arrival event for the next station arrival to the simul event_queue
        simul.add_event(VehicleArrival(arrival_time, self.vehicle))

        self.vehicle.eta = arrival_time

        if operation_logger and operation_logger.enabled:
            operation_logger.log_departure(
                time=self.time,
                vehicle_id=self.vehicle.id,
                origin_id=arrival_station_id,
                destination_id=action.next_location,
                travel_time=driving_time,
                action_time=action_time,
                expected_arrival=arrival_time,
            )

    def __repr__(self):
        return f"<{self.__class__.__name__} at time {self.time} to location {self.vehicle.location.id}>"
