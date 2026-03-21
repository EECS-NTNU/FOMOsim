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

        # 1. DO NOT log yet! We need to get the action first to see if it's an idle step.
        
        arrival_time = 0
        arrival_station_id = self.vehicle.location.id
        
        # 2. find the best action from the current simul state
        action = self.vehicle.policy.get_action(simul.state, self.vehicle)
        # print the policy it is using
        print(f"Vehicle {self.vehicle.id} using policy {self.vehicle.policy} at station {arrival_station_id} at time {simul_time}")
        if isinstance(action, tuple):
            action, _ = action

        # 3. Check if this is an "idle" step (at D0, staying at D0)
        is_idle_at_depot = (arrival_station_id == "D0" and action.next_location == "D0")

        # 4. Only log if NOT idle
        if operation_logger and operation_logger.enabled and not is_idle_at_depot:
            operation_logger.log_arrival(self.time, self.vehicle.id, arrival_station_id)
            operation_logger.log_decision_trigger(self.time, self.vehicle.id, arrival_station_id)
            
            operation_logger.log_action_selected(
                time=self.time,
                vehicle_id=self.vehicle.id,
                station_id=arrival_station_id,
                next_station=action.next_location,
                pick_ups=action.pick_ups,
                deliveries=action.delivery_bikes,
                battery_swaps=action.battery_swaps,
                onsite_repairs=action.onsite_repairs,
                #maintenance_time=float(getattr(action, "maintenance_time", 0.0)),
            )
            
            if arrival_station_id.startswith('D') and action.delivery_bikes:
                vehicle_inventory = {b.bike_id: b for b in self.vehicle.get_bike_inventory()}
                depot_repair_bikes = [
                    b_id for b_id in action.delivery_bikes
                    if getattr(vehicle_inventory.get(b_id), 'damage_status', None) == 'depot'
                ]
                if depot_repair_bikes:
                    operation_logger.log_depot_dropoff(
                        time=self.time,
                        vehicle_id=self.vehicle.id,
                        depot_id=arrival_station_id,
                        num_bikes=len(depot_repair_bikes),
                        bike_ids=depot_repair_bikes,
                    )

            # also log if picking up repaired bikes from depot
            if action.pick_ups and arrival_station_id.startswith('D'):
                vehicle_inventory = {b.bike_id: b for b in self.vehicle.get_bike_inventory()}
                depot_pickup_bikes = [
                    b_id for b_id in action.pick_ups
                    if getattr(vehicle_inventory.get(b_id), 'damage_status', None) == 'depot'
                ]
                if depot_pickup_bikes:
                    operation_logger.log_depot_pickup(
                        time=self.time,
                        vehicle_id=self.vehicle.id,
                        depot_id=arrival_station_id,
                        num_bikes=len(depot_pickup_bikes),
                        bike_ids=depot_pickup_bikes,
                    )

        # perform the best action on the state and send vehicle to new location
        refill_time = simul.state.do_action(action, self.vehicle, simul_time)

        action_time = (
            action.get_action_time(
                simul.state.get_vehicle_travel_time(arrival_station_id, action.next_location)
            )
            + refill_time
        )

        driving_time = simul.state.get_vehicle_travel_time(arrival_station_id, action.next_location)

        arrival_time += self.time + action_time

        simul.add_event(VehicleArrival(arrival_time, self.vehicle))

        self.vehicle.eta = arrival_time

        # --- Log Vehicle Departure Summary ---
        if arrival_station_id.startswith('D') and not is_idle_at_depot:
            func_cargo = sum(1 for b in self.vehicle.get_bike_inventory() if getattr(b, 'damage_status', None) not in ['depot', 'onsite'])
            broken_cargo = sum(1 for b in self.vehicle.get_bike_inventory() if getattr(b, 'damage_status', None) == 'depot')
            print(f"[VEHICLE DEPARTURE] t={simul_time:.1f} | Vehicle {self.vehicle.id} leaving {arrival_station_id} -> routing to {action.next_location} | Cargo: {func_cargo} functional, {broken_cargo} broken")

        # 5. Also suppress the departure log if idle
        if operation_logger and operation_logger.enabled and not is_idle_at_depot:
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
