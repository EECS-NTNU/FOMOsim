import time

from sim.events.Event import Event
import copy


class VehicleArrival(Event):
    """
    Event where the main decision is done. A vehicle arrives to a cluster and need to determine what to do.
    Different policies can be applied depending on the policy object in the world object.
    """

    def __init__(self, arrival_time: int, vehicle_id: int, visualize=True):
        super().__init__(arrival_time)
        self.visualize = visualize
        self.vehicle_id = vehicle_id

    def perform(self, world, **kwargs) -> None:
        """
        :param world: world object
        """
        try:
            vehicle = [
                vehicle
                for vehicle in world.state.vehicles
                if vehicle.id == self.vehicle_id
            ][0]
        except IndexError:
            raise ValueError(
                "OBS! Something went wrong. The vehicle is not in this state."
            )

        # Remove current vehicle state from tabu list
        world.tabu_list = [
            cluster_id
            for cluster_id in world.tabu_list
            if cluster_id != vehicle.current_location.id
        ]

        if self.visualize:
            # copy state before action for visualization purposes
            state_before_action = copy.deepcopy(world.state)
            vehicle_before_action = copy.deepcopy(vehicle)

        arrival_time = 0
        # find the best action from the current world state
        action = world.policy.get_best_action(world, vehicle)

        print("\n", action)

        if isinstance(action, tuple):
            action, _ = action

        if self.visualize:
            # visualize vehicle route
            world.state.visualize_vehicle_routes(
                self.vehicle_id,
                vehicle.current_location.id,
                action.next_location,
                world.tabu_list,
                world.policy.__str__(),
            )

        # clear world flow counter dictionary
        world.clear_flow_dict()

        # Record current location of vehicle to compute action time
        arrival_cluster_id = vehicle.current_location.id

        # perform the best action on the state and send vehicle to new location
        refill_time = world.state.do_action(action, vehicle, world.time)

        # Add next vehicle location to tabu list
        if not vehicle.is_at_depot():
            world.tabu_list.append(action.next_location)

        action_time = (
            action.get_action_time(
                world.state.get_distance(arrival_cluster_id, action.next_location)
            )
            + refill_time
        )

        if self.visualize:
            # visualize action performed by vehicle
            state_before_action.visualize_action(
                vehicle_before_action,
                world.state,
                vehicle,
                action,
                world.time,
                action_time,
                False,
                world.policy.__str__(),
            )

        # set time of world to this event's time
        super(VehicleArrival, self).perform(world, **kwargs)

        # Compute the arrival time for the Vehicle arrival event created by the action
        arrival_time += self.time + action_time

        # Add a new Vehicle Arrival event for the next cluster arrival to the world event_queue
        world.add_event(VehicleArrival(arrival_time, vehicle.id, self.visualize))
