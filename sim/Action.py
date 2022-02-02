from globals import *


class Action:
    """
    Class representing an action.
    """

    def __init__(
        self,
        battery_swaps: [int],
        pick_ups: [int],
        delivery_scooters: [int],
        next_location: int,
    ):
        """
        Object to represent an action
        :param battery_swaps: ids of scooters to swap batteries on
        :param pick_ups: ids of scooters to pick up
        :param delivery_scooters: ids of scooters to deliver
        :param next_location: id of next location to visit
        """
        self.battery_swaps = battery_swaps
        self.pick_ups = pick_ups
        self.delivery_scooters = delivery_scooters
        self.next_location = next_location

    def get_action_time(self, distance):
        """
        Get the time consumed from performing an action (travel from cluster 1 to 2) in a given state.
        Can add time for performing actions on scooters as well.
        :param distance: distance in km from current cluster to next cluster
        :return: Total time to perform action in minutes
        """
        operation_duration = (
            len(self.battery_swaps) + len(self.pick_ups) + len(self.delivery_scooters)
        ) * MINUTES_PER_ACTION
        travel_duration = (
            round((distance / VEHICLE_SPEED) * MINUTES_IN_HOUR)
            + MINUTES_CONSTANT_PER_ACTION
        )
        return operation_duration + travel_duration

    def __repr__(self):
        return (
            f"<Action - ({len(self.battery_swaps)} bat. swaps, {len(self.pick_ups)} pickups,"
            f" {len(self.delivery_scooters)} deliveries), next: {self.next_location} >"
        )

    def get_reward(
        self,
        vehicle,
        lost_trip_reward,
        deopt_reward,
        vehicle_inventory_step,
        pick_up_reward,
    ):
        battery_reward = 0
        # Record number of scooters that become available during the action
        available_scooters = 0
        vehicle_location = vehicle.current_location
        if not vehicle.is_at_depot():
            available_scooters = len(vehicle_location.get_available_scooters())
            for scooter_id in self.battery_swaps:
                battery_swap_scooter = vehicle.current_location.get_scooter_from_id(
                    scooter_id
                )
                battery_reward += (
                    (100.0 - battery_swap_scooter.battery) / 100.0
                )# * vehicle_location.prob_of_scooter_usage(available_scooters)
                if battery_swap_scooter.battery < BATTERY_LIMIT:
                    # If the swapped scooter was unavailable, make sure probability of scooter usage decrease.
                    available_scooters += 1
            # Calculate estimated lost trip reward
            estimated_lost_trip_reward = lost_trip_reward * max(
                vehicle_location.trip_intensity_per_iteration - available_scooters,
                0,
            )
            # Get 1 in reward for every delivery and battery reward according to probability of usage
            return (
                len(self.delivery_scooters)
                + len(self.pick_ups) * pick_up_reward
                + battery_reward
                + estimated_lost_trip_reward
            )
        else:
            return (
                deopt_reward
                if vehicle.battery_inventory / vehicle.battery_inventory_capacity
                <= vehicle_inventory_step
                else 0
            )
