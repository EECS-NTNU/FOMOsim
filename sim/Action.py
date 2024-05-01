from settings import *


class Action:
    """
    Class representing an action.
    """

    def __init__(
        self,
        battery_swaps,
        pick_ups,
        delivery_bikes,
        next_location,
        cluster = None,
        helping_pickup = [],
        helping_delivery = [],
        helping_cluster = None
    ):
        """
        Object to represent an action
        :param battery_swaps: ids of bikes to swap batteries on
        :param pick_ups: ids of bikes to pick up
        :param delivery_bikes: ids of bikes to deliver
        :param next_location: id of next location to visit
        """
        self.battery_swaps = battery_swaps
        self.pick_ups = pick_ups
        self.delivery_bikes = delivery_bikes
        self.next_location = next_location
        self.cluster = cluster
        self.helping_pickup = helping_pickup
        self.helping_delivery = helping_delivery
        self.helping_cluster = helping_cluster

    def get_action_time(self, travel_time):
        """
        Get the time consumed from performing an action (travel from station 1 to 2) in a given state.
        Can add time for performing actions on bikes as well.
        :param travel_time: travel_time in min from current station to next station
        :return: Total time to perform action in minutes
        """
        operation_duration = (
            len(self.battery_swaps) + len(self.pick_ups) + len(self.delivery_bikes) + len(self.helping_pickup) + len(self.helping_delivery)
        ) * MINUTES_PER_ACTION
        travel_duration = (
            travel_time
            + MINUTES_CONSTANT_PER_ACTION
        )
        return operation_duration + travel_duration

    def __repr__(self):
        return (
            f"<Action - ({self.battery_swaps} bat. swaps, {self.pick_ups} pickups,"
            f" {self.delivery_bikes} deliveries), next: {self.next_location} >"
            f" {self.helping_pickup} h_pickups), {self.helping_delivery} h_delivry, {self.helping_cluster} cluster>"
        )
