import copy
import datetime
from typing import List

import numpy as np
import bisect
import sim
import globals
from sim.SaveMixin import SaveMixin

from globals import (
    HyperParameters,
    WHITE,
    SIM_CACHE_DIR,
    ITERATION_LENGTH_MINUTES,
)

from progress.bar import IncrementalBar


class Simulator(SaveMixin, HyperParameters):
    """
    Class containing all metadata about an instance. This class contains both the state, the policy and parameters.
    This class uses the state as the environment and the policy as the actor. Additionally, it is the main driver of the
    event based simulation system using the event classes.
    """

    class Metric:
        """
        Class for storing and aggregate the metric data of the instance
        """

        def __init__(self, test_parameter_name="", test_parameter_value=0.0):
            self.lost_demand = []
            self.deficient_battery = []
            self.timeline = []
            self.total_available_scooters = []
            self.testing_parameter_name = test_parameter_name
            self.testing_parameter_value = test_parameter_value

        @classmethod
        def aggregate_metrics(cls, metrics):
            def lists_average(lists):
                return np.mean(np.stack(lists, axis=0), axis=1).tolist()

            new_sim_metric = cls()
            if all([len(metric.timeline) == 0 for metric in metrics]):
                return new_sim_metric
            number_of_metrics = len(metrics)

            # Fields to take the average of
            average_fields = [
                "lost_demand",
                "deficient_battery",
                "total_available_scooters",
            ]
            # Create dict with list for every field, start all values on zero
            fields = {field: [[0] * number_of_metrics] for field in average_fields}
            # Find the time for the latest event
            max_time = np.max(np.concatenate([metric.timeline for metric in metrics]))
            new_sim_metric.timeline = list(range(int(max_time) + 1))
            # populate fields with average at every time step
            for time in new_sim_metric.timeline[1:]:
                # If there is a new value in the timeline, update the timeline
                if any([time in metric.timeline for metric in metrics]):
                    for field in fields.keys():
                        # Add new value if there is a new one, otherwise add previous value
                        fields[field].append(
                            [
                                getattr(metric, field)[
                                    metric.timeline.index(time)
                                ]  # Takes the first recording in current time
                                if time in metric.timeline
                                else fields[field][time - 1][i]
                                for i, metric in enumerate(metrics)
                            ]
                        )
                # Otherwise, add previous values
                else:
                    for field in fields.keys():
                        fields[field].append(fields[field][-1])
            # Take the average of all the runs
            new_sim_metric.__dict__.update(
                {
                    field: lists_average(metric_list)
                    for field, metric_list in fields.items()
                }
            )

            new_sim_metric.testing_parameter_name = metrics[0].testing_parameter_name
            new_sim_metric.testing_parameter_value = metrics[
                0
            ].testing_parameter_value
            return new_sim_metric

        def add_analysis_metrics(self, sim):
            """
            Add data to analysis
            :param sim: sim object to record state from
            """
            self.lost_demand.append(
                sum(
                    [
                        1
                        for reward, location in sim.rewards
                        if reward == sim.LOST_TRIP_REWARD
                    ]
                )
                if len(sim.rewards) > 0
                else 0
            )
            self.deficient_battery.append(
                sum([scooter.battery for scooter in sim.state.get_scooters()]) / len(sim.state.get_scooters())
                # sum(
                #     [
                #         cluster.ideal_state * 100
                #         - (
                #             sum(
                #                 [
                #                     scooter.battery
                #                     for scooter in cluster.get_available_scooters()
                #                 ]
                #             )
                #         )
                #         for cluster in sim.state.stations
                #         if len(cluster.scooters) < cluster.ideal_state
                #     ]
                # )
                # / len(sim.state.get_scooters())
            )
            self.total_available_scooters.append(
                sum(
                    [
                        len(cluster.get_available_scooters())
                        for cluster in sim.state.stations
                    ]
                )
            )
            self.timeline.append(sim.time)

        def get_all_metrics(self):
            """
            Returns all metrics recorded for analysis
            """
            return (
                self.lost_demand,
                self.deficient_battery,
                self.total_available_scooters,
            )

    def __init__(
        self,
        shift_duration: int,
        policy,
        initial_state,
        test_parameter_name="",
        test_parameter_value=None,
        verbose=False,
        visualize=True,
        label=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.created_at = datetime.datetime.now().isoformat(timespec="minutes")
        self.shift_duration = shift_duration
        self.state = initial_state
        self.time = 0
        self.rewards = []
        self.stack: List[sim.Event] = []
        self.tabu_list = []
        # Initialize the stack with a vehicle arrival for every vehicle at time zero
        number_of_vans, number_of_bikes = 0, 0
        for vehicle in self.state.vehicles:
            self.stack.append(
                sim.VehicleArrival(0, vehicle.id, visualize=visualize)
            )
            if vehicle.scooter_inventory_capacity > 0:
                number_of_vans += 1
            else:
                number_of_bikes += 1
        self.NUMBER_OF_VANS = number_of_vans
        self.NUMBER_OF_BIKES = number_of_bikes
        # Add Generate Scooter Trip event to the stack
        self.stack.append(sim.GenerateScooterTrips(ITERATION_LENGTH_MINUTES))
        self.cluster_flow = {
            (start, end): 0
            for start in np.arange(len(self.state.stations))
            for end in np.arange(len(self.state.stations))
            if start != end
        }
        self.policy = policy
        policy.initSim(self)
        self.metrics = Simulator.Metric(test_parameter_name, test_parameter_value)
        self.verbose = verbose
        self.visualize = visualize
        if label is None:
          self.label = self.__class__.__name__
        else:
          self.label = label
        self.disable_training = False
        if verbose:
            self.progress_bar = IncrementalBar(
                "Running Sim",
                check_tty=False,
                max=round(shift_duration / ITERATION_LENGTH_MINUTES) + 1,
                color=WHITE,
                suffix="%(percent)d%% - ETA %(eta)ds",
            )

    def __repr__(self):
        return f"<Sim with {self.time} of {self.shift_duration} elapsed. {len(self.stack)} events in stack>"

    def run(self):
        """
        Main method for running the Event Based Simulation Engine.

        The sim object uses a stack initialized with vehicle arrival events and a GenerateScooterTrips event.
        It then pops events from this stack. The stack is always sorted in by the time of the events.
        """
        while self.time < self.shift_duration:
            event = self.stack.pop(0)
            event.perform(self)
            if isinstance(event, sim.GenerateScooterTrips) and self.verbose:
                self.progress_bar.next()
        if self.verbose:
            self.progress_bar.finish()

    def get_remaining_time(self) -> int:
        """
        Computes the remaining time by taking the difference between the shift duration
        and the current time of the sim object.
        :return: the remaining time as a float
        """
        return self.shift_duration - self.time

    def add_reward(self, reward: float, location_id: int, discount=False) -> None:
        """
        Adds the input reward to the rewards list of the sim object
        :param location_id: location where the reward was conducted
        :param discount: boolean if the reward is to be discounted
        :param reward: reward given
        """
        self.rewards.append(
            (reward * self.get_discount(), location_id)
            if discount
            else (reward, location_id)
        )

    def get_total_reward(self) -> float:
        """
        Get total accumulated reward at current point of time
        :return:
        """
        return sum([reward for reward, location_id in self.rewards])

    def add_event(self, event: sim.Event) -> None:
        """
        Adds event to the sorted stack.
        Avoids calling sort on every iteration by using the bisect package
        :param event: event to insert
        """
        insert_index = bisect.bisect([event.time for event in self.stack], event.time)
        self.stack.insert(insert_index, event)

    def add_trip_to_flow(self, start: int, end: int) -> None:
        """
        Adds a trip from start to end for cluster flow
        :param start: departure cluster
        :param end: arrival cluster
        """
        self.cluster_flow[(start, end)] += 1

    def get_cluster_flow(self) -> [(int, int, int)]:
        """
        Get all flows between cluster since last vehicle arrival
        :return: list: tuple (start, end, flow) flow from departure cluster to arrival cluster
        """
        return [(start, end, flow) for (start, end), flow in self.cluster_flow.items()]

    def clear_flow_dict(self) -> None:
        """
        Clears the cluster flow dict
        """
        for key in self.cluster_flow.keys():
            self.cluster_flow[key] = 0

    def get_scooters_on_trip(self) -> [(int, int, int)]:
        """
        Get all scooters that are currently out on a trip
        :return: list of all scooters that are out on a trip
        """
        return [
            (event.departure_cluster_id, event.arrival_cluster_id, event.scooter.id)
            for event in self.stack
            if isinstance(event, sim.ScooterArrival)
        ]

    def get_discount(self):
        # Divide by 60 as there is 60 minutes in an hour. We want this number in hours to avoid big numbers is the power
        return self.DISCOUNT_RATE ** (self.time / 60)

    def get_filename(self):
        return (
            f"{self.created_at}_Sim_T_e{self.time}_t_{self.shift_duration}_"
            f"S_c{len(self.state.stations)}_s{len(self.state.get_scooters())}"
        )

    def save_sim(self, cache_directory=None, suffix=""):
        directory = SIM_CACHE_DIR
        if cache_directory:
            directory = f"{SIM_CACHE_DIR}/{cache_directory}"
        super().save(directory, f"-{suffix}")

    def get_train_directory(self, suffix=None):
        suffix = suffix if suffix else f"{self.created_at}"
        return (
            f"trained_models/{self.policy.value_function.__repr__()}/"
            f"c{len(self.state.stations)}_s{len(self.state.get_scooters())}/{suffix}"
        )

    def system_simulate(self):
        """
        Simulation of poisson process on the system
        Poisson distributed number of trips out of each cluster, markov chain decides where the trip goes
        :param state: current world
        :return: flows generated by the system simulation
        """
        flow_counter = {
            (start, end): 0
            for start in np.arange(len(self.state.stations))
            for end in np.arange(len(self.state.stations))
            if start != end
        }
        trips = []
        lost_demand = []
        scenario = random.choice(self.state.simulation_scenarios)
        for start_cluster_id, number_of_trips, end_cluster_indices in scenario:
            start_cluster = self.state.get_location_by_id(start_cluster_id)
            # if there is more trips than scooters available, the system has lost demand
            valid_scooters = start_cluster.get_available_scooters()
            if number_of_trips > len(valid_scooters):
                lost_demand.append(
                    (number_of_trips - len(valid_scooters), start_cluster_id)
                )
                end_cluster_indices = end_cluster_indices[: len(valid_scooters)]

            # loop to generate trips from the cluster
            for j, end_cluster_index in enumerate(end_cluster_indices):
                trips.append(
                    (
                        start_cluster,
                        self.state.get_location_by_id(end_cluster_index),
                        valid_scooters.pop(0),
                    )
                )
                flow_counter[(start_cluster.id, end_cluster_index)] += 1

        # compute trip after all trips are generated to avoid handling inflow in cluster
        for start_cluster, end_cluster, scooter in trips:
            start_cluster.scooters.remove(scooter)
            trip_distance = self.state.get_distance(start_cluster.id, end_cluster.id)
            scooter.travel(trip_distance)
            end_cluster.add_scooter(scooter)

        return (
            [(start, end, flow) for (start, end), flow in list(flow_counter.items())],
            trips,
            lost_demand,
        )

    def __deepcopy__(self, *args):
        new_sim = Simulator(
            self.shift_duration,
            self.policy,
            copy.deepcopy(self.state),
            verbose=self.verbose,
            visualize=self.visualize,
        )
        new_sim.time = self.time
        new_sim.rewards = self.rewards.copy()
        new_sim.stack = copy.deepcopy(self.stack)
        new_sim.tabu_list = self.tabu_list.copy()
        new_sim.cluster_flow = self.cluster_flow.copy()
        new_sim.metrics = copy.deepcopy(self.metrics)
        new_sim.disable_training = self.disable_training
        # Set all hyper parameters
        for parameter in HyperParameters().__dict__.keys():
            setattr(new_sim, parameter, getattr(self, parameter))
        return new_sim
