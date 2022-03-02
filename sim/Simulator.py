import copy
import datetime
from typing import List

import numpy as np
import bisect
import sim
import settings
from sim.SaveMixin import SaveMixin
from sim import Metric

from progress.bar import IncrementalBar

class Simulator(SaveMixin):
    """
    Class containing all metadata about an instance. This class contains both the state, the policy and parameters.
    This class uses the state as the environment and the policy as the actor. Additionally, it is the main driver of the
    event based simulation system using the event classes.
    """

    def __init__(
        self,
        shift_duration: int,
        policy,
        initial_state,
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
        self.event_queue: List[sim.Event] = []
        self.tabu_list = []
        # Initialize the event_queue with a vehicle arrival for every vehicle at time zero
        for vehicle in self.state.vehicles:
            self.event_queue.append(
                sim.VehicleArrival(0, vehicle.id, visualize=visualize)
            )
        # Add Generate Scooter Trip event to the event_queue
        self.event_queue.append(sim.GenerateScooterTrips(settings.ITERATION_LENGTH_MINUTES))
        self.cluster_flow = {
            (start, end): 0
            for start in np.arange(len(self.state.stations))
            for end in np.arange(len(self.state.stations))
            if start != end
        }
        self.policy = policy
        policy.initSim(self)
        self.metrics = Metric()
        self.verbose = verbose
        self.visualize = visualize
        if label is None:
          self.label = self.__class__.__name__
        else:
          self.label = label
        if verbose:
            self.progress_bar = IncrementalBar(
                "Running Sim",
                check_tty=False,
                max=round(shift_duration / settings.ITERATION_LENGTH_MINUTES) + 1,
                color=settings.WHITE,
                suffix="%(percent)d%% - ETA %(eta)ds",
            )

    def __repr__(self):
        string = f"<Sim with {self.time} of {self.shift_duration} elapsed. {len(self.event_queue)} events in event_queue>"
        return string

    def single_step(self):
        event = self.event_queue.pop(0)

        if settings.VERBOSE:
          print()
          print("Time:", event.time)
          print(self.state)

        event.perform(self)

        if settings.VERBOSE:
            print("\n", event)

        return event

    def full_step(self):
        while True:
            event = self.single_step()
            if isinstance(event, sim.GenerateScooterTrips):
                break

    def run(self):
        """
        Main method for running the Event Based Simulation Engine.

        The sim object uses a queue initialized with vehicle arrival events and a GenerateScooterTrips event.
        It then pops events from this queue. The queue is always sorted in by the time of the events.
        """
        while self.time < self.shift_duration:
            self.full_step()
            if self.verbose:
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

    def add_event(self, event: sim.Event) -> None:
        """
        Adds event to the sorted queue.
        Avoids calling sort on every iteration by using the bisect package
        :param event: event to insert
        """
        insert_index = bisect.bisect([event.time for event in self.event_queue], event.time)
        self.event_queue.insert(insert_index, event)

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
            for event in self.event_queue
            if isinstance(event, sim.ScooterArrival)
        ]

    def get_discount(self):
        # Divide by 60 as there is 60 minutes in an hour. We want this number in hours to avoid big numbers is the power
        return settings.DISCOUNT_RATE ** (self.time / 60)

    def get_filename(self):
        return (
            f"{self.created_at}_Sim_T_e{self.time}_t_{self.shift_duration}_"
            f"S_c{len(self.state.stations)}_s{len(self.state.get_scooters())}"
        )

    def save_sim(self, cache_directory=None, suffix=""):
        directory = settings.SIM_CACHE_DIR
        if cache_directory:
            directory = f"{settings.SIM_CACHE_DIR}/{cache_directory}"
        super().save(directory, f"-{suffix}")

    # def sloppycopy(self, *args):
    #     new_sim = Simulator(
    #         self.shift_duration,
    #         self.policy,
    #         self.state.sloppycopy(),
    #         verbose=self.verbose,
    #         visualize=self.visualize,
    #     )
    #     new_sim.time = self.time
    #     new_sim.stack = copy.deepcopy(self.stack)
    #     new_sim.tabu_list = self.tabu_list.copy()
    #     new_sim.cluster_flow = self.cluster_flow.copy()
    #     new_sim.metrics = copy.deepcopy(self.metrics)
    #     # Set all hyper parameters
    #     for parameter in HyperParameters().__dict__.keys():
    #         setattr(new_sim, parameter, getattr(self, parameter))
    #     return new_sim
