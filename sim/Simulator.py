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

from init_state.cityBike.helpers import loggTime, loggLocations, loggEvent

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
        start_time = 0,
        verbose=False,
        label=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.created_at = datetime.datetime.now().isoformat(timespec="minutes")
        self.shift_duration = start_time + shift_duration
        self.state = initial_state
        self.time = start_time
        self.event_queue: List[sim.Event] = []
        # Initialize the event_queue with a vehicle arrival for every vehicle at time zero
        for vehicle in self.state.vehicles:
            self.event_queue.append(
                sim.VehicleArrival(self.time, vehicle)
            )
        # Add Generate Scooter Trip event to the event_queue
        self.event_queue.append(sim.GenerateScooterTrips(start_time + settings.ITERATION_LENGTH_MINUTES))
        self.policy = policy
        policy.init_sim(self)
        self.metrics = Metric()
        self.verbose = verbose
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

    def init(
        self,
        shift_duration: int,
        initial_state,
        start_time = 0,
        verbose=False,
        label=None,
    ):
        self.shift_duration = start_time + shift_duration
        self.state = initial_state
        self.time = start_time
        self.event_queue: List[sim.Event] = []
        # Initialize the event_queue with a vehicle arrival for every vehicle at time zero
        for vehicle in self.state.vehicles:
            self.event_queue.append(
                sim.VehicleArrival(self.time, vehicle)
            )
        # Add Generate Scooter Trip event to the event_queue
        self.event_queue.append(sim.GenerateScooterTrips(start_time + settings.ITERATION_LENGTH_MINUTES))
        self.metrics = Metric()
        self.verbose = verbose
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

        if settings.TRAFFIC_LOGGING:
            loggTime(event.time)
            loggLocations(self.state)

        event.perform(self)

        if settings.TRAFFIC_LOGGING:
            loggEvent(event)

        self.metrics.add_analysis_metrics(self)

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

    def day(self):
        return (self.time // (60*24)) % 7

    def hour(self):
        return (self.time // 60) % 24

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
        if label is not None:
            return label
        else:
            return "sim"

    def save_sim(self, filename):
        directory = f"{settings.SIM_CACHE_DIR}/{filename}"
        super().save(directory)

    def sloppycopy(self, *args):
        new_sim = Simulator(
            0,
            self.policy,
            self.state.sloppycopy(),
            self.time,
            self.verbose,
            self.label,
        )
        new_sim.shift_duration = self.shift_duration
        new_sim.event_queue = copy.deepcopy(self.event_queue)
        new_sim.metrics = copy.deepcopy(self.metrics)
        return new_sim
