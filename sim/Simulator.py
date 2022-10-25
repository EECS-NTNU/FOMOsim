import copy
import datetime
from typing import List
import time
import sys

import bisect
import sim
import settings
from sim.LoadSave import LoadSave
from sim import Metric
import target_state
import demand

from progress.bar import IncrementalBar

from helpers import loggTime, loggLocations, loggEvent

class Simulator(LoadSave):
    """
    Class containing all metadata about an instance. This class contains both the state, the policy and parameters.
    This class uses the state as the environment and the policy as the actor. Additionally, it is the main driver of the
    event based simulation system using the event classes.
    """

    def __init__(
            self,
            duration,
            initial_state,
            target_state,
            demand,
            start_time = 0,
            cluster=False,
            verbose=False,
            label=None,
    ):
        super().__init__()
        self.created_at = datetime.datetime.now().isoformat(timespec="minutes")
        self.init(duration=duration, initial_state=initial_state, target_state=target_state, demand=demand, start_time=start_time, cluster=cluster, verbose=verbose, label=label)

    def init(
        self,
        initial_state,
        target_state,
        demand,
        start_time = 0,
        duration = 0,
        cluster = False,
        verbose = False,
        label = None,
    ):
        self.end_time = start_time + duration
        self.state = initial_state
        self.target_state = target_state
        self.demand = demand
        self.time = start_time
        self.event_queue: List[sim.Event] = []
        # Initialize the event_queue with a vehicle arrival for every vehicle at time zero
        for vehicle in self.state.vehicles:
            self.event_queue.append(
                sim.VehicleArrival(self.time, vehicle)
            )
        # Add generate trip event to the event_queue
        self.event_queue.append(sim.GenerateBikeTrips(start_time))
        self.metrics = Metric()
        self.cluster = cluster
        self.verbose = verbose
        if label is None:
          self.label = "Sim"
        else:
          self.label = label
        if cluster:
            self.last_monotonic = time.monotonic()
        if verbose:
            self.progress_bar = IncrementalBar(
                "Running Sim",
                check_tty=False,
                max=round(duration / settings.ITERATION_LENGTH_MINUTES) + 1,
                suffix="%(percent)d%% - ETA %(eta)ds",
            )

        for vehicle in self.state.vehicles:
            vehicle.policy.init_sim(self)

    def __repr__(self):
        string = f"<Sim with {self.time} of {self.end_time} elapsed. {len(self.event_queue)} events in event_queue>"
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

        monotonic = time.monotonic()
        if self.cluster:
            if (monotonic - self.last_monotonic) > 1:
                print(".", end="")
                sys.stdout.flush()
                self.last_monotonic = monotonic

        return event

    def full_step(self):
        while True:
            event = self.single_step()
            if isinstance(event, sim.GenerateBikeTrips):
                break

    def run(self):
        """
        Main method for running the Event Based Simulation Engine.

        The sim object uses a queue initialized with vehicle arrival events and a GenerateBikeTrips event.
        It then pops events from this queue. The queue is always sorted in by the time of the events.
        """
        while self.time < self.end_time:
            # TODO: check time
            self.demand.update_demands(self.state, self.day(), self.hour())
            self.target_state.update_target_state(self.state, self.day(), self.hour())
            self.full_step()
            if self.verbose:
                self.progress_bar.next()
        if self.verbose:
            self.progress_bar.finish()

    def day(self):
        return int((self.time // (60*24)) % 7)

    def hour(self):
        return int((self.time // 60) % 24)

    def add_event(self, event: sim.Event) -> None:
        """
        Adds event to the sorted queue.
        Avoids calling sort on every iteration by using the bisect package
        :param event: event to insert
        """
        insert_index = bisect.bisect([event.time for event in self.event_queue], event.time)
        self.event_queue.insert(insert_index, event)

    def save_sim(self, filename):
        directory = f"{settings.SIM_CACHE_DIR}/{filename}.pickle.gz"
        super().save(directory)

    @staticmethod
    def load_sim(filename):
        directory = f"{settings.SIM_CACHE_DIR}/{filename}.pickle.gz"
        return sim.Simulator.load(directory)

    def sloppycopy(self, *args):
        new_sim = Simulator(
            0,
            self.state.sloppycopy(),
            self.time,
            self.verbose,
            self.label,
        )
        new_sim.duration = self.end_time
        new_sim.event_queue = copy.deepcopy(self.event_queue)
        new_sim.metrics = copy.deepcopy(self.metrics)
        return new_sim
