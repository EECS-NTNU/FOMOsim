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

from progress.bar import IncrementalBar

from helpers import loggTime, loggLocations, loggEvent
# import policies.inngjerdingen_moeller

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
        self.state.time = start_time
        self.event_queue: List[sim.Event] = []

        # Add generate trip event to the event_queue
        self.event_queue.append(sim.GenerateBikeTrips(start_time))
        self.event_queue.append(sim.GenerateEScooterTrips(start_time))
        # Initialize the event_queue with a vehicle arrival for every vehicle at time zero
        for vehicle in self.state.get_vehicles():
            self.event_queue.append(
                sim.VehicleArrival(self.state.time, vehicle)
            )

        #self.metrics = Metric()
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
                check_tty=True,
                max=round(duration / settings.ITERATION_LENGTH_MINUTES) + 1,
                suffix="%(percent)d%% - ETA %(eta)ds",
            )

        for vehicle in self.state.get_vehicles():
            vehicle.policy.init_sim(self)

    def __repr__(self):
        string = f"<Sim with {self.state.time} of {self.end_time} elapsed. {len(self.event_queue)} events in event_queue>"
        return string

    def single_step(self):
        event = self.event_queue.pop(0)

        if settings.TRAFFIC_LOGGING:
            loggTime(event.time)
            loggLocations(self.state)

        if self.state.time <= event.time:
            self.state.time = event.time
        else:
            raise ValueError(
                f"{event.__class__.__name__} object tries to move the simul backwards in time. "
                f"Event time: {event.time}, World time: {self.state.time}"
            )
        self._tick_repair_queues()

        event.perform(self)

        if settings.TRAFFIC_LOGGING:
            loggEvent(event)

        #self.metrics.add_analysis_metrics(self)
        self.state.metrics.add_analysis_metrics(self.state)
        
        # ── Tick repair queues ─────────────────────────────────────────────
        self._tick_repair_queues()

        monotonic = time.monotonic()
        if self.cluster:
            if (monotonic - self.last_monotonic) > 1:
                print(".", end="")
                sys.stdout.flush()
                self.last_monotonic = monotonic

        # Plotting system state
        # if self.state.time > 5000 and self.state.time <= 5000: 
        #     policies.inngjerdingen_moeller.visualize_stations_from_simulator(self)

    def full_step(self):
        while True:
            event = self.event_queue[0]
            if isinstance(event, sim.GenerateBikeTrips):
                d = int((event.time // (60*24)) % 7)
                h = int((event.time // 60) % 24)
                # self.demand.update_demands(self.state, d, h)
                if self.target_state is not None:
                    #print(f"Target state: {self.target_state}")
                    self.target_state.update_target_state(self.state, d, h)
                self.single_step()
                break

            self.single_step()

    def run(self):
        """
        Main method for running the Event Based Simulation Engine.

        The sim object uses a queue initialized with vehicle arrival events and a GenerateBikeTrips event.
        It then pops events from this queue. The queue is always sorted in by the time of the events.
        """
        # Print all stations and depots for verification
        #print("\n=== Stations and Depots ===")
        for station in self.state.get_stations():
            is_depot = isinstance(station, sim.Depot)
            #marker = "[DEPOT]" if is_depot else "[STATION]"
            #print(f"{marker} {station.id:4s} | capacity={station.capacity:3d} | bikes_available={station.number_of_bikes():3d}")
        for depot in self.state.get_depots():
            #print(f"[DEPOT] {depot.id:4s} | capacity={depot.capacity:3d} | bikes_available={depot.number_of_bikes():3d}")
            print()
        
        while self.state.time < self.end_time:
            self.full_step()
            if self.verbose:
                self.progress_bar.next()
        if self.verbose:
            self.progress_bar.finish()

    def _tick_depot_repair_queues(self) -> None:
        """
        Process repair queues at all depots.
        
        Called after each event to move bikes from in_repair → fixed_queue
        as their 24-hour repair period completes.
        """
        for depot in self.state.get_depots():
            depot.tick_repair_queue(self.state.time)

    def _tick_onsite_repair_queues(self) -> None:
        """Complete station on-site repairs whose service time has elapsed."""
        for station in self.state.get_stations():
            if hasattr(station, "tick_onsite_repair_queue"):
                station.tick_onsite_repair_queue(self.state)

    def _tick_repair_queues(self) -> None:
        """Process all maintenance queues that can complete at the current time."""
        self._tick_depot_repair_queues()
        self._tick_onsite_repair_queues()

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

    '''def sloppycopy(self, *args):
        new_sim = Simulator(
            0,
            self.state.sloppycopy(),
            self.state.time,
            self.verbose,
            self.label,
        )
        new_sim.duration = self.end_time
        new_sim.event_queue = copy.deepcopy(self.event_queue)
        #new_sim.metrics = copy.deepcopy(self.metrics)
        new_sim.state.metrics = copy.deepcopy(self.state.metrics)
        return new_sim'''
        
    def sloppycopy(self, *args):
        import copy
        new_sim = Simulator(
            duration=0,
            initial_state=self.state.sloppycopy(),
            target_state=self.target_state,
            demand=self.demand,
            start_time=self.state.time,
            cluster=self.cluster,
            verbose=self.verbose,
            label=self.label,
        )
        new_sim.end_time = self.end_time
        
        # FIXED EVENT QUEUE COPY (No more ghost pointers!)
        new_sim.event_queue = []
        for e in self.event_queue:
            e_copy = copy.copy(e)
            if hasattr(e_copy, 'vehicle') and e_copy.vehicle is not None:
                e_copy.vehicle = new_sim.state.get_vehicle_by_id(e_copy.vehicle.id)
            new_sim.event_queue.append(e_copy)
            
        # 🚨 METRICS FIX: We explicitly DO NOT copy the metrics here!
        # By doing nothing, new_sim.state.metrics remains a completely fresh, 
        # empty scoreboard created by State.__init__(). 
        # The fake rollout actions will no longer inflate the real numbers!
            
        return new_sim
    
    
    # ---------------------------------------------------------
    # DUMMY LOGGING METHODS FOR ROLLOUTS
    # ---------------------------------------------------------
    # These prevent console flooding when a cloned base Simulator 
    # (used in rollout lookaheads) processes events. 
    # We intentionally do NOT want rollouts to write to the real logs!
    
    def log_bike_movement(self, *args, **kwargs):
        pass

    def log_trip_request(self, *args, **kwargs):
        pass
