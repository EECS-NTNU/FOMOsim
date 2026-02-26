from sim import Event
import sim
from settings import *
from settings import MAINTENANCE_LIMIT_TO_CHECK
from sim.bike_degradation_modeling.maintenance_model import update_bike_maintenance
from settings import ENABLE_COMPONENT_FAILURES, VERBOSE_FAILURE_TRACKING, SAMPLE_BIKES_TO_TRACK

class BikeArrival(Event):
    """
    Event performed when a bike arrives at a station after a bike departure
    """

    def __init__(
        self,
        time, 
        travel_time,
        bike,
        arrival_station_id,
        departure_station_id,
        congested = False,
        distance_km=0.0
    ):
        super().__init__(time + travel_time)
        self.bike = bike
        self.arrival_station_id = arrival_station_id
        self.departure_station_id = departure_station_id
        self.travel_time = travel_time
        self.congested = congested
        self.distance_km = distance_km

    def perform(self, simul) -> None:
        """
        :param simul: Simulation object
        """

        super().perform(simul)


        # get arrival station 
        arrival_station = simul.state.get_location_by_id(self.arrival_station_id)

        if not FULL_TRIP:
            self.bike = simul.state.get_used_bike()

        if self.bike is not None:
            """verbose_tracking = VERBOSE_FAILURE_TRACKING or (self.bike.bike_id in SAMPLE_BIKES_TO_TRACK)
            
            # Update bike state based on trip (includes failure probability checking)
            if ENABLE_COMPONENT_FAILURES and self.distance_km > 0 and verbose_tracking:
                # Manual call with verbose for tracked bikes
                self.bike._check_component_failures(simul, self.distance_km)
                self.bike.total_distance_km += self.distance_km
            else:
                # Normal travel call
                self.bike.travel(simul, self.travel_time, distance_km=self.distance_km, congested=self.congested)"""
            #
            self.bike.travel(simul, self.travel_time, distance_km=self.distance_km, congested=self.congested)
            #self.bike.total_distance_km += self.distance_km
            #

            if self.bike.battery < 0:
                simul.state.metrics.add_aggregate_metric(simul.state, "battery violations", 1)
                simul.state.metrics.add_aggregate_metric(simul.state, "failed events", 1)
                self.bike.battery = 0

            # add bike to the arrived station (location is changed in add_bike method)
            if arrival_station.add_bike(self.bike):
                if FULL_TRIP:
                    simul.state.remove_used_bike(self.bike)

                    # Check if maintenance is enabled in the policy (assuming uniform policy)
                    maintenance_enabled = True
                    if len(simul.state.vehicles) > 0:
                        # Get the first vehicle's policy
                        first_vehicle = next(iter(simul.state.vehicles.values()))
                        maintenance_enabled = first_vehicle.policy.maintenance_enabled
                    if maintenance_enabled:
                        try:
                            #old_crit = self.bike.maintenance_criticality
                            new_crit = update_bike_maintenance(
                                self.bike,
                                self.travel_time,
                                simul.state.rng,
                                battery_level=self.bike.battery,
                                congested=self.congested
                            )
                            self.bike.maintenance_criticality = new_crit
                        except Exception as e:
                            print(f"ERROR updating bike maintenance for {self.bike.bike_id}: {e}")
                            import traceback
                            traceback.print_exc()

                #if self.bike.usable() == False:
                    #simul.state.metrics.add_aggregate_metric(simul.state, "Maintenance violations", 1)

                # Track if this drop-off leaves the bike above the rental threshold
                #if self.bike.maintenance_criticality >= MAINTENANCE_THRESHOLD_FOR_NO_RENTAL:
                    #simul.state.metrics.add_aggregate_metric(simul.state, "maintenance_violation", 1)

                if self.bike.usable() == False:
                    simul.state.metrics.add_aggregate_metric(simul.state, "maintenance violations", 1)

                simul.state.metrics.add_aggregate_metric(simul.state, "bike arrival", 1)

                # Log bike movement for CSV output
                if hasattr(simul, 'log_bike_movement'):
                    simul.log_bike_movement(
                        time=self.time,
                        bike_id=self.bike.bike_id,
                        departure_station_id=self.departure_station_id,
                        arrival_station_id=self.arrival_station_id,
                        did_roam=False,
                        bike_criticality=self.bike.maintenance_criticality
                    )

            else:
                if FULL_TRIP:
                    # go to another station
                    next_station = simul.state.get_neighbouring_stations(arrival_station, 1, not_full=True)[0]

                    travel_time = simul.state.get_travel_time(
                        arrival_station.id,
                        next_station.id,
                    )
                    
                    # Calculate distance for roaming trip (used for both component failures AND metrics)
                    roaming_distance_km = arrival_station.distance_to(
                        next_station.lat, 
                        next_station.lon
                    )

                    # Print roaming information
                    maint_status = f"maint={self.bike.maintenance_criticality:.3f}" if hasattr(self.bike, 'maintenance_criticality') else ""
                    print(f"   ROAMING: Bike {self.bike.bike_id} - {arrival_station.id} FULL -> routing to {next_station.id} from {arrival_station.id} "
                          f"(t={self.time:.1f}, {maint_status}, +{travel_time:.1f}min, +{roaming_distance_km:.2f}km)")

                    # create an arrival event for the departed bike
                    simul.add_event(
                        sim.BikeArrival(
                            self.time,
                            travel_time,
                            self.bike,
                            next_station.id,
                            arrival_station.id,
                            congested = True,
                            distance_km=roaming_distance_km  # Pass the distance
                        )
                    )

                    simul.state.metrics.add_aggregate_metric(simul.state, "events", 1)


                else:
                    simul.state.set_bike_in_use(self.bike)

                distance = arrival_station.distance_to(next_station.get_lat(), next_station.get_lon())
                if distance <= MAX_ROAMING_DISTANCE_SOLUTIONS:
                    simul.state.metrics.add_aggregate_metric(simul.state, "short congestions", 1)
                else:
                    simul.state.metrics.add_aggregate_metric(simul.state, "long congestions", 1)
                    simul.state.metrics.add_aggregate_metric(simul.state, "failed events", 1)
                
                simul.state.metrics.add_aggregate_metric(simul.state, "roaming for locks", 1)
                simul.state.metrics.add_aggregate_metric(simul.state, "roaming distance for locks", distance)

    def __repr__(self):
        return f"<{self.__class__.__name__} at time {self.time}, arriving at station {self.arrival_station_id}>"
 