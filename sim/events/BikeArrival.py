from sim import Event
import sim
from settings import *
from settings import MAINTENANCE_LIMIT_TO_CHECK
from sim.maintenance_model import update_bike_maintenance

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
        congested = False
    ):
        super().__init__(time + travel_time)
        self.bike = bike
        self.arrival_station_id = arrival_station_id
        self.departure_station_id = departure_station_id
        self.travel_time = travel_time
        self.congested = congested

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
            # Update bike state based on trip (includes maintenance increase)
            self.bike.travel(simul, self.travel_time, self.congested)
        

            if self.bike.battery < 0:
                simul.state.metrics.add_aggregate_metric(simul.state, "battery violations", 1)
                simul.state.metrics.add_aggregate_metric(simul.state, "failed events", 1)
                self.bike.battery = 0

            # add bike to the arrived station (location is changed in add_bike method)
            if arrival_station.add_bike(self.bike):
                if FULL_TRIP:
                    print(f"   DROP-OFF: Bike {self.bike.bike_id} at {arrival_station.id} from {self.departure_station_id} "
                          f"(t={self.time:.1f}, +{self.travel_time:.1f}min)")
                    simul.state.remove_used_bike(self.bike)

                    # Check if maintenance is enabled in the policy (assuming uniform policy)
                    maintenance_enabled = True
                    if len(simul.state.vehicles) > 0:
                        print("DEBUG: Checking maintenance policy from vehicles...")
                        # Get the first vehicle's policy
                        first_vehicle = next(iter(simul.state.vehicles.values()))
                        maintenance_enabled = first_vehicle.policy.maintenance_enabled
                    print(f"DEBUG: Maintenance enabled: {maintenance_enabled}")
                    if maintenance_enabled:
                        try:
                            old_crit = self.bike.maintenance_criticality
                            print("HEEEEEEI",old_crit)
                            new_crit = update_bike_maintenance(
                                self.bike,
                                self.travel_time,
                                simul.state.rng,
                                battery_level=self.bike.battery,
                                congested=self.congested
                            )
                            print("HAAAAALLLAAA",new_crit)
                            self.bike.maintenance_criticality = new_crit
                            # Debug: print first few updates to verify it's working
                            if simul.state.time < 500:  # Only print early in simulation
                                expected_increase = self.travel_time * 0.00125
                                actual_increase = new_crit - old_crit
                                print(f"DEBUG MAINT UPDATE: Bike {self.bike.bike_id} travel={self.travel_time:.1f}min, "
                                      f"crit: {old_crit:.4f} -> {new_crit:.4f} (expected +{expected_increase:.6f}, actual +{actual_increase:.6f})")
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

                # CHECKPOINT
                # Print bike arrival information
                maint_status = f"maint={self.bike.maintenance_criticality:.3f}" if hasattr(self.bike, 'maintenance_criticality') else ""
                usable_status = " usable" if self.bike.usable() else " UNUSABLE"
                congestion_str = " [CONGESTED]" if self.congested else ""
                print(f"   ARRIVAL: Bike {self.bike.bike_id} at {arrival_station.id} "
                      f"(t={self.time:.1f}, {maint_status}, {usable_status}){congestion_str}")

                simul.state.metrics.add_aggregate_metric(simul.state, "bike arrival", 1)

            else:
                if FULL_TRIP:
                    # go to another station
                    next_station = simul.state.get_neighbouring_stations(arrival_station, 1, not_full=True)[0]

                    travel_time = simul.state.get_travel_time(
                        arrival_station.id,
                        next_station.id,
                    )
                    
                    # Print roaming information
                    maint_status = f"maint={self.bike.maintenance_criticality:.3f}" if hasattr(self.bike, 'maintenance_criticality') else ""
                    print(f"   ROAMING: Bike {self.bike.bike_id} - {arrival_station.id} FULL -> routing to {next_station.id} from {arrival_station.id} "
                          f"(t={self.time:.1f}, {maint_status}, +{travel_time:.1f}min)")

                    # create an arrival event for the departed bike
                    simul.add_event(
                        sim.BikeArrival(
                            self.time,
                            travel_time,
                            self.bike,
                            next_station.id,
                            arrival_station.id,
                            congested = True
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
 