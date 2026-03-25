import sim
from sim import Event
from settings import *
import numpy as np
import random
#from sim.bike_degradation_modeling.calibrated_severity_params import get_severity_params
from sim.bike_degradation_modeling.calibrated_severity_params import assign_damage_severity  # Changed import
from sim.bike_degradation_modeling import ComponentFailureModel
from settings import SAMPLE_BIKES_TO_TRACK
 
class BikeDeparture(Event):
    """
    Event fired when a customer requests a trip from a given departure station. Creates a Lost Trip or Bike Arrival
    event based on the availability of the station
    """
 
    def __init__(self, time, departure_station_id=None, bike=None, departure_station=None, 
             arrival_station=None, travel_time=None, distance_km=0.0, congested=False, 
             arrival_station_id=None):
        super().__init__(time)
        self.bike = bike
        self.departure_station = departure_station
        self.arrival_station = arrival_station
        self.travel_time = travel_time
        self.distance_km = distance_km
        self.congested = congested
        self.departure_station_id = departure_station_id
        self.arrival_station_id = arrival_station_id
 
    def perform(self, simul) -> None:
        """
        :param simul: Simulation object
        """
 
        super().perform(simul)
 
        # get departure station
        departure_station = simul.state.get_location_by_id(self.departure_station_id)
 
        # get all available bike in the station
        available_bikes = departure_station.get_available_bikes()
        total_bikes_at_station = departure_station.number_of_bikes()
 
        # if there are no more available bikes -> make a LostTrip event for that departure time
        if len(available_bikes) > 0:
            bike = available_bikes.pop(0)
 
            if FULL_TRIP:
                if simul.state.rng.random() < RANDOM_DESTINATION_PROB:
                    # Exclude the current area from the random selection
                    other_stations = [station.id for station in simul.state.get_stations() if station.id != self.departure_station_id]
                    arrival_station_id = simul.state.rng.choice(other_stations)
                else:
                    # get an arrival station from the leave prob distribution
                    mp = departure_station.get_move_probabilities(simul.state, simul.state.day(), simul.state.hour())
                    p = list(mp.values())
                    sum = 0.0
                    for i in range(len(p)):
                        sum += p[i]
                    p_normalized = []
                    for i in range(len(p)):
                        if sum > 0:
                            p_normalized.append(p[i] * (1.0/sum))
                        else:
                            p_normalized.append(1/len(p))
                    arrival_station_id = simul.state.rng.choice(list(mp.keys()), p = p_normalized)
 
                arrival_station = simul.state.get_location_by_id(arrival_station_id)
 
                travel_time = simul.state.get_travel_time(
                    departure_station.id,
                    arrival_station.id,
                )

                # NEW: Calculate distance for component failure tracking
                distance_km = 0.0
                if ENABLE_COMPONENT_FAILURES:
                    distance_km = departure_station.distance_to(
                        arrival_station.lat, 
                        arrival_station.lon
                    )

                    # NEW: Evaluate component failure risk BEFORE the trip starts
                    self._evaluate_trip_failure_risk(simul, bike, distance_km, 
                                                     departure_station.id, arrival_station_id)
 
 
                # create an arrival event for the departed bike
                simul.add_event(
                    sim.BikeArrival(
                        self.time,
                        travel_time,
                        bike,
                        arrival_station.id,
                        departure_station.id,
                        distance_km=distance_km,  # Pass distance to BikeArrival
                    )
                )
 
            # remove bike from the departure station
            departure_station.remove_bike(bike)
 
            simul.state.set_bike_in_use(bike)
 
            # Print bike departure information
            #maint_status = f"maint={bike.maintenance_criticality:.3f}" if hasattr(bike, 'maintenance_criticality') else ""
            
            # Extra logging for S51

            '''if departure_station.id == "S51":
                print(bike.bike_id)
                print(f"  [S51 DEPARTURE] Bike {bike.bike_id} -> {arrival_station_id} (t={self.time:.1f}, travel={travel_time:.1f}min)")'''

            # print all bike departures for debugging
            #print(f"  DEPARTURE:  Bike {bike.bike_id} from {departure_station.id} -> to {arrival_station_id} (t={self.time:.1f}, travel={travel_time:.1f}min)")
            

            #print(f"  DEPARTURE: Bike {bike.bike_id} from {departure_station.id} -> to {arrival_station_id} "
                  #f"(t={self.time:.1f}, {maint_status}, travel={travel_time:.1f}min)")
            
            # Log bike movement
            try:
                bike_crit = getattr(bike, 'maintenance_criticality', 0.0)
                simul.log_bike_movement(
                    time=self.time,
                    bike_id=bike.bike_id,
                    departure_station_id=departure_station.id,
                    arrival_station_id=arrival_station_id,
                    did_roam=False,
                    bike_criticality=bike_crit
                )
            except AttributeError as e:
                print(f"ERROR: Cannot log bike movement: {e}, simul type: {type(simul)}")
            
            # Log successful trip request
            try:
                bike_crit = getattr(bike, 'maintenance_criticality', 0.0)
                simul.log_trip_request(
                    time=self.time,
                    station_id=departure_station.id,
                    success=True,
                    failure_reason=None,
                    did_roam=False,
                    arrival_station_id=arrival_station_id,
                    travel_time=travel_time,
                    bike_id=bike.bike_id,
                    bike_criticality=bike_crit
                )
            except AttributeError as e:
                print(f"ERROR: Cannot log trip request: {e}, simul type: {type(simul)}")
 
            simul.state.metrics.add_aggregate_metric(simul.state, "bike departure", 1)
            simul.state.metrics.add_aggregate_metric(simul.state, "events", 2)
 
        else:
            if FULL_TRIP:
                closest_neighbours = simul.state.get_neighbouring_stations(departure_station, 1, not_empty=True)
                
                # --- THE FIX: Safety check for absolute starvation ---
                if not closest_neighbours:
                    # Treat as standard lost trip, bail out early
                    simul.state.metrics.add_aggregate_metric(simul.state, "bike starvations", 1)
                    simul.state.metrics.add_aggregate_metric(simul.state, "events", 1)
                    simul.state.metrics.add_aggregate_metric(simul.state, "starvations", 1)
                    simul.state.metrics.add_aggregate_metric(simul.state, "failed events", 1)
                    return # Exit the perform() method cleanly
                    
                closest_neighbour_with_bikes = closest_neighbours[0]
                
                distance = departure_station.distance_to(closest_neighbour_with_bikes.get_lat(), closest_neighbour_with_bikes.get_lon())
                mp=departure_station.get_move_probabilities(simul.state, simul.state.day(), simul.state.hour())
                p = list(mp.values())
                sum = 0.0
                for i in range(len(p)):
                    sum += p[i]
                p_normalized = []
                for i in range(len(p)):
                    if sum > 0:
                        p_normalized.append(p[i] * (1.0/sum))
                    else:
                        p_normalized.append(1/len(p))
                if self.acceptance_rejection(distance, simul):
                    available_bikes = closest_neighbour_with_bikes.get_available_bikes()
                    bike=available_bikes.pop(0)
                    
                    if simul.state.rng.random() < RANDOM_DESTINATION_PROB:
                        # Exclude the current area from the random selection
                        other_stations = [station.id for station in simul.state.get_stations() if station.id != self.departure_station_id]
                        arrival_station_id = simul.state.rng.choice(other_stations)
                    else:
                        arrival_station_id = simul.state.rng.choice(list(mp.keys()), p = p_normalized)
                    
                    arrival_station = simul.state.get_location_by_id(arrival_station_id)
 
                    travel_time = simul.state.get_travel_time(
                        closest_neighbour_with_bikes.id,
                        arrival_station.id,) + simul.state.get_travel_time(departure_station.id,
                        closest_neighbour_with_bikes.id)*(BIKE_SPEED/WALKING_SPEED)
                    #total travel time, roaming for bike from departure station to neighbour + cycling to arrival station
 
                    # NEW: Calculate distance for roaming case
                    distance_km = 0.0
                    if ENABLE_COMPONENT_FAILURES:
                        distance_km = closest_neighbour_with_bikes.distance_to(
                            arrival_station.lat, 
                            arrival_station.lon
                        )

                        # NEW: Evaluate component failure risk for roaming case
                        self._evaluate_trip_failure_risk(simul, bike, distance_km,
                                                        closest_neighbour_with_bikes.id, arrival_station_id)
 
                    # create an arrival event for the roaming user from the new departure station
                    simul.add_event(
                        sim.BikeArrival(
                            self.time,
                            travel_time,
                            bike,
                            arrival_station.id,
                            closest_neighbour_with_bikes.id,
                            distance_km=distance_km  # NEW: Pass distance to BikeArrival
                        )
                    )
 
                    # remove bike from the new departure station
                    closest_neighbour_with_bikes.remove_bike(bike)
 
                    simul.state.set_bike_in_use(bike)
                    
                    # Print roaming for bike
                    maint_status = f"maint={bike.maintenance_criticality:.3f}" if hasattr(bike, 'maintenance_criticality') else ""

                    '''
                    print(f"   ROAMING DEP: User walks from {departure_station.id} to {closest_neighbour_with_bikes.id} "
                          f"(dist={distance:.2f}km)  bike {bike.bike_id} -> {arrival_station_id} "
                          f"(t={self.time:.1f}, {maint_status})")                    
                    
                    '''

                    
                    # Log bike movement if simulator supports it
                    if hasattr(simul, 'log_bike_movement'):
                        bike_crit = getattr(bike, 'maintenance_criticality', 0.0)
                        simul.log_bike_movement(
                            time=self.time,
                            bike_id=bike.bike_id,
                            departure_station_id=closest_neighbour_with_bikes.id,
                            arrival_station_id=arrival_station_id,
                            did_roam=True,
                            bike_criticality=bike_crit
                        )
                    
                    # Log successful roaming trip request
                    if hasattr(simul, 'log_trip_request'):
                        bike_crit = getattr(bike, 'maintenance_criticality', 0.0)
                        simul.log_trip_request(
                            time=self.time,
                            station_id=departure_station.id,
                            success=True,
                            failure_reason=None,
                            did_roam=True,
                            arrival_station_id=arrival_station_id,
                            travel_time=travel_time,
                            bike_id=bike.bike_id,
                            bike_criticality=bike_crit
                        )

                    simul.state.metrics.add_aggregate_metric(simul.state, "bike departure", 1)
                    simul.state.metrics.add_aggregate_metric(simul.state, "events", 2)
 
                    simul.state.metrics.add_aggregate_metric(simul.state, "roaming for bikes", 1)
                    simul.state.metrics.add_aggregate_metric(simul.state, "roaming distance for bikes", distance)
 
                else:
                    # Print lost trip information
                    # checkpoint
                    #if departure_station.number_of_bikes() <= 0:
                    if total_bikes_at_station <= 0:
                        #print(f"  LOST TRIP: No bikes at {departure_station.id} (bike starvation, t={self.time:.1f})")
                        simul.state.metrics.add_aggregate_metric(simul.state, "bike starvations", 1)
                        
                        # Log failed trip - bike starvation
                        if hasattr(simul, 'log_trip_request'):
                            simul.log_trip_request(
                                time=self.time,
                                station_id=departure_station.id,
                                success=False,
                                failure_reason='bike_starvation',
                                did_roam=False
                            )
                    else:
                        unusable_count = len(departure_station.get_unusable_bikes())
                        #print(f"  LOST TRIP: No usable bikes at {departure_station.id}, unusabe bikes are {departure_station.get_unusable_bikes()}, total_bikes_at_station are {departure_station.number_of_bikes()} (t={self.time:.1f})")

                        if unusable_count == total_bikes_at_station:
                            #print(f"  LOST TRIP: All bikes at {departure_station.id} require maintenance "
                            #      f"(maintenance starvation, t={self.time:.1f})")
                            
                            # Log failed trip - maintenance starvation
                            if hasattr(simul, 'log_trip_request'):
                                simul.log_trip_request(
                                    time=self.time,
                                    station_id=departure_station.id,
                                    success=False,
                                    failure_reason='maintenance_starvation',
                                    did_roam=False
                                )
                            
                            all_stations = simul.state.get_stations()

                            for station in sorted(all_stations, key=lambda s: s.id):
                                total = station.number_of_bikes()
                                usable_bikes = station.get_available_bikes()
                                unusable_bikes = station.get_unusable_bikes()
                                num_usable = len(usable_bikes)
                                num_unusable = len(unusable_bikes)
                                
                                # Get demand for this station at current time
                                demand = 0
                                if hasattr(station, 'get_demand'):
                                    demand = station.get_demand(simul.state.day(), simul.state.hour())
                                
                                # Get criticalities of unusable bikes
                                criticalities = []
                                for bike in unusable_bikes:
                                    if hasattr(bike, 'maintenance_criticality'):
                                        criticalities.append(f"{bike.maintenance_criticality:.3f}")
                                
                                crit_str = ", ".join(criticalities) if criticalities else "N/A"
                                if len(crit_str) > 50:
                                    crit_str = crit_str[:47] + "..."
                                
                                #print(f"{station.id:<10} {total:<7} {num_usable:<8} {num_unusable:<10} {demand:<8.2f} {crit_str}")
                            
                            print("=" * 100)
                            print()
                            
                            simul.state.metrics.add_aggregate_metric(simul.state, "battery starvations", 1)
                            simul.state.metrics.add_aggregate_metric(simul.state, "maintenance_starvation", 1)
                        else:
                            #print(f"  LOST TRIP: No usable bikes at {departure_station.id} "
                                 # f"({unusable_count} unusable, t={self.time:.1f})")
                            simul.state.metrics.add_aggregate_metric(simul.state, "battery starvations", 1)
                            
                            # Log failed trip - maintenance starvation (partial)
                            """if hasattr(simul, 'log_trip_request'):
                                simul.log_trip_request(
                                    time=self.time,
                                    station_id=departure_station.id,
                                    success=False,
                                    failure_reason='maintenance_starvation',
                                    did_roam=False
                                )"""

                    simul.state.metrics.add_aggregate_metric(simul.state, "events", 1)
                    simul.state.metrics.add_aggregate_metric(simul.state, "starvations", 1)
                    simul.state.metrics.add_aggregate_metric(simul.state, "failed events", 1)
                    
        simul.state.metrics.add_aggregate_metric(simul.state, "trips", 1)

 
    
    def _evaluate_trip_failure_risk(self, simul, bike, trip_distance_km, departure_id, arrival_id):
        """
        Evaluate component failure risk using ComponentFailureModel.
        """
        verbose = VERBOSE_FAILURE_TRACKING or (bike.bike_id in SAMPLE_BIKES_TO_TRACK)
        
        failed_components = ComponentFailureModel.check_component_failures_on_trip(
            bike, trip_distance_km, simul.state.rng, verbose=verbose
        )
        
        # Process any failures
        for category, failure_info in failed_components.items():
            self._trigger_component_failure(
                simul, bike, category, 
                failure_info['failure_probability'],
                failure_info['hazard_rate'],
                departure_id, arrival_id,
                failure_info['component_odometer']
            )



    def _trigger_component_failure(self, simul, bike, category, probability, hazard_rate, 
                                    departure_id, arrival_id, component_km):
        """
        Trigger component failure with SIMPLE 50/50 severity classification.
        """
        
        # === SIMPLIFIED: 50/50 DEPOT vs ON-SITE ===
        damage_label = assign_damage_severity(simul.state.rng, category)
        
        # Increment failure counter
        if 'total_failures' not in bike.component_failures[category]:
            bike.component_failures[category]['total_failures'] = 0
        bike.component_failures[category]['total_failures'] += 1
        
        # === LOG FAILURE ===
        print("\n" + "!"*80)
        print(f"COMPONENT FAILURE - SEVERITY: {damage_label}")
        print(f"  Bike: {bike.bike_id}")
        print(f"  Component: {category}")
        print(f"  Odometer: {bike.total_distance_km:.1f} km")
        print(f"  Weibull failure probability: {probability:.6f}")
        print("!"*80 + "\n")
        
        # === LOG TO SIMULATOR ===
        if hasattr(simul, 'log_component_failure'):
            simul.log_component_failure(
                time=self.time,
                bike_id=bike.bike_id,
                component_category=category,
                damage_severity=damage_label,
                odometer_km=bike.total_distance_km,
                component_odometer_km=component_km,
                failure_probability=probability,
                departure_station=departure_id,
                arrival_station=arrival_id
            )
        
        # === HANDLE BASED ON DAMAGE LABEL ===
        if damage_label == "depot":
            # DEPOT FIX: Will be removed from service upon arrival
            bike.pending_depot_fix = True  # NEW FLAG
            bike.pending_failure_category = category
            bike.last_failure_time = self.time
            
            # Update metrics
            simul.state.metrics.add_aggregate_metric(simul.state, "depot_failures", 1)
            simul.state.metrics.add_aggregate_metric(simul.state, "component_failures", 1)
            simul.state.metrics.add_aggregate_metric(simul.state, f"failure_{category}", 1)
            simul.state.metrics.add_aggregate_metric(simul.state, f"failure_{category}_depot", 1)
            
            print(f"[DEPOT FIX PENDING] Bike {bike.bike_id} will be REMOVED FROM SERVICE upon arrival\n")
            return 'trip_completes_then_unavailable'
        
        else:  # "damaged: on-site fix"
            # ON-SITE FIX: Will be flagged for inspection upon arrival
            bike.pending_onsite_fix = True  # NEW FLAG
            bike.pending_failure_category = category
            
            # Update metrics
            simul.state.metrics.add_aggregate_metric(simul.state, "onsite_failures", 1)
            simul.state.metrics.add_aggregate_metric(simul.state, "component_failures", 1)
            simul.state.metrics.add_aggregate_metric(simul.state, f"failure_{category}", 1)
            simul.state.metrics.add_aggregate_metric(simul.state, f"failure_{category}_onsite", 1)
            
            print(f"[ON-SITE FIX PENDING] Bike {bike.bike_id} will be FLAGGED upon arrival (still rentable)\n")
            return 'trip_completes_normally'



    def __repr__(self):
        return f"<{self.__class__.__name__} at time {self.time}, departing from station {self.departure_station_id}>"
    
    def acceptance_rejection(self,distance, simul):
        prob_acceptance = -1.6548*distance**2-0.7036*distance+1.0133
        random_roaming_limit = simul.state.rng.uniform(0,1)
        if random_roaming_limit <= prob_acceptance:
            return True
        else:
            return False
 

