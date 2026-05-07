import sim
from sim import Event
from settings import *
from settings import MAINTENANCE_LIMIT_TO_CHECK
#from sim.bike_degradation_modeling.maintenance_model import update_bike_maintenance
from sim.bike_degradation_modeling.bike_component_maintenance_model import ComponentMaintenanceManager
#from sim.bike_degradation_modeling.repair_policy import BaselineRepairPolicy
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
        Enhanced to track trip outcomes and generate HMM observations.
        
        :param simul: Simulation object
        """

        super().perform(simul)

        # get arrival station 
        arrival_station = simul.state.get_location_by_id(self.arrival_station_id)

        if not FULL_TRIP:
            self.bike = simul.state.get_used_bike()

        if self.bike is not None:
            # === UPDATE BIKE STATE (travel consumes battery, checks for failures) ===
            self.bike.travel(simul, self.travel_time, distance_km=self.distance_km, congested=self.congested)

            # === PROCESS PENDING COMPONENT FAILURES ===
            self._process_pending_failures(simul)
            #self._process_component_failures(simul)
            
            # === DETERMINE TRIP OUTCOME FOR HMM ===
            #trip_result = 'completed'  # Default
            
            # Check for battery violation
            if self.bike.battery < 0:
                trip_result = 'failed_battery'
                simul.state.metrics.add_aggregate_metric(simul.state, "battery violations", 1)
                simul.state.metrics.add_aggregate_metric(simul.state, "failed events", 1)
                self.bike.battery = 0
            
            # Check if trip was degraded (moderate failure flag from BikeDeparture)
            elif hasattr(self.bike, 'failure_severity') and self.bike.failure_severity == 'moderate':
                trip_result = 'completed_degraded'
                '''if VERBOSE_FAILURE_TRACKING or (self.bike.bike_id in SAMPLE_BIKES_TO_TRACK):
                    print(f"[DEGRADED TRIP] Bike {self.bike.bike_id} completed with degraded performance")'''
            
            # Check for congestion
            elif self.congested:
                trip_result = 'completed_congested'
            

            # === PROCESS ARRIVAL ===
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
                    
                    '''if maintenance_enabled:
                        try:
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
                            traceback.print_exc()'''

                # Track maintenance violations
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
                # === STATION FULL - BIKE MUST ROAM ===
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
                    #maint_status = f"maint={self.bike.maintenance_criticality:.3f}" if hasattr(self.bike, 'maintenance_criticality') else ""
                    #print(f"   ROAMING: Bike {self.bike.bike_id} - {arrival_station.id} FULL -> routing to {next_station.id} from {arrival_station.id} "
                         # f"(t={self.time:.1f}, {maint_status}, +{travel_time:.1f}min, +{roaming_distance_km:.2f}km)")

                    # create an arrival event for the departed bike
                    simul.add_event(
                        BikeArrival(
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

    '''def _process_component_failures(self, simul):
        """
        Process pending component failures using the configured repair policy.
        For bikes that have pending failures, we will flag them for repair and potentially remove them from service.
        """
        verbose = VERBOSE_FAILURE_TRACKING or (self.bike.bike_id in SAMPLE_BIKES_TO_TRACK)
        
        # Check if bike has any pending failures
        has_pending = (
            (hasattr(self.bike, 'pending_depot_fix') and self.bike.pending_depot_fix) or
            (hasattr(self.bike, 'pending_onsite_fix') and self.bike.pending_onsite_fix)
        )
        
        if not has_pending:
            return
        
        # Get the configured policy (baseline for now, vFA later)
        repair_policy = self._get_repair_policy(simul)
        
        # Delegate decision and execution to policy
        repair_policy.decide_and_execute(
            bike=self.bike,
            simul=simul,
            arrival_station_id=self.arrival_station_id,
            arrival_time=self.time,
            verbose=verbose
        )'''

    '''def _get_repair_policy(self, simul):
        """
        Get the repair policy to use.
        Uses VFA agent if available, otherwise falls back to BaselineRepairPolicy.
        """
        if hasattr(simul, 'vfa_agent') and simul.vfa_agent is not None:
            print("Using VFA agent for component failure processing.")
            return simul.vfa_agent
        else:
            print("No VFA agent found in simulation, using BaselineRepairPolicy for component failure processing.")
            return BaselineRepairPolicy()'''

    def _process_pending_failures(self, simul):
        """
        Process component failures that were predicted during the trip (BikeDeparture logic) and update bike state accordingly.
        This is separate from the actual repair decision logic
        
        """
        
        # Check for pending depot fix
        if hasattr(self.bike, 'pending_depot_fix') and self.bike.pending_depot_fix:
            category = self.bike.pending_failure_category
            
            print(f"\n{'='*80}")
            print(f"[BIKE ARRIVAL - DEPOT FIX TRIGGERED]")
            print(f"  Bike {self.bike.bike_id} arrived at {self.arrival_station_id}")
            print(f"  Component failed during trip: {category}")
            print(f"  Total bike odometer: {self.bike.total_distance_km:.1f} km")
            print(f"  Component odometer: {self.bike.component_odometers[category]:.1f} km")
            print(f"  ACTION: Bike REMOVED FROM SERVICE")
            print(f"{'='*80}\n")
            
            # REMOVE BIKE FROM SERVICE
            self.bike.is_available = False
            #print(f" !!!!!!!!!!!!!!!!!!! Bike {self.bike.bike_id} flagged and set as {self.bike.is_available} due to {category} failure")
            self.bike.needs_maintenance = True
            self.bike.damage_status = "depot"
            self.bike.last_failure_category = category
            
            # Clear pending flags
            self.bike.pending_depot_fix = False
            #self.bike.pending_failure_category = None
        
        # Check for pending on-site fix
        elif hasattr(self.bike, 'pending_onsite_fix') and self.bike.pending_onsite_fix:
            category = self.bike.pending_failure_category
            
            '''print(f"\n{'='*80}")
            print(f"[BIKE ARRIVAL - ON-SITE FIX TRIGGERED]")
            print(f"  Bike {self.bike.bike_id} arrived at {self.arrival_station_id}")
            print(f"  Component failed during trip: {category}")
            print(f"  Total bike odometer: {self.bike.total_distance_km:.1f} km")
            print(f"  Component odometer: {self.bike.component_odometers[category]:.1f} km")
            print(f"  ACTION: Bike FLAGGED for on-site repair (REMOVED FROM SERVICE)")
            print(f"{'='*80}\n")'''
            
            # REMOVE BIKE FROM SERVICE UNTIL REPAIRED
            self.bike.is_available = False
            self.bike.needs_inspection = True
            self.bike.damage_status = "onsite"
            self.bike.last_failure_category = category
            
            # Clear pending flags
            self.bike.pending_onsite_fix = False
            #self.bike.pending_failure_category = None

    def __repr__(self):
        return f"<{self.__class__.__name__} at time {self.time}, arriving at station {self.arrival_station_id}>"
 