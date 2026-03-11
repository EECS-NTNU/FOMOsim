from .bike_component_maintenance_model import ComponentMaintenanceManager
from settings import SAMPLE_BIKES_TO_TRACK

class BaselineRepairPolicy:
    """
    Baseline repair policy: Repairs all component failures immediately upon arrival.
    
    This policy simply calls ComponentMaintenanceManager functions immediately.
    Future policies will call the same functions at different times/conditions.
    """
    
    def __init__(self):
        self.name = "BaselineImmediateRepair"
    
    def decide_and_execute(self, bike, simul, arrival_station_id, arrival_time, verbose=False):
        """
        Baseline decision: Always repair immediately.
        
        Calls the SAME ComponentMaintenanceManager functions that vFA will use later.
        The only difference is WHEN and under WHAT CONDITIONS we call them.
        """
        
        # === DEPOT FAILURES ===
        if hasattr(bike, 'pending_depot_fix') and bike.pending_depot_fix:
            category = bike.pending_failure_category
            
            print(f"\n[BASELINE POLICY] Bike {bike.bike_id} - Repairing {category} (depot)")

            # print component odometers after fix
            print(f"  Pre-inspection component odometers for Bike {bike.bike_id}:")
            for comp_cat, odometer in bike.component_odometers.items():
                print(f"    {comp_cat}: {odometer:.2f} km")
            
            # Call the same function vFA will use later
            ComponentMaintenanceManager.perform_depot_repair(bike, verbose=verbose)

            # print component odometers after fix
            print(f"  Post-inspection component odometers for Bike {bike.bike_id}:")
            for comp_cat, odometer in bike.component_odometers.items():
                print(f"    {comp_cat}: {odometer:.2f} km")
            
            # Track as baseline decision
            simul.state.metrics.add_aggregate_metric(simul.state, "baseline_depot_repairs", 1)
        
        # === ON-SITE FAILURES ===
        elif hasattr(bike, 'pending_onsite_fix') and bike.pending_onsite_fix:
            category = bike.pending_failure_category
            

            print(f"\n[BASELINE POLICY] Bike {bike.bike_id} - Inspecting {category} (on-site)")

            # print component odometers after fix
            print(f"  Pre-inspection component odometers for Bike {bike.bike_id}:")
            for comp_cat, odometer in bike.component_odometers.items():
                print(f"    {comp_cat}: {odometer:.2f} km")
            
            # Call the same function vFA will use later
            ComponentMaintenanceManager.perform_onsite_inspection(bike, verbose=verbose)

            # print component odometers after fix
            print(f"  Post-inspection component odometers for Bike {bike.bike_id}:")
            for comp_cat, odometer in bike.component_odometers.items():
                print(f"    {comp_cat}: {odometer:.2f} km")
            
            # Track as baseline decision
            simul.state.metrics.add_aggregate_metric(simul.state, "baseline_onsite_repairs", 1)


class VFARepairPolicy:
    """
    vFA-based repair policy (FUTURE)
    
    Will call the SAME ComponentMaintenanceManager functions,
    but based on learned value function estimates.
    """
    
    def __init__(self, vfa_agent):
        self.name = "VfaBased"
        self.vfa_agent = vfa_agent
    
    def decide_and_execute(self, bike, simul, arrival_station_id, arrival_time, verbose=False):
        """
        vFA decision: Repair based on value estimates.
        
        Uses the SAME repair functions, just different decision logic.
        """
        
        # Check for pending failures
        has_depot = hasattr(bike, 'pending_depot_fix') and bike.pending_depot_fix
        has_onsite = hasattr(bike, 'pending_onsite_fix') and bike.pending_onsite_fix
        
        if not (has_depot or has_onsite):
            return
        
        category = bike.pending_failure_category
        
        # === EXTRACT STATE FEATURES ===
        state_features = self._extract_state(bike, simul, arrival_station_id, arrival_time)
        
        # === QUERY vFA FOR DECISION ===
        action = self.vfa_agent.choose_action(state_features)
        # action could be: 'repair_now', 'defer', 'replace_component', etc.
        
        # === EXECUTE USING SAME MAINTENANCE FUNCTIONS ===
        if action == 'repair_now':
            if has_depot:
                # Same function as baseline uses
                ComponentMaintenanceManager.perform_depot_repair(bike, verbose=verbose)
                simul.state.metrics.add_aggregate_metric(simul.state, "vfa_depot_repairs", 1)
            elif has_onsite:
                # Same function as baseline uses
                ComponentMaintenanceManager.perform_onsite_inspection(bike, verbose=verbose)
                simul.state.metrics.add_aggregate_metric(simul.state, "vfa_onsite_repairs", 1)
        
        elif action == 'defer':
            if verbose:
                print(f"[vFA POLICY] Bike {bike.bike_id} - DEFERRING repair of {category}")
            simul.state.metrics.add_aggregate_metric(simul.state, "vfa_deferred_repairs", 1)
            # Bike keeps its pending flags, will be processed later
        
        # === UPDATE vFA (learning) ===
        reward = self._calculate_reward(bike, simul, action)
        self.vfa_agent.update(state_features, action, reward)
    
    def _extract_state(self, bike, simul, station_id, time):
        """Extract state features for vFA decision"""
        return {
            'bike_id': bike.bike_id,
            'component': bike.pending_failure_category,
            'component_odometer': bike.component_odometers[bike.pending_failure_category],
            'total_odometer': bike.total_distance_km,
            'station_id': station_id,
            'station_occupancy': simul.state.get_location_by_id(station_id).current_bikes,
            'time_of_day': time % (24 * 60),  # minutes in day
            'is_depot': bike.pending_depot_fix,
            # Add more features as needed
        }
    
    def _calculate_reward(self, bike, simul, action):
        """Calculate reward for the repair decision"""
        # TODO: Implement reward function
        # Could be based on: repair cost, downtime, future failure probability, etc.
        return 0.0


