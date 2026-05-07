from settings import SAMPLE_BIKES_TO_TRACK

class ComponentMaintenanceManager:
    """
    Manages component repairs, odometer resets, and maintenance operations.

    This class provides static methods to perform repairs and reset component-specific odometers.

    It is designed to be called by different repair policies (e.g., BaselineRepairPolicy, VFARepairPolicy)
    to ensure consistent repair logic and tracking across policies.
    """
    
    @staticmethod
    def repair_component(bike, category, repair_type="depot", verbose=False):
        """
        Repair a specific component and reset its odometer.
        
        Args:
            bike: Bike object
            category: Component category to repair
            repair_type: "depot" or "onsite"
            verbose: Print repair information
        
        Returns:
            bool: True if repair successful
        """
        if category not in bike.component_failures:
            return False
        
        # Update failure statistics
        bike.component_failures[category]['total_failures'] += 1
        bike.component_failures[category]['last_failure_km'] = bike.total_distance_km
        
        if repair_type == "depot":
            bike.component_failures[category]['depot_fixes'] += 1
        else:
            bike.component_failures[category]['onsite_fixes'] += 1
        
        # Reset component-specific odometer
        ComponentMaintenanceManager.reset_component_odometer(bike, category, verbose=verbose)
        
        return True
    
    @staticmethod
    def reset_component_odometer(bike, category, verbose=False):
        """
        Reset the odometer for a specific component after repair/replacement.
        
        Args:
            bike: Bike object
            category: Component category to reset
            verbose: Print reset information
        """
        if category in bike.component_odometers:
            old_value = bike.component_odometers[category]
            bike.component_odometers[category] = 0.0
            
            '''if verbose and (bike.bike_id in SAMPLE_BIKES_TO_TRACK):
                print(f"\n COMPONENT REPAIRED on Bike {bike.bike_id}")
                print(f"   Component: {category}")
                print(f"   Odometer reset: {old_value:.2f} km -> 0.0 km")
                print(f"   Total bike odometer: {bike.total_distance_km:.2f} km (unchanged)")
                print(f"   Other components continue accumulating usage\n")'''
            # Always print for debug
            print(f"[DEBUG] After odometer reset: Bike {bike.bike_id} {category} odometer = {bike.component_odometers[category]:.2f} km (was {old_value:.2f} km)")
    
    @staticmethod
    def perform_depot_repair(bike, verbose=False):
        """
        Perform depot repair on a bike with pending failure.
        
        Args:
            bike: Bike object with pending_depot_fix flag
            verbose: Print repair information
        
        Returns:
            str: Category of repaired component, or None
        """
        if not hasattr(bike, 'pending_failure_category') or not bike.pending_failure_category:
            return None
        
        category = bike.pending_failure_category
        
        # Perform the repair
        ComponentMaintenanceManager.repair_component(bike, category, repair_type="depot", verbose=verbose)
        
        # Clear damage status
        bike.damage_status = None
        bike.is_available = True
        bike.needs_maintenance = False
        bike.pending_failure_category = None
        bike.pending_depot_fix = False
        
        if verbose and (bike.bike_id in SAMPLE_BIKES_TO_TRACK):
            print(f"[DEPOT REPAIR COMPLETE] Bike {bike.bike_id} - {category} repaired")
            print(f"  Bike returned to service\n")
        
        return category
    
    @staticmethod
    def perform_onsite_inspection(bike, verbose=False):
        """
        Perform on-site inspection/repair on a bike.
        
        Args:
            bike: Bike object with pending_onsite_fix flag
            verbose: Print repair information
        
        Returns:
            str: Category of inspected component, or None
        """
        if not hasattr(bike, 'pending_failure_category') or not bike.pending_failure_category:
            return None
        
        category = bike.pending_failure_category
        
        # Perform the inspection/minor repair
        ComponentMaintenanceManager.repair_component(bike, category, repair_type="onsite", verbose=verbose)
        ComponentMaintenanceManager.clear_damage_status(bike)
        
        # Clear inspection flag
       # bike.damage_status = None
        #bike.needs_inspection = False
       # bike.pending_failure_category = None
       # bike.pending_onsite_fix = False
      
        return category
    
    @staticmethod
    def clear_damage_status(bike):
        """
        Clear all damage-related status flags on the bike after repair.
        
        Args:
            bike: Bike object
        """
        bike.damage_status = None
        bike.is_available = True
        bike.needs_maintenance = False
        bike.needs_inspection = False
        bike.last_failure_time = None
        bike.last_failure_category = None
        bike.pending_depot_fix = False
        bike.pending_onsite_fix = False
        bike.pending_failure_category = None