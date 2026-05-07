
from sim.Location import Location
from sim.Metric import Metric
from settings import *
from sim.bike_degradation_modeling import damage_configuration
from sim.bike_degradation_modeling.bike_component_maintenance_model import ComponentMaintenanceManager

class Bike(Location):
    """
    Bike class - manages bike state and basic operations.
    Complex degradation logic is handled by external models.
    """
    
    # Class-level set to keep track of all unique bike IDs created in the system.
    # This neatly bypasses having to search through depots, vehicles and stations.
    created_bike_ids = set()

    @classmethod
    def reset_bike_tracker(cls):
        """Reset the bike tracker between separate simulation runs."""
        cls.created_bike_ids = set()

    def __init__(self, 
                 is_station_based, 
                 lat: float = 0, 
                 lon: float = 0, 
                 location_id = 0, 
                 bike_id = 0):
        super().__init__(lat, lon, location_id)
        self.is_station_based = is_station_based
        self.metrics = Metric()
        self.bike_id = bike_id
        Bike.created_bike_ids.add(bike_id)
        self.battery = 100.0
        self.log = []

        # DAMAGE TRACKING - STATE ONLY
        self.total_distance_km = 0.0
        self.component_failures = {}
        self.component_odometers = {}
        self.damage_status = None  # "depot" or "onsite" or None
        self.is_available = True
        self.needs_maintenance = False
        self.needs_inspection = False
        self.last_failure_time = None
        self.last_failure_category = None
        
        # Pending failure flags (set during trip, processed on arrival)
        self.pending_depot_fix = False
        self.pending_onsite_fix = False
        self.pending_failure_category = None

        if ENABLE_COMPONENT_FAILURES:
            self._initialize_component_failures()

    def _initialize_component_failures(self, verbose=False):
        """Initialize component failure tracking structures."""
        self.component_failures = {}
        self.component_odometers = {}

        '''if verbose and (self.bike_id in SAMPLE_BIKES_TO_TRACK):
            print(f"\n{'='*100}")
            print(f"INITIALIZING BIKE {self.bike_id} - COMPONENT TRACKING")
            print(f"{'='*100}")
            print(f"{'Component':<30} {'Scale(λ)':<15} {'Shape(k)':<12} {'MTTF(km)':<15}")
            print(f"{'-'*100}")'''

        for category, params in damage_configuration.DAMAGE_CATEGORIES.items():
            self.component_failures[category] = {
                'scale': params['scale'],
                'shape': params['shape'],
                'mttf_km': params['mttf_km'],
                'total_failures': 0,
                'last_failure_km': 0.0,
                'depot_fixes': 0,
                'onsite_fixes': 0,
            }
            self.component_odometers[category] = 0.0

            if verbose and (self.bike_id in SAMPLE_BIKES_TO_TRACK):
                print(f"{category:<30} {params['scale']:<15.1f} {params['shape']:<12.2f} {params['mttf_km']:<15.1f}")
                
        if verbose and (self.bike_id in SAMPLE_BIKES_TO_TRACK):
            print(f"{'='*100}\n")

    def travel(self, simul, travel_time, distance_km=0.0, congested=False):
        """
        Update bike state after travel.
        
        :param simul: Simulator object
        :param travel_time: Duration of travel in minutes
        :param distance_km: Actual distance traveled in kilometers
        :param congested: Whether this trip was congested
        """
        if congested:
            self.metrics.add_metric(simul.state, "travel_time_congested", travel_time)
        else:
            self.metrics.add_metric(simul.state, "travel_time", travel_time)
        
        # Update total distance
        self.total_distance_km += distance_km

        # Update each component's odometer
        if ENABLE_COMPONENT_FAILURES and hasattr(self, 'component_odometers'):
            for category in self.component_odometers:
                self.component_odometers[category] += distance_km

    def usable(self):
        """Check if bike can be rented."""
        if not getattr(self, "is_available", True):
            return False
        if getattr(self, "damage_status", None) is not None:
            return False
        return True

    def hasBattery(self):
        return False
    
    def needsMaintenance(self):
        """Check if bike needs any form of maintenance."""
        return hasattr(self, 'damage_status') and self.damage_status is not None

    def clear_damage(self):
        """Clear all damage-related flags. Use ComponentMaintenanceManager instead."""
        
        ComponentMaintenanceManager.clear_damage_status(self)

    def __repr__(self):
        return f"ID-{self.bike_id}-{self.lat}-{self.lon}"
