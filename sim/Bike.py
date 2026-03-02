from sim.Location import Location
from sim.Metric import Metric
from settings import *
#from sim.maintenance_model import update_bike_maintenance
import numpy as np
from sim.bike_degradation_modeling import damage_configuration

class Bike(Location):
    """
    Bike class containing state and all operations necessary
    """

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
        self.battery = 100.0
        #self.maintenance_criticality= 0.0
        self.log = []

        ####
        self.total_distance_km = 0.0
        self.component_failures = {}

        if ENABLE_COMPONENT_FAILURES:
            self._initialize_component_failures()

    def _initialize_component_failures(self, verbose=False):
        """Initialize failure distances for each component using Weibull distribution"""

        self.component_failures = {}
    
        if verbose and (self.bike_id in SAMPLE_BIKES_TO_TRACK):
            print(f"\n{'='*100}")
            print(f"INITIALIZING BIKE {self.bike_id} - WEIBULL CONDITIONAL FAILURE MODEL")
            print(f"{'='*100}")
            print(f"{'Component':<30} {'Scale(lambda)':<15} {'Shape(k)':<12} {'MTTF(km)':<15}")
            print(f"{'-'*100}")

        for category, params in damage_configuration.DAMAGE_CATEGORIES.items():
            self.component_failures[category] = {
                'scale': params['scale'],      # λ (lambda) - characteristic life
                'shape': params['shape'],      # k - wear rate (>1 means wear-out)
                'mttf_km': params['mttf_km'],  # mean time to failure
                'total_failures': 0,
                'last_failure_km': 0.0,
            # ADD THIS:
            'severity_counts': {
                'critical': 0,
                'moderate': 0,
                'minor': 0
            }
            }
            
            if verbose:
                print(f"{category:<30} {params['scale']:<15.1f} {params['shape']:<12.2f} {params['mttf_km']:<15.1f}")
                
        if verbose:
            print(f"{'='*100}")
            print(f"Conditional Failure Probability Model:")
            print(f"  Formula: P(fail on trip) = 1 - exp[(L/lambda)^k - ((L+d)/lambda)^k]")
            print(f"  Where:")
            print(f"    L = current odometer (km)")
            print(f"    d = trip distance (km)")
            print(f"    lambda = scale parameter (characteristic life)")
            print(f"    k = shape parameter (wear rate)")
            print(f"  ")
            print(f"  Interpretation:")
            print(f"    - Failure probability increases with bike age (L)")
            print(f"    - k > 1: Wear-out failures (aging components)")
            print(f"    - k = 1: Constant failure rate (random failures)")
            print(f"    - k < 1: Infant mortality (early-life failures)")
            print(f"  ")
            print(f"  On each trip:")
            print(f"    1. Calculate P(fail) based on current odometer L and trip distance d")
            print(f"    2. Draw random u ~ Uniform(0,1)")
            print(f"    3. If u < P(fail), component fails on this trip")
            print(f"{'='*100}\n")
    
    '''def calculate_trip_failure_probability(self, current_km, trip_distance_km, scale, shape):
        """
        Calculate probability of failure during a specific trip using conditional reliability.
        
        Formula: P(L < T <= L+d | T > L) = [R(L) - R(L+d)] / R(L)
        Simplified: P = 1 - exp[(L/λ)^k - ((L+d)/λ)^k]
        
        Args:
            current_km: L - current lifetime odometer reading
            trip_distance_km: d - distance of this trip
            scale: λ (lambda) - characteristic life
            shape: k - shape parameter
        
        Returns:
            Probability of failure during this trip (0 to 1)
        """
        L = current_km
        d = trip_distance_km
        lam = scale
        k = shape
        
        # Handle edge case: if bike hasn't traveled yet
        if L == 0 and d == 0:
            return 0.0
        
        # Calculate exponent difference: (L/λ)^k - ((L+d)/λ)^k
        exponent_diff = (L / lam) ** k - ((L + d) / lam) ** k
        
        # P = 1 - exp[exponent_diff]
        probability = 1.0 - np.exp(exponent_diff)
        
        return probability'''
    
    def calculate_reliability(self, distance_km, scale, shape):
        """
        Calculate Weibull reliability function R(t) = exp(-(t/λ)^k)
        
        Args:
            distance_km: Lifetime mileage
            scale: λ (lambda) - characteristic life
            shape: k - shape parameter
        
        Returns:
            Probability of surviving until distance_km
        """
        return np.exp(-((distance_km / scale) ** shape))
    

    def calculate_conditional_reliability(self, current_km, trip_distance_km, scale, shape):
        """
        Calculate conditional reliability: probability of surviving trip given current age.
        
        Formula: R(t+Δt|t) = R(t+Δt) / R(t)
        
        Args:
            current_km: Current lifetime odometer
            trip_distance_km: Distance of upcoming trip
            scale: λ (lambda)
            shape: k
        
        Returns:
            Conditional probability of surviving the trip
        """
        R_current = self.calculate_reliability(current_km, scale, shape)
        R_after = self.calculate_reliability(current_km + trip_distance_km, scale, shape)
        
        if R_current > 0:
            return R_after / R_current
        else:
            return 0.0
        
    

    def travel(self, simul, travel_time, distance_km=0.0, congested=False):
        """
        Updated to properly track distance traveled
        
        :param simul: Simulator object
        :param travel_time: Duration of travel in minutes
        :param distance_km: Actual distance traveled in kilometers
        :param congested: Whether this trip was congested
        """
        if congested:
            self.metrics.add_metric(simul.state, "travel_time_congested", travel_time)
        else:
            self.metrics.add_metric(simul.state, "travel_time", travel_time)
        
        self.total_distance_km += distance_km
        
    def get_failure_probabilities_for_occured_trip(self, trip_distance_km):
        """
        Get failure probability for all components for a given trip distance.
        Useful for visualization and decision-making.
        
        Returns:
            dict: {category: probability}
        """
        probabilities = {}
        for category, failure_data in self.component_failures.items():
            prob = self.calculate_trip_failure_probability(
                current_km=self.total_distance_km,
                trip_distance_km=trip_distance_km,
                scale=failure_data['scale'],
                shape=failure_data['shape']
            )
            probabilities[category] = prob
        return probabilities
    
    def calculate_trip_failure_probability(self, current_km, trip_distance_km, scale, shape):
        """
        Calculate probability of failure during a specific trip.
        
        Formula: P = 1 - R(Δt|t) = 1 - exp[(t/λ)^k - ((t+Δt)/λ)^k]
        
        Args:
            current_km: L - current lifetime odometer reading
            trip_distance_km: Δt - distance of this trip
            scale: λ (lambda) - characteristic life
            shape: k - shape parameter
        
        Returns:
            Probability of failure during this trip (0 to 1)
        """
        conditional_reliability = self.calculate_conditional_reliability(
            current_km, trip_distance_km, scale, shape
        )
        return 1.0 - conditional_reliability

    def usable(self):
      #return self.maintenance_criticality < MAINTENANCE_THRESHOLD_FOR_NO_RENTAL
      return True

    def hasBattery(self):
      return False
    
    def needsMaintenance(self):
      return False

    def __repr__(self):
        return f"ID-{self.bike_id}-{self.lat}-{self.lon}"
