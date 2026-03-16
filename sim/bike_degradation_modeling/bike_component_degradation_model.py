import numpy as np
from settings import ENABLE_COMPONENT_FAILURES, SAMPLE_BIKES_TO_TRACK
from .damage_configuration import DAMAGE_CATEGORIES

class ComponentFailureModel:
    """
    Weibull-based component failure probability model.
    Handles all failure calculations using component-specific odometers.
    """
    
    @staticmethod
    def calculate_reliability(distance_km, scale, shape):
        """
        Calculate Weibull reliability function R(t) = exp(-(t/λ)^k)
        
        Args:
            distance_km: Component lifetime mileage
            scale: λ (lambda) - characteristic life
            shape: k - shape parameter
        
        Returns:
            Probability of surviving until distance_km
        """
        return np.exp(-((distance_km / scale) ** shape))
    
    @staticmethod
    def calculate_conditional_reliability(current_km, trip_distance_km, scale, shape):
        """
        Calculate conditional reliability: probability of surviving trip given current age.
        
        Formula: R(t+Δt|t) = R(t+Δt) / R(t)
        
        Args:
            current_km: Current component odometer (km since last repair)
            trip_distance_km: Distance of upcoming trip
            scale: λ (lambda)
            shape: k
        
        Returns:
            Conditional probability of surviving the trip
        """
        R_current = ComponentFailureModel.calculate_reliability(current_km, scale, shape)
        R_after = ComponentFailureModel.calculate_reliability(
            current_km + trip_distance_km, scale, shape
        )
        
        if R_current > 0:
            return R_after / R_current
        else:
            return 0.0
    
    @staticmethod
    def calculate_trip_failure_probability(current_km, trip_distance_km, scale, shape):
        """
        Calculate probability of failure during a specific trip.
        
        Formula: P = 1 - R(Δt|t) = 1 - exp[(t/λ)^k - ((t+Δt)/λ)^k]
        
        Args:
            current_km: L - current component odometer reading
            trip_distance_km: Δt - distance of this trip
            scale: λ (lambda) - characteristic life
            shape: k - shape parameter
        
        Returns:
            Probability of failure during this trip (0 to 1)
        """
        conditional_reliability = ComponentFailureModel.calculate_conditional_reliability(
            current_km, trip_distance_km, scale, shape
        )
        return 1.0 - conditional_reliability
    
    @staticmethod
    def calculate_hazard_rate(current_km, scale, shape):
        """
        Calculate instantaneous hazard rate (failure rate).
        
        Formula: z(t) = (k/λ) * (t/λ)^(k-1)
        
        Args:
            current_km: Current component odometer
            scale: λ (lambda)
            shape: k
        
        Returns:
            Hazard rate at current_km
        """
        if current_km > 0:
            return (shape / scale) * ((current_km / scale) ** (shape - 1))
        else:
            return (shape / scale)
    
    @staticmethod
    def check_component_failures_on_trip(bike, trip_distance_km, rng, verbose=False):
        """
        Check if any component fails during this trip using component-specific odometers.
        
        Args:
            bike: Bike object with component_odometers and component_failures
            trip_distance_km: Distance of the current trip
            rng: Random number generator
            verbose: Print detailed failure probability info
        
        Returns:
            dict: Failed components with their details, or empty dict if no failures
        """
        if not ENABLE_COMPONENT_FAILURES:
            return {}
        
        failed_components = {}
        
        if verbose and (bike.bike_id in SAMPLE_BIKES_TO_TRACK):
            print(f"\n{'='*100}")
            print(f"[FAILURE RISK EVALUATION] Bike {bike.bike_id}")
            print(f"  Total bike odometer: {bike.total_distance_km:.2f}km | Trip distance: {trip_distance_km:.2f}km")
            print(f"{'='*100}")
            print(f"  {'Component':<30} {'Comp.Odo(km)':<15} {'Hazard z(t)':<15} {'P(fail)':<12} {'u~U(0,1)':<12} {'Result':<15}")
            print(f"  {'-'*100}")
        
        for category, failure_data in bike.component_failures.items():
            # Use component-specific odometer
            component_km = bike.component_odometers[category]
            scale = failure_data['scale']
            shape = failure_data['shape']
            
            # Calculate failure probability
            failure_prob = ComponentFailureModel.calculate_trip_failure_probability(
                component_km, trip_distance_km, scale, shape
            )
            
            # Calculate hazard rate for logging
            hazard_rate = ComponentFailureModel.calculate_hazard_rate(component_km, scale, shape)
            
            # Draw random number
            u = rng.random()
            
            # Check if failure occurs
            failed = (u < failure_prob)
            
            if verbose and (bike.bike_id in SAMPLE_BIKES_TO_TRACK):
                result = "FAILURE!" if failed else "OK"
                print(f"  {category:<30} {component_km:<15.2f} {hazard_rate:<15.6f} {failure_prob:<12.6f} {u:<12.4f} {result:<15}")
            
            if failed:
                failed_components[category] = {
                    'component_odometer': component_km,
                    'total_bike_km': bike.total_distance_km,
                    'failure_probability': failure_prob,
                    'hazard_rate': hazard_rate,
                    'trip_distance': trip_distance_km
                }
        
        if verbose and (bike.bike_id in SAMPLE_BIKES_TO_TRACK):
            print(f"  {'-'*100}")
            print(f"{'='*100}\n")
        
        return failed_components
    
    @staticmethod
    def get_component_health_summary(bike):
        """
        Get a summary of all component odometers and health status.
        
        Args:
            bike: Bike object
        
        Returns:
            dict: Component health information
        """
        summary = {}
        
        for category in bike.component_odometers:
            odometer = bike.component_odometers[category]
            mttf = bike.component_failures[category]['mttf_km']
            scale = bike.component_failures[category]['scale']
            shape = bike.component_failures[category]['shape']
            
            # Calculate current reliability
            reliability = ComponentFailureModel.calculate_reliability(odometer, scale, shape)
            
            summary[category] = {
                'odometer_km': odometer,
                'mttf_km': mttf,
                'reliability': reliability,
                'usage_ratio': odometer / mttf if mttf > 0 else 0,
                'total_failures': bike.component_failures[category]['total_failures'],
                'depot_fixes': bike.component_failures[category]['depot_fixes'],
                'onsite_fixes': bike.component_failures[category]['onsite_fixes']
            }
        
        return summary