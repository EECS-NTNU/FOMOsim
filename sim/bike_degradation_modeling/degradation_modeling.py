"""
Component-level failure tracking using Weibull distributions.
Separate from maintenance criticality - tracks physical part lifespans.
"""
import numpy as np
from sim.bike_degradation_modeling import damage_configuration


'''def initialize_component_failures(bike):
    """
    Initialize failure distances for each component using Weibull distribution.
    Called once when bike is created.
    """
    bike.component_failures = {}
    
    for category, params in damage_configuration.DAMAGE_CATEGORIES.items():
        # Sample from Weibull distribution
        u = np.random.random()
        failure_km = params['scale'] * ((-np.log(1 - u)) ** (1 / params['shape']))
        
        bike.component_failures[category] = {
            'next_failure_km': failure_km,
            'total_failures': 0,
            'last_failure_km': 0.0
        }'''


'''def check_component_failures(bike, simul):
    """
    Check if any components have failed based on distance traveled.
    Called after each trip.
    """
    for category, failure_data in bike.component_failures.items():
        if bike.total_distance_km >= failure_data['next_failure_km']:
            _trigger_component_failure(bike, simul, category)
            
            # Sample next failure distance
            params = damage_configuration.DAMAGE_CATEGORIES[category]
            u = np.random.random()
            interval_km = params['scale'] * ((-np.log(1 - u)) ** (1 / params['shape']))
            
            failure_data['total_failures'] += 1
            failure_data['last_failure_km'] = bike.total_distance_km
            failure_data['next_failure_km'] = bike.total_distance_km + interval_km'''


'''def _trigger_component_failure(bike, simul, category):
    """Record the component failure in metrics"""
    simul.state.metrics.add_aggregate_metric(simul.state, f"failure_{category}", 1)
    simul.state.metrics.add_aggregate_metric(simul.state, "total_failures", 1)

    # Get current time - try multiple access patterns
    if hasattr(simul, 'time'):
        current_time = simul.time
    elif hasattr(simul.state, 'time'):
        current_time = simul.state.time
    else:
        current_time = 0.0  # Fallback
    
    # Log the failure
    if hasattr(bike, 'log'):
        bike.log.append({
            'time': current_time,
            'event': 'component_failure',
            'category': category,
            'odometer_km': bike.total_distance_km
        })
    
    # Optional: Print immediate notification
    print(f"COMPONENT FAILURE: Bike {bike.bike_id} - {category} failed at {bike.total_distance_km:.1f}km")'''


'''def update_bike_odometer(bike, distance_km, simul):
    """
    Update bike's total distance and check for component failures.
    Called after each trip.
    """
    bike.total_distance_km += distance_km
    
    if hasattr(bike, 'component_failures'):
        check_component_failures(bike, simul)'''