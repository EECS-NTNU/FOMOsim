from settings import MAINTENANCE_INCREASE_PER_MINUTE
import random

def update_bike_maintenance(bike, travel_time, battery_level=None, congested=False):
    """
    Advanced maintenance model considering multiple factors.
    
    Args:
        bike: Bike object
        travel_time: Minutes of travel
        battery_level: Current battery (for e-bikes), optional
        congested: Whether the trip was congested (extra wear)
    
    Returns:
        float: New maintenance criticality (0.0 to 1.0)
    """
    # Base wear from distance/time

    base_criticality = bike.maintenance_criticality
    base_wear = travel_time * MAINTENANCE_INCREASE_PER_MINUTE
    
    # Congestion penalty (rerouting, extra stops)
    # congestion_multiplier = 1.3 if congested else 1.0
    
    # Random wear (simulate unexpected damage)
    # Small chance of extra wear (1% chance of 0.1-0.3 extra)
    random_wear = 0.0
    if random.random() < 0.02:  # 2% chance
        random_wear = random.uniform(0.1, 0.3)
    
    # Accelerated degradation when already high
    # High maintenance bikes degrade faster (cascading failures)
    #degradation_multiplier = 1.0
    #if bike.maintenance_criticality > 0.7:
        #degradation_multiplier = 1.5  # 50% faster when already critical
    #elif bike.maintenance_criticality > 0.5:
        #degradation_multiplier = 1.2  # 20% faster when high
    
    # Calculate total wear
    total_wear = ( base_wear + random_wear)
    
    # Update and cap at 1.0
    new_criticality = min(1.0, base_criticality + total_wear)
    
    return new_criticality


def perform_maintenance(bike, maintenance_time):
    """
    Reduce maintenance criticality based on time spent servicing.
    
    Args:
        bike: Bike object
        maintenance_time: Minutes spent on maintenance
    
    Returns:
        float: New maintenance criticality
    """
    # Any maintenance performed = complete reset to 0
    # This assumes that once a bike is checked/serviced, it's fully maintained
    if maintenance_time > 0:
        return 0.0
    
    return bike.maintenance_criticality

