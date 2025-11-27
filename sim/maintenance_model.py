from settings import MAINTENANCE_INCREASE_PER_MINUTE, MINUTES_PER_ACTION
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

def perform_maintenance_on_bike(bike, maintenance_time):
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

def process_maintenance_action(vehicle, maintenance_time, time):
    """
    Perform maintenance on bikes at the vehicle's current location.
    """
    if maintenance_time <= 0 or vehicle.is_at_depot():
        return

    # Get bikes at current location that need maintenance, sorted by criticality (highest first)
    if hasattr(vehicle.location, 'get_bikes_in_need_of_maintenance'):
        bikes_to_maintain = vehicle.location.get_bikes_in_need_of_maintenance(threshold=0.0)
    else:
        # Fallback: get all bikes and sort by maintenance criticality
        bikes_at_location = vehicle.location.get_bikes()
        bikes_to_maintain = sorted(
            [bike for bike in bikes_at_location if hasattr(bike, 'maintenance_criticality')],
            key=lambda b: b.maintenance_criticality,
            reverse=True
        )
    
    # Calculate time per bike and number of bikes that can be serviced
    if bikes_to_maintain:
        time_per_bike = MINUTES_PER_ACTION  # 3 minutes per bike for maintenance
        num_bikes_to_service = int(maintenance_time / time_per_bike)
        
        # Print BEFORE maintenance
        print(f"\n  MAINTENANCE at {vehicle.location.id} (t={time:.1f}, allocated={maintenance_time:.1f}min)")
        print(f"     Bikes at station BEFORE maintenance ({len(bikes_to_maintain)} total):")
        for bike in bikes_to_maintain[:min(10, len(bikes_to_maintain))]:  # Show first 10
            usable = "usable" if bike.usable() else "UNUSABLE"
            print(f"       {bike.bike_id}: maint={bike.maintenance_criticality:.3f} ({usable})")
        if len(bikes_to_maintain) > 10:
            print(f"       ... and {len(bikes_to_maintain) - 10} more bikes")
        
        # Apply maintenance to the bikes with highest criticality
        serviced_bikes = []
        for i, bike in enumerate(bikes_to_maintain[:num_bikes_to_service]):
            old_criticality = bike.maintenance_criticality
            bike.maintenance_criticality = perform_maintenance_on_bike(bike, time_per_bike)
            serviced_bikes.append((bike, old_criticality))
        
        # Print AFTER maintenance
        print(f"\n     Serviced {len(serviced_bikes)} bikes ({time_per_bike:.1f} min each):")
        for bike, old_crit in serviced_bikes:
            print(f"       {bike.bike_id}: {old_crit:.3f} -> {bike.maintenance_criticality:.3f} (now usable)")
        
        # Show remaining bikes needing maintenance
        remaining_high = [b for b in bikes_to_maintain[num_bikes_to_service:] if b.maintenance_criticality >= 0.5]
        if remaining_high:
            print(f"\n     Remaining bikes still needing maintenance ({len(remaining_high)} with maint>=0.5):")
            for bike in remaining_high[:5]:  # Show first 5
                usable = "usable" if bike.usable() else "UNUSABLE"
                print(f"       {bike.bike_id}: maint={bike.maintenance_criticality:.3f} ({usable})")
            if len(remaining_high) > 5:
                print(f"       ... and {len(remaining_high) - 5} more")
        print()

