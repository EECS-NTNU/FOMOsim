import math
from settings import MAINTENANCE_INCREASE_PER_MINUTE, MAINTENANCE_FULL_FIX

####################################################################
# Update bike maintenance criticality after a trip
####################################################################

def update_bike_maintenance(bike, travel_time, rng ,battery_level=None, congested=False):
    # Base wear from distance/time

    base_criticality = bike.maintenance_criticality
    base_wear = travel_time * MAINTENANCE_INCREASE_PER_MINUTE
    
    # Random wear (simulate unexpected damage)
    random_wear = 0.0
    if rng.random() < 0.01:  # 1% chance
        random_wear = rng.uniform(0.0, 0.2)

    # Calculate total wear
    total_wear = ( base_wear + random_wear)
    
    # Update and cap at 1.0
    new_criticality = min(1.0, base_criticality + total_wear)
    
    return new_criticality

###################################################################
# Performing maintenance at station
# Assumes if 3 full minutes is given, the bike is fully repaired to 0.0 criticality
# Partial maintenance reduces criticality proportionally
###################################################################

def perform_maintenance_on_bike(bike, maintenance_time):
    if maintenance_time > 0:
        reduction = maintenance_time / MAINTENANCE_FULL_FIX
        # Reduce criticality proportionally, but not below 0
        new_criticality = max(0.0, bike.maintenance_criticality * (1 - reduction))
        return new_criticality
    
    return bike.maintenance_criticality

####################################################################
# Process maintenance action for a vehicle at its current location
# This function gets the bike in need of maintenance at a station and sorts them by criticality
# It then performs maintenance on as many bikes as possible within the allocated maintenance_time

####################################################################

def process_maintenance_action(vehicle, maintenance_time, time):

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

    # If there are no bikes needing maintenance, exit
    if not bikes_to_maintain:
        return
    
    # Calculate time per bike and number of bikes that can be serviced
    #time_per_bike = MINUTES_PER_ACTION  # 3 minutes per bike for maintenance

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
    remaining_budget = maintenance_time
    MAX_TIME_PER_BIKE = 5.0

    for bike in bikes_to_maintain:
        if remaining_budget <= 0.001:
            break
            
        # Calculate time needed: 0.8 crit -> 5 min, scaled linearly
        # Formula: time = (crit / 0.8) * 5.0
        needed_time = (bike.maintenance_criticality) * MAX_TIME_PER_BIKE
        needed_time = min(MAX_TIME_PER_BIKE, needed_time)
        
        # Allocate what is available or needed
        time_to_spend = min(remaining_budget, needed_time)
        
        if time_to_spend > 0:
            old_criticality = bike.maintenance_criticality
            bike.maintenance_criticality = perform_maintenance_on_bike(bike, time_to_spend)
            serviced_bikes.append((bike, old_criticality, time_to_spend))
            remaining_budget -= time_to_spend
    
    # Print AFTER maintenance
    print(f"\n     Serviced {len(serviced_bikes)} bikes:")
    for bike, old_crit, used_time in serviced_bikes:
        usable = "usable" if bike.usable() else "UNUSABLE"
        print(
            f"       {bike.bike_id}: {old_crit:.3f} -> {bike.maintenance_criticality:.3f} "
            f"(maintenance {used_time:.1f} min, {usable})"
        )
   
    # Show remaining bikes needing maintenance
    remaining_high = [
        b for b in bikes_to_maintain 
        if b.maintenance_criticality >= 0.8
        ]


    if remaining_high:
        print(f"\n     Remaining bikes still needing maintenance ({len(remaining_high)} with maint>=0.8):")
        for bike in remaining_high[:5]:  # Show first 5
            usable = "usable" if bike.usable() else "UNUSABLE"
            print(f"       {bike.bike_id}: maint={bike.maintenance_criticality:.3f} ({usable})")
        if len(remaining_high) > 5:
            print(f"       ... and {len(remaining_high) - 5} more")
    print()

