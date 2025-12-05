from settings import MAINTENANCE_INCREASE_PER_MINUTE, MAINTENANCE_FULL_FIX

####################################################################
# Update bike maintenance criticality after a trip
####################################################################

def update_bike_maintenance(bike, travel_time, rng, battery_level=None, congested=False):
    # Base wear from distance/time

    base_criticality = bike.maintenance_criticality
    base_wear = travel_time * MAINTENANCE_INCREASE_PER_MINUTE
    
    # Random wear (simulate unexpected damage)
    random_wear = 0.0
    if rng.random() < 0.02:  # 2% chance
        random_wear = rng.uniform(0.1, 0.3)

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

    # How many bikes  can you service in the allocated time
    full_bikes = int(maintenance_time // MAINTENANCE_FULL_FIX)
    remaining_time = maintenance_time - (full_bikes * MAINTENANCE_FULL_FIX)
    
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

    # Perform full bike repairs
    if full_bikes > 0:
        for i, bike in enumerate(bikes_to_maintain[:full_bikes]):
            old_criticality = bike.maintenance_criticality
            bike.maintenance_criticality = perform_maintenance_on_bike(bike, MAINTENANCE_FULL_FIX)
            serviced_bikes.append((bike, old_criticality, remaining_time))

    # Perform fractional repair on the next bike, if any time remains
    if remaining_time > 0 and full_bikes < len(bikes_to_maintain):
        bike = bikes_to_maintain[full_bikes]
        old_criticality = bike.maintenance_criticality
        bike.maintenance_criticality = perform_maintenance_on_bike(bike, remaining_time)
        serviced_bikes.append((bike, old_criticality, remaining_time))
    
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

