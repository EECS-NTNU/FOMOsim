import sim
from typing import List
from policies.sjovik_sund.mdp.mdp_formulation import MdpAction
from policies.sjovik_sund.mdp.action_bridge import mdp_action_to_sim_action

def _generate_operational_profiles(state, vehicle, maintenance_enabled: bool):
    """
    Returns discrete sets of operational actions for the current location.
    Each profile is a dict: {'rebalancing': x, 'onsite_repairs': y, 'depot_removals': z, 'load_from_queue': w}
    """
    profiles = []
    
    # If at depot, options are mostly about loading from queue
    if vehicle.is_at_depot():
        n_vehicle = len(vehicle.get_bike_inventory())
        vehicle_capacity = int(getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", n_vehicle)))
        free_cap = max(0, vehicle_capacity - n_vehicle)
        
        repaired_available = len(getattr(vehicle.location, "fixed_queue", {}))
        
        # Profile 1: Load as many repaired bikes as possible
        profiles.append({
            'rebalancing': 0, 'onsite_repairs': 0, 'depot_removals': 0,
            'load_from_queue': min(repaired_available, free_cap)
        })
        # Profile 2: Do not load anything (Drive-through)
        profiles.append({
            'rebalancing': 0, 'onsite_repairs': 0, 'depot_removals': 0,
            'load_from_queue': 0
        })
        return profiles
        
    # --- STATION LOGIC ---
    functional_bikes = [b for b in vehicle.location.get_bikes() if getattr(b, 'is_available', True)]
    target = round(vehicle.location.get_target_state(state.day(), state.hour()))
    
    n_vehicle = len(vehicle.get_bike_inventory())
    n_vehicle_func = sum(1 for b in vehicle.get_bike_inventory() if getattr(b, 'damage_status', None) not in ['depot', 'onsite'])
    vehicle_capacity = int(getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", n_vehicle)))
    free_cap = max(0, vehicle_capacity - n_vehicle)
    
    # Maintenance items
    if maintenance_enabled:
        repairable_bikes = vehicle.location.get_unusable_bikes()
        num_repairable = len(repairable_bikes)
    else:
        num_repairable = 0

    # Calculate greedy rebalancing
    delta = len(functional_bikes) - target
    rebalancing_qty = 0
    if delta > 0: # Pickup (rebalancing < 0)
        rebalancing_qty = -min(delta, free_cap)
    elif delta < 0: # Delivery (rebalancing > 0)
        rebalancing_qty = min(-delta, n_vehicle_func)

    # Diverse operational actions (avoid over-pruning)
    station_spare_cap = getattr(vehicle.location, 'spare_capacity', lambda: 999)()
    max_pickup = -min(len(functional_bikes), free_cap)
    max_delivery = min(n_vehicle_func, station_spare_cap)
    
    rebalancing_options = {rebalancing_qty, max_pickup, max_delivery, 0}
    if max_pickup < 0:
        rebalancing_options.add(int(max_pickup / 2))
    if max_delivery > 0:
        rebalancing_options.add(int(max_delivery / 2))

    # Create distinct profiles based on diverse rebalancing options
    for reb in rebalancing_options:
        profiles.append({'rebalancing': reb, 'onsite_repairs': 0, 'depot_removals': 0, 'load_from_queue': 0})
        
        if maintenance_enabled:
            # All on-site maintenance
            profiles.append({'rebalancing': reb, 'onsite_repairs': num_repairable, 'depot_removals': 0, 'load_from_queue': 0})
            
            # Depot removals
            depot_removals = min(num_repairable, free_cap)
            reb_with_broken = reb
            if reb < 0:  # If picking up bikes, share capacity with broken bikes
                reb_with_broken = -min(abs(reb), max(0, free_cap - depot_removals))
                
            profiles.append({'rebalancing': reb_with_broken, 'onsite_repairs': 0, 'depot_removals': depot_removals, 'load_from_queue': 0})
        
    # Remove duplicates
    unique_profiles = []
    seen = set()
    for p in profiles:
        tup = (p['rebalancing'], p['onsite_repairs'], p['depot_removals'], p['load_from_queue'])
        if tup not in seen:
            seen.add(tup)
            unique_profiles.append(p)
            
    return unique_profiles

def _generate_routing_candidates(state, vehicle, tabu_list, maintenance_enabled: bool, n_func_avail_post: int, free_space_post: int, n_candidates=10):
    """
    Returns a list of target location IDs based on the ANTICIPATED post-operation inventory.
    """
    candidates = []
    
    # Criticality-based stations
    all_stations = [s for s in state.get_stations() if s.id != vehicle.location.id and s.id not in tabu_list]
    
    _cap = int(getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", 1)))
    _cap = max(1, _cap) # Prevent division by zero
    free_ratio = free_space_post / _cap
    
    def score_station(s):
        functional = len([b for b in s.get_bikes() if getattr(b, 'is_available', True)])
        target = round(s.get_target_state(state.day(), state.hour()))
        delta = target - functional # > 0 means starving, < 0 means congested
        
        can_deliver = (delta > 0 and n_func_avail_post > 0)
        can_pickup = (delta < 0 and free_space_post > 0)
        
        score = 0
        if free_ratio <= 0.2 and can_deliver:
            score = delta # Prioritize delivery
        elif free_ratio >= 0.8 and can_pickup:
            score = abs(delta) # Prioritize pickup
        elif 0.2 < free_ratio < 0.8 and (can_deliver or can_pickup):
            score = abs(delta) # Balanced / mixed
        
        base_score = score
        maint_score = 0
        
        # Maintenance need
        if maintenance_enabled:
            broken = len(s.get_unusable_bikes())
            maint_score = broken * 1.5 # Arbitrary weight for broken bikes
            score += maint_score
            
        # Distance penalty
        travel_time = state.get_vehicle_travel_time(vehicle.location.id, s.id)
        penalty_factor = 1.0
        if score > 0 and travel_time > 0:
            # Dampen criticality by root of travel time
            penalty_factor = travel_time ** 0.5
            score = score / penalty_factor
            
        # Temporarily store debug variables on the station object
        s._debug_score = {
            'total': score, 'base_delta': base_score, 'maint': maint_score,
            'travel_time': travel_time, 'penalty': penalty_factor
        }
        return score
    
    all_stations.sort(key=score_station, reverse=True)
    
    # --- Debug print for scoring balance ---
    # Inspect scoring weights (triggers during hour 12 for the first vehicle)
    if state.hour() == 12 and len(tabu_list) == 0:
        print(f"\n--- Routing Scoring Debug (from {vehicle.location.id}) ---")
        for s in all_stations[:3]:
            d = getattr(s, '_debug_score', {})
            print(f"Station {s.id:<3}: Total={d.get('total', 0):.2f} | BaseDelta={d.get('base_delta', 0):.2f} | Maint={d.get('maint', 0):.2f} | Time={d.get('travel_time', 0):.1f} (Penalty=/{d.get('penalty', 1):.2f})")
    
    candidates.extend([s.id for s in all_stations[:n_candidates]])
    
    # Include nearest depot
    if maintenance_enabled:
        depots = state.get_depots()
        if depots:
            depots.sort(key=lambda d: state.get_vehicle_travel_time(vehicle.location.id, d.id))
            nearest_depot = depots[0].id
            if nearest_depot not in candidates and nearest_depot not in tabu_list and nearest_depot != vehicle.location.id:
                candidates.append(nearest_depot)
                
    return candidates

def generate_candidates(state, vehicle, maintenance_enabled: bool) -> List[sim.Action]:
    """
    Generates operational profiles and routing candidates dynamically based on post-operation inventory.
    """
    tabu_list = [v.location.id for v in state.get_vehicles() if v.id != vehicle.id]
    
    debug_mode = (state.hour() == 12 and len(tabu_list) == 0)
    if debug_mode:
        print(f"\n========== CANDIDATE GENERATION DEBUG (Hour 12, Vehicle {vehicle.id} at {vehicle.location.id}) ==========")
        print(f"Current Vehicle Inventory: {sum(1 for b in vehicle.get_bike_inventory() if getattr(b, 'damage_status', None) not in ['depot', 'onsite'])} functional, {sum(1 for b in vehicle.get_bike_inventory() if getattr(b, 'damage_status', None) in ['depot', 'onsite'])} broken bicycles.")
        
    op_profiles = _generate_operational_profiles(state, vehicle, maintenance_enabled)
    
    if debug_mode:
        print(f"Total Base Operational Profiles Evaluated: {len(op_profiles)}")
    
    sim_actions = []
    cur_id = vehicle.location.id
    
    broken_inventory = sum(1 for b in vehicle.get_bike_inventory() if getattr(b, 'damage_status', None) in ['depot', 'onsite'])
    n_vehicle = len(vehicle.get_bike_inventory())
    n_vehicle_func = sum(1 for b in vehicle.get_bike_inventory() if getattr(b, 'damage_status', None) not in ['depot', 'onsite'])
    vehicle_capacity = int(getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", n_vehicle)))
    
    for op in op_profiles:
        broken_after_op = broken_inventory + op['depot_removals']
        # Note: rebalancing < 0 means pickup (vehicle gains bikes), rebalancing > 0 means delivery (vehicle loses bikes)
        functional_after_op = n_vehicle_func - op['rebalancing'] + op['load_from_queue']
        total_after_op = broken_after_op + functional_after_op
        free_space_post = max(0, vehicle_capacity - total_after_op)
        
        if debug_mode:
            print(f"\n--- Operation: {op} ---")
            print(f"Anticipated Post-Op Inventory -> Func: {functional_after_op} | Broken: {broken_after_op} | Free space: {free_space_post}")
            fullness = free_space_post / max(1, vehicle_capacity)
            if fullness <= 0.2:
                print("  -> Expected Heuristic: Vehicle mostly FULL. Will focus strictly on delivering to STARVING stations.")
            elif fullness >= 0.8:
                print("  -> Expected Heuristic: Vehicle mostly EMPTY. Will focus strictly on picking up from CONGESTED stations.")
            else:
                print("  -> Expected Heuristic: Vehicle BALANCED. Will balance evaluation between pickups and deliveries.")
        
        # GENERATE ROUTES SPECIFIC TO THIS OPERATION'S RESULTING INVENTORY!
        routing_targets = _generate_routing_candidates(state, vehicle, tabu_list, maintenance_enabled, n_func_avail_post=functional_after_op, free_space_post=free_space_post)

        if debug_mode:
            print(f"  -> Generated {len(routing_targets)} target routes: {routing_targets}")

        for route in routing_targets:
            is_depot = any(d.id == route for d in state.get_depots())
            
            # Pruning 1: Only go to depot if we have broken bikes (skip if strictly moving functional bikes there)
            if is_depot and broken_after_op == 0 and op['load_from_queue'] == 0:
                # We might want to go to depot if we are completely empty and need functional bikes, but we handle that elsewhere or assume depots mostly deal with broken bikes
                continue
                
            # Pruning 2: If we are fully loaded with broken bikes, ONLY go to depot
            if broken_after_op >= vehicle_capacity and not is_depot:
                continue
                
            mdp_action = MdpAction(
                current_station=cur_id,
                rebalancing=int(op['rebalancing']),
                onsite_repairs=int(op['onsite_repairs']),
                depot_removals=int(op['depot_removals']),
                load_from_queue=int(op['load_from_queue']),
                next_station=route
            )
            sim_actions.append(mdp_action_to_sim_action(mdp_action, state, vehicle))
            

    # Ensure at least one valid action if pruning removed everything 
    # (very rare edge case, e.g. when completely full of broken bikes but no depot available)
    if not sim_actions:
        fallback_route = sorted(state.get_stations(), key=lambda s: state.get_vehicle_travel_time(cur_id, s.id))[1].id # find nearest station
        fallback_action = MdpAction(
            current_station=cur_id,
            rebalancing=0, onsite_repairs=0, depot_removals=0, load_from_queue=0,
            next_station=fallback_route
        )
        sim_actions.append(mdp_action_to_sim_action(fallback_action, state, vehicle))

    if debug_mode:
        print(f"\n[Total valid candidate configurations assembled: {len(sim_actions)}]")
        print("==========================================================================================================\n")

    return sim_actions


