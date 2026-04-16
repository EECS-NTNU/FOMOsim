import sim
from typing import List
from policies.sjovik_sund.mdp.mdp_formulation import MdpAction
from policies.sjovik_sund.mdp.action_bridge import mdp_action_to_sim_action

def _generate_operational_profiles(state, vehicle, maintenance_enabled: bool):
    """
    Returns discrete sets of operational actions for the current location.
    Each profile is a dict: {'rebalancing': x, 'onsite_repairs': y, 'depot_removals': z, 'load_from_queue': w}
    """
    print("HEI jeg genererer operasjonelle profiler")
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

    # Create distinct profiles
    if not maintenance_enabled:
        # 1. Max rebalancing
        profiles.append({'rebalancing': rebalancing_qty, 'onsite_repairs': 0, 'depot_removals': 0, 'load_from_queue': 0})
        # 2. Medium rebalancing
        profiles.append({'rebalancing': int(rebalancing_qty / 2), 'onsite_repairs': 0, 'depot_removals': 0, 'load_from_queue': 0})
        # 3. No rebalancing
        profiles.append({'rebalancing': 0, 'onsite_repairs': 0, 'depot_removals': 0, 'load_from_queue': 0})
    else:
        # 1. Greedy rebalancing, no maintenance
        profiles.append({'rebalancing': rebalancing_qty, 'onsite_repairs': 0, 'depot_removals': 0, 'load_from_queue': 0})
        
        # 2. All on-site maintenance, no depot load, greedy rebalancing
        profiles.append({'rebalancing': rebalancing_qty, 'onsite_repairs': num_repairable, 'depot_removals': 0, 'load_from_queue': 0})
        
        # 3. Greedy depot load, no on-site, greedy rebalancing with remaining capacity
        depot_removals = min(num_repairable, free_cap)
        reb_with_broken = rebalancing_qty
        if delta > 0:
            reb_with_broken = min(delta, max(0, free_cap - depot_removals))
            
        profiles.append({'rebalancing': reb_with_broken, 'onsite_repairs': 0, 'depot_removals': depot_removals, 'load_from_queue': 0})
        
        # 4. No action (Drive-through)
        profiles.append({'rebalancing': 0, 'onsite_repairs': 0, 'depot_removals': 0, 'load_from_queue': 0})
        
    # Remove duplicates
    unique_profiles = []
    seen = set()
    for p in profiles:
        tup = (p['rebalancing'], p['onsite_repairs'], p['depot_removals'], p['load_from_queue'])
        if tup not in seen:
            seen.add(tup)
            unique_profiles.append(p)
            
    print(f"1. Operational profiles generated: {unique_profiles}")
    return unique_profiles

def _generate_routing_candidates(state, vehicle, tabu_list, maintenance_enabled: bool, n_candidates=5):
    """
    Returns a list of target location IDs.
    """
    print("HEI jeg genererer neste rute")
    candidates = []
    
    # Criticality-based stations
    all_stations = [s for s in state.get_stations() if s.id != vehicle.location.id and s.id not in tabu_list]
    
    _inv = vehicle.get_bike_inventory()
    _n_func_avail = sum(1 for b in _inv if getattr(b, 'damage_status', None) not in ['depot', 'onsite'])
    _cap = int(getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", max(1, len(_inv)))))
    _cap = max(1, _cap) # Prevent division by zero
    _free_space = max(0, _cap - len(_inv))
    free_ratio = _free_space / _cap
    
    def score_station(s):
        functional = len([b for b in s.get_bikes() if getattr(b, 'is_available', True)])
        target = round(s.get_target_state(state.day(), state.hour()))
        delta = target - functional # > 0 means starving, < 0 means congested
        
        can_deliver = (delta > 0 and _n_func_avail > 0)
        can_pickup = (delta < 0 and _free_space > 0)
        
        score = 0
        if free_ratio <= 0.2 and can_deliver:
            score = delta # Prioritize delivery
        elif free_ratio >= 0.8 and can_pickup:
            score = abs(delta) # Prioritize pickup
        elif 0.2 < free_ratio < 0.8 and (can_deliver or can_pickup):
            score = abs(delta) # Balanced / mixed
        
        # Maintenance need
        if maintenance_enabled:
            broken = len(s.get_unusable_bikes())
            score += broken * 1.5 # Arbitrary weight for broken bikes
            
        # Distance penalty
        if score > 0:
            travel_time = state.get_vehicle_travel_time(vehicle.location.id, s.id)
            if travel_time > 0:
                # Dampen criticality by root of travel time to heavily favor near stations unless far ones are extremely critical
                score = score / (travel_time ** 0.5) 
            
        return score
    
    all_stations.sort(key=score_station, reverse=True)
    candidates.extend([s.id for s in all_stations[:n_candidates]])
    
    # Include nearest depot
    if maintenance_enabled:
        depots = state.get_depots()
        if depots:
            depots.sort(key=lambda d: state.get_vehicle_travel_time(vehicle.location.id, d.id))
            nearest_depot = depots[0].id
            if nearest_depot not in candidates and nearest_depot not in tabu_list and nearest_depot != vehicle.location.id:
                candidates.append(nearest_depot)
                
    print(f"2. Routing candidates generated: {candidates}")
    return candidates

def generate_candidates(state, vehicle, maintenance_enabled: bool) -> List[sim.Action]:
    """
    Generates operational profiles and routing candidates independently, building a cartesian product 
    pruned by logical rules to avoid redundant or impossible actions.
    """
    print("HEI jeg sammenslår alt")

    tabu_list = [v.location.id for v in state.get_vehicles() if v.id != vehicle.id]
    
    op_profiles = _generate_operational_profiles(state, vehicle, maintenance_enabled)
    routing_targets = _generate_routing_candidates(state, vehicle, tabu_list, maintenance_enabled)
    
    sim_actions = []
    cur_id = vehicle.location.id
    
    broken_inventory = sum(1 for b in vehicle.get_bike_inventory() if getattr(b, 'damage_status', None) in ['depot', 'onsite'])
    n_vehicle = len(vehicle.get_bike_inventory())
    n_vehicle_func = sum(1 for b in vehicle.get_bike_inventory() if getattr(b, 'damage_status', None) not in ['depot', 'onsite'])
    vehicle_capacity = int(getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", n_vehicle)))
    
    print(f"Vehicle capacity: {vehicle_capacity}, Vehcile inventory: {n_vehicle} (Functional: {n_vehicle_func}, Broken: {broken_inventory})")
    for route_id in routing_targets:
        station = next((s for s in state.get_stations() if s.id == route_id), None)
        if station:
            func_bikes = len([b for b in station.get_bikes() if getattr(b, 'is_available', True)])
            t_state = station.get_target_state(state.day(), state.hour())
            print(f"  Candidate {route_id}: target state={t_state:.2f}, current functional inventory={func_bikes}")
        else:
            print(f"  Candidate {route_id}: (Depot)")

    for op in op_profiles:
        broken_after_op = broken_inventory + op['depot_removals']
        functional_after_op = n_vehicle_func + op['rebalancing'] + op['load_from_queue']
        total_after_op = broken_after_op + functional_after_op

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
    if not sim_actions and routing_targets:
        fallback_action = MdpAction(
            current_station=cur_id,
            rebalancing=0, onsite_repairs=0, depot_removals=0, load_from_queue=0,
            next_station=routing_targets[0]
        )
        sim_actions.append(mdp_action_to_sim_action(fallback_action, state, vehicle))

    print(f"3. Total options generated: {len(sim_actions)}")
    return sim_actions
