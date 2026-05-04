import sim
from typing import List, Tuple, Union
from policies.sjovik_sund.mdp.mdp_formulation import MdpAction
from policies.sjovik_sund.mdp.action_bridge import mdp_action_to_sim_action
from settings import (
    SYSTEM_OPEN_HOUR, 
    SYSTEM_CLOSE_HOUR, 
    SERVICE_TIME_TO, 
    TTV_MAX_OPERATIONAL_HOURS, 
    LATE_SHIFT_HOURS
)

def _generate_operational_profiles(state, vehicle, maintenance_enabled: bool, verbose: bool):
    if verbose:
        print("HEI jeg genererer operasjonelle profiler")
    profiles = []
    
    n_vehicle = len(vehicle.get_bike_inventory())
    vehicle_capacity = int(getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", n_vehicle)))
    free_cap = max(0, vehicle_capacity - n_vehicle)
    
    if vehicle.is_at_depot():
        repaired_available = len(getattr(vehicle.location, "fixed_queue", {}))
        in_repair_count = sum(len(bikes) for _, bikes in getattr(vehicle.location, "in_repair", []))

        # Only depot-damaged bikes are dropped off; onsite bikes are never at the depot
        n_depot_on_vehicle = sum(1 for b in vehicle.get_bike_inventory()
                                 if getattr(b, 'damage_status', None) == 'depot')
        n_func_on_vehicle = sum(1 for b in vehicle.get_bike_inventory()
                                if getattr(b, 'damage_status', None) not in ['depot', 'onsite'])

        # Depot cargo is dropped off first, freeing space for repaired bikes
        effective_free_cap = free_cap + n_depot_on_vehicle
        will_load = min(repaired_available, effective_free_cap)

        if verbose:
            print(
                f"[DEPOT VISIT] t={state.time:.0f} "
                f"| fixed_queue={repaired_available} in_repair={in_repair_count} "
                f"| vehicle: {n_func_on_vehicle} func + {n_depot_on_vehicle} depot / cap={vehicle_capacity} "
                f"| eff_free={effective_free_cap} → will_load={will_load}"
            )

        profiles.append({
            'rebalancing': 0, 'onsite_repairs': 0, 'depot_removals': 0,
            'depot_dropoffs': n_depot_on_vehicle,
            'load_from_queue': will_load,
        })
        return profiles
        
    functional_bikes = [b for b in vehicle.location.get_bikes() if getattr(b, 'is_available', True)]
    target = round(vehicle.location.get_target_state(state.day(), state.hour()))
    
    n_vehicle_func = sum(1 for b in vehicle.get_bike_inventory() if getattr(b, 'damage_status', None) not in ['depot', 'onsite'])
    station_spare_cap = getattr(vehicle.location, 'spare_capacity', lambda: 999)()
    
    if maintenance_enabled:
        all_station_bikes = vehicle.location.get_bikes()
        num_onsite = sum(1 for b in all_station_bikes if getattr(b, 'damage_status', None) == 'onsite')
        num_depot  = sum(1 for b in all_station_bikes if getattr(b, 'damage_status', None) == 'depot')
    else:
        num_onsite, num_depot = 0, 0
 
 
    # --- A. Rebalancing Options (Target-based) ---
    # rebalancing > 0 means DROP OFF. rebalancing < 0 means PICK UP.
    rebalancing_options = {0} # Always include a "do no rebalancing" option
 
    # 1. Rebalance exactly to Target
    delta = len(functional_bikes) - target
    if delta > 0:
        rebalancing_options.add(-min(delta, free_cap)) # Pick up excess
    elif delta < 0:
        rebalancing_options.add(min(-delta, n_vehicle_func)) # Drop off deficit
 
    # 2. Rebalance +25% above target (Drop off more)
    target_plus = round(target * 1.25)
    delta_plus = len(functional_bikes) - target_plus
    if delta_plus < 0:
        rebalancing_options.add(min(-delta_plus, n_vehicle_func, station_spare_cap))
        
    # 3. Rebalance -25% below target (Pick up more)
    target_minus = round(target * 0.75)
    delta_minus = len(functional_bikes) - target_minus
    if delta_minus > 0:
        rebalancing_options.add(-min(delta_minus, free_cap))
        
    # --- B. Fractional Maintenance Options ---
    '''fractions = [0.0, 0.25, 0.50, 0.75, 1.0]
    
    onsite_options = {round(num_onsite * f) for f in fractions} if num_onsite > 0 else {0}
    depot_options  = {round(num_depot * f) for f in fractions} if num_depot > 0 else {0}'''
    
    onsite_options = {num_onsite} if maintenance_enabled else {0}
    depot_options  = {num_depot} if maintenance_enabled else {0}

    #max_pickup = -min(len(functional_bikes), free_cap)
    #max_delivery = min(n_vehicle_func, station_spare_cap)


    seen = set() # To track unique combinations of rebalancing and maintenance
    for reb in rebalancing_options:
        for onsite in onsite_options:
            for depot in depot_options:
                
                # 1: Depot removals require physical truck space.
                # If reb > 0 (dropping off functional bikes), it creates MORE space for broken bikes.
                # If reb < 0 (picking up functional bikes), it uses up space.
                space_available_for_broken = free_cap + max(0, reb) + min(0, reb) 
                
                # If the chosen depot fraction wants to load more broken bikes than we have space for,
                # clip it to the maximum allowable space so it becomes a valid action.
                valid_depot_removal = min(depot, max(0, space_available_for_broken))
                
                # 2: Total delivery cannot exceed station spare capacity
                # Delivering functional bikes (reb > 0) takes up station space.
                if reb > 0 and reb > station_spare_cap:
                    continue # Impossible action
                
                tup = (reb, onsite, valid_depot_removal, 0)
                if tup not in seen:
                    seen.add(tup)
                    profiles.append({
                        'rebalancing': reb,
                        'onsite_repairs': onsite,
                        'depot_removals': valid_depot_removal,
                        'depot_dropoffs': 0,
                        'load_from_queue': 0,
                    })
            
    if verbose:
        print(f"1. Operational profiles generated ({len(profiles)}): {profiles}")
    return profiles
    
    '''for reb in rebalancing_options:
        profiles.append({'rebalancing': reb, 'onsite_repairs': 0, 'depot_removals': 0, 'load_from_queue': 0})
        
        if maintenance_enabled:
            # Onsite repair: only bikes flagged damage_status=='onsite'
            if num_onsite > 0:
                profiles.append({'rebalancing': reb, 'onsite_repairs': num_onsite, 'depot_removals': 0, 'load_from_queue': 0})

            # Depot removal: only bikes flagged damage_status=='depot'
            delivery_count = max(0, reb)  # If rebalancing is delivering to station, we have more capacity for depot removals
            depot_removals = min(num_depot, free_cap + delivery_count)
            if num_depot > 0:
                reb_with_broken = reb
                if reb < 0:
                    reb_with_broken = -min(abs(reb), max(0, free_cap - depot_removals))
                profiles.append({'rebalancing': reb_with_broken, 'onsite_repairs': 0, 'depot_removals': depot_removals, 'load_from_queue': 0})
        
    unique_profiles = []
    seen = set()
    for p in profiles:
        tup = (p['rebalancing'], p['onsite_repairs'], p['depot_removals'], p['load_from_queue'])
        if tup not in seen:
            seen.add(tup)
            unique_profiles.append(p)
            
    if verbose:
        print(f"1. Operational profiles generated: {unique_profiles}")
    return unique_profiles'''


def _generate_routing_targets(
    state, 
    vehicle, 
    tabu_list, 
    maintenance_enabled: bool, 
    n_candidates: int, 
    wide_search: bool, 
    verbose: bool, 
    current_func: int, 
    current_broken: int, 
    current_space: int,
    training_mode: bool = False
):
    if verbose:
        print("HEI jeg genererer neste rute")
    candidates = []
   
    all_stations = [s for s in state.get_stations() if s.id != vehicle.location.id and s.id not in tabu_list]
   
    # --- POST-OP VEHICLE STATE FOR STATE-DEPENDENT ROUTING ---
    _n_func_avail = current_func
    _free_space = current_space
    _cap = current_func + current_broken + current_space
    free_ratio = _free_space / max(1, _cap)

    def score_station(s):
        functional = len([b for b in s.get_bikes() if getattr(b, 'is_available', True)])
        target = round(s.get_target_state(state.day(), state.hour()))
        delta = target - functional  # > 0 starving, < 0 congested

        # --- TTV / URGENCY (MATCHING INGVILD) ---
        net_demand = (s.get_arrive_intensity(state.day(), state.hour())
                      - s.get_leave_intensity(state.day(), state.hour()))
        
        # Time to violation: how many calendar hours until station overflows or empties
        if net_demand > 0:
            ttv = (s.capacity - functional) / max(0.001, net_demand)
        elif net_demand < 0:
            ttv = functional / max(0.001, -net_demand)
        else:
            ttv = float(TTV_MAX_OPERATIONAL_HOURS)

        # Skip overnight closure: if violation falls during or after the closed window,
        # those hours are operationally free — add the closure duration to TTV.
        # Closure window: [SYSTEM_CLOSE_HOUR, SYSTEM_OPEN_HOUR) wrapping midnight.
        _closure_duration = (SYSTEM_OPEN_HOUR + 24 - SYSTEM_CLOSE_HOUR) % 24
        _current_hour = state.hour()
        _hours_until_close = (SYSTEM_CLOSE_HOUR - _current_hour) % 24
        
        if 0 < _hours_until_close <= ttv:
            ttv += _closure_duration

        # Adjusted cap: calendar hours that cover TTV_MAX_OPERATIONAL_HOURS operational hours
        _ttv_cap = TTV_MAX_OPERATIONAL_HOURS + (
            _closure_duration if _hours_until_close < TTV_MAX_OPERATIONAL_HOURS else 0
        )
        
        ttv = min(ttv, _ttv_cap)
        urgency = 1.0 - ttv / _ttv_cap  # 0 = no urgency, 1 = boundary imminent

        # Late-shift morning pre-positioning: ramp in morning gap as shift end approaches.
        # Uses tomorrow's target at SYSTEM_OPEN_HOUR (4am) — the state customers see first.
        _hours_until_end = (SERVICE_TIME_TO - _current_hour) % 24
        if _hours_until_end <= LATE_SHIFT_HOURS:
            _morning_target = round(s.get_target_state((state.day() + 1) % 7, SYSTEM_OPEN_HOUR))
            _morning_gap = abs(functional - _morning_target) / max(1, s.capacity)
            _shift_weight = 1.0 - (_hours_until_end / LATE_SHIFT_HOURS)  # 0→1 as shift ends
            urgency = min(1.0, urgency + _shift_weight * _morning_gap)

        # --- SCORE BASE ---
        can_deliver = (delta > 0 and _n_func_avail > 0)
        can_pickup  = (delta < 0 and _free_space > 0)

        score = 0
        if free_ratio <= 0.2 and can_deliver:
            score = delta
        elif free_ratio >= 0.8 and can_pickup:
            score = abs(delta)
        elif 0.2 < free_ratio < 0.8 and (can_deliver or can_pickup):
            score = abs(delta)

        # Demand direction: amplify when demand worsens imbalance, dampen when self-correcting
        if score > 0:
            demand_worsens  = (delta > 0 and net_demand < 0) or (delta < 0 and net_demand > 0)
            demand_corrects = (delta > 0 and net_demand > 0) or (delta < 0 and net_demand < 0)
            if demand_worsens:
                score *= (1.0 + urgency)
            elif demand_corrects:
                score *= max(0.1, 1.0 - urgency * 0.5)

        if maintenance_enabled and _free_space > 0:
            broken = sum(1 for b in s.get_bikes() if getattr(b, 'damage_status', None) in ('depot', 'onsite'))
            score += broken * 1.5

        # --- TRAVEL TIME DISCOUNT ---
        if score > 0:
            travel_time = state.get_vehicle_travel_time(vehicle.location.id, s.id)
            if travel_time > 0:
                score = score / (travel_time ** 0.35)

        return score
   
    all_stations.sort(key=score_station, reverse=True)
 
    # NEURAL NETWORK ROUTING FILTERS
    if wide_search:
        n_top_critical = 8
        n_nearest = 7
        
        top_critical = all_stations[:n_top_critical]
        stations_by_distance = sorted(all_stations, key=lambda s: state.get_vehicle_travel_time(vehicle.location.id, s.id))
        nearest_stations = stations_by_distance[:n_nearest]
        
        combined_pool = top_critical + nearest_stations
        seen = set()
        chosen_stations = []
        for s in combined_pool:
            if s.id not in seen:
                seen.add(s.id)
                chosen_stations.append(s)
        candidates.extend([s.id for s in chosen_stations])
    else:
        candidates.extend([s.id for s in all_stations[:n_candidates]])
        
    if training_mode:
        all_station_ids = [s.id for s in state.get_stations()]
        num_wildcards = min(3, len(all_station_ids))
        wildcards = state.rng.choice(all_station_ids, size=num_wildcards, replace=False)
        for wc_id in wildcards:
            if wc_id not in candidates:
                candidates.append(wc_id)
   
    if maintenance_enabled:
        depots = state.get_depots()
        if depots:
            depots.sort(key=lambda d: state.get_vehicle_travel_time(vehicle.location.id, d.id))
            nearest_depot = depots[0].id
            if nearest_depot not in candidates and nearest_depot not in tabu_list and nearest_depot != vehicle.location.id:
                candidates.append(nearest_depot)
               
    if verbose:
        print(f"2. Routing candidates generated: {candidates}")
    return candidates


def generate_candidates(
    state,
    vehicle,
    maintenance_enabled: bool,
    return_pairs: bool = False,
    wide_search: bool = False,
    verbose: bool = False,
    n_candidates: int = 8,
    training_mode: bool = True,
    **kwargs,
) -> Union[List['sim.Action'], List[Tuple]]:
    
    if verbose:
        print(f"\n--- GENERATING CANDIDATES FOR VEHICLE {vehicle.id} AT t={state.time} ---")
 
    cur_id = vehicle.location.id if hasattr(vehicle.location, 'id') else "DEPOT"
    tabu_list = [v.location.id for v in state.get_vehicles() if v.id != vehicle.id]
   
    # 1. Get diverse operations
    op_profiles = _generate_operational_profiles(state, vehicle, maintenance_enabled, verbose)
    
    # BASE STATE (Before Operations)
    n_vehicle = len(vehicle.get_bike_inventory())
    n_vehicle_func = sum(1 for b in vehicle.get_bike_inventory() if getattr(b, 'damage_status', None) not in ['depot', 'onsite'])
    n_vehicle_broken = n_vehicle - n_vehicle_func
    vehicle_capacity = int(getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", n_vehicle)))
    
    sim_actions = []
    mdp_actions = []
    depot_ids = {d.id for d in state.get_depots()}

    for op in op_profiles:
        # 1. Calculate exactly what the vehicle will look like AFTER this specific operation
        func_after_op = n_vehicle_func - op['rebalancing'] + op.get('load_from_queue', 0)
        # depot_dropoffs unloads broken cargo before counting removals from stations
        broken_after_op = n_vehicle_broken - op.get('depot_dropoffs', 0) + op['depot_removals']
        total_after_op = func_after_op + broken_after_op
        space_after_op = max(0, vehicle_capacity - total_after_op)
        
        # Extra Safety Pruning: Don't generate impossible actions that exceed capacity
        if total_after_op > vehicle_capacity:
            continue
            
        # 2. STATE-DEPENDENT ROUTING: Generate targets tailored to this exact post-op inventory
        routing_targets = _generate_routing_targets(
            state, 
            vehicle, 
            tabu_list,
            maintenance_enabled,
            n_candidates=n_candidates,
            wide_search=wide_search,
            verbose=verbose,
            current_func=func_after_op,
            current_broken=broken_after_op,
            current_space=space_after_op,
            training_mode=training_mode
        )

        for route in routing_targets:
            is_depot = route in depot_ids
           
            # Pruning logic: Don't go to depot if we have nothing to drop off or pick up
            if is_depot and broken_after_op == 0 and op.get('load_from_queue', 0) == 0:
                continue
               
            # Pruning logic: If fully loaded with broken bikes, ONLY go to depot
            if broken_after_op >= vehicle_capacity and not is_depot:
                continue
               
            mdp_action = MdpAction(
                current_station=cur_id,
                rebalancing=int(op['rebalancing']),
                onsite_repairs=int(op['onsite_repairs']),
                depot_removals=int(op['depot_removals']),
                load_from_queue=int(op.get('load_from_queue', 0)),
                next_station=route,
                depot_dropoffs=int(op.get('depot_dropoffs', 0)),
            )
            sim_action = mdp_action_to_sim_action(mdp_action, state, vehicle)
            sim_actions.append(sim_action)
            mdp_actions.append(mdp_action)
 
    if not sim_actions:
        fallback_target = sorted(state.get_stations(), key=lambda s: state.get_vehicle_travel_time(cur_id, s.id))[1].id
        fallback_mdp = MdpAction(
            current_station=cur_id,
            rebalancing=0, onsite_repairs=0, depot_removals=0, load_from_queue=0,
            next_station=fallback_target,
            depot_dropoffs=0,
        )
        sim_actions.append(mdp_action_to_sim_action(fallback_mdp, state, vehicle))
        mdp_actions.append(fallback_mdp)
 
    if verbose:
        print(f"3. Total options generated: {len(sim_actions)}")
        
    if return_pairs:
        return list(zip(mdp_actions, sim_actions))
    return sim_actions