import sim
from typing import List
from policies.sjovik_sund.mdp.mdp_formulation import MdpAction
from policies.sjovik_sund.mdp.action_bridge import mdp_action_to_sim_action
from settings import SERVICE_TIME_FROM, SYSTEM_CLOSE_HOUR, SYSTEM_OPEN_HOUR, TTV_MAX_OPERATIONAL_HOURS, LATE_SHIFT_HOURS, SERVICE_TIME_TO

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
    
    # Maintenance items — split by repair type since they draw from different bike pools
    if maintenance_enabled:
        station_bikes = vehicle.location.get_bikes()
        num_depot_broken = len([b for b in station_bikes if getattr(b, 'damage_status', None) == 'depot'])
        num_onsite_broken = len([b for b in station_bikes if getattr(b, 'damage_status', None) == 'onsite'])
    else:
        num_depot_broken = 0
        num_onsite_broken = 0

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
            # All on-site maintenance (draws from onsite-damaged bikes, not depot bikes)
            if num_onsite_broken > 0:
                profiles.append({'rebalancing': reb, 'onsite_repairs': num_onsite_broken, 'depot_removals': 0, 'load_from_queue': 0})

            # Depot removals: when delivering (reb > 0), space is freed post-op so account for it
            delivery_count = max(0, reb)
            depot_removals = min(num_depot_broken, free_cap + delivery_count)
            if depot_removals > 0:
                reb_with_broken = reb
                if reb < 0:  # Picking up bikes shares capacity with broken bikes
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

def _generate_routing_candidates(state, vehicle, tabu_list, maintenance_enabled: bool, n_func_avail_post: int, free_space_post: int, n_candidates=10, debug_mode: bool = False):
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
        delta = functional - target # > 0 means congested, < 0 means starving

        # Net demand over next hour (bikes/hour)
        net_demand = (s.get_arrive_intensity(state.day(), state.hour())
                      - s.get_leave_intensity(state.day(), state.hour()))

        # Time to violation: how many calendar hours until station overflows or empties
        if net_demand > 0:
            ttv = (s.capacity - functional) / net_demand
        elif net_demand < 0:
            ttv = functional / (-net_demand)
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
            _morning_target = round(s.get_target_state((state.day() + 1) % 7, SERVICE_TIME_FROM))
            _morning_gap = abs(functional - _morning_target) / max(1, s.capacity)
            _shift_weight = 1.0 - (_hours_until_end / LATE_SHIFT_HOURS)  # 0→1 as shift ends
            urgency = min(1.0, urgency + _shift_weight * _morning_gap)

        can_deliver = (delta < 0 and n_func_avail_post > 0)
        can_pickup = (delta > 0 and free_space_post > 0)

        score = 0
        if free_ratio <= 0.2 and can_deliver:
            score = abs(delta) # Prioritize delivery
        elif free_ratio >= 0.8 and can_pickup:
            score = delta # Prioritize pickup
        elif 0.2 < free_ratio < 0.8 and (can_deliver or can_pickup):
            score = abs(delta) # Balanced / mixed

        base_score = score

        # Demand direction: amplify when demand worsens the imbalance, dampen when self-correcting
        if score > 0:
            demand_worsens = (delta < 0 and net_demand < 0) or (delta > 0 and net_demand > 0)
            demand_corrects = (delta < 0 and net_demand > 0) or (delta > 0 and net_demand < 0)
            if demand_worsens:
                score *= (1.0 + urgency)
            elif demand_corrects:
                score *= max(0.1, 1.0 - urgency * 0.5)

        maint_score = 0

        # Maintenance need
        if maintenance_enabled:
            broken = len([b for b in s.get_bikes() if getattr(b, 'damage_status', None) == 'depot'])
            maint_score = broken * 1.5
            score += maint_score

        # Distance penalty
        travel_time = state.get_vehicle_travel_time(vehicle.location.id, s.id)
        penalty_factor = 1.0
        if score > 0 and travel_time > 0:
            # Dampen criticality by root of travel time
            penalty_factor = travel_time ** 0.35
            score = score / penalty_factor

        # Temporarily store debug variables on the station object
        s._debug_score = {
            'total': score, 'base_delta': delta, 'maint': maint_score,
            'travel_time': travel_time, 'penalty': penalty_factor,
            'functional': functional, 'target': target,
            'net_demand': net_demand, 'ttv': ttv, 'ttv_cap': _ttv_cap, 'urgency': urgency
        }
        return score
    
    all_stations.sort(key=score_station, reverse=True)
    
    # --- Debug print for scoring balance ---
    if debug_mode:
        print(f"\n--- Routing Scoring Debug (from {vehicle.location.id}) ---")
        print(f"Vehicle Free Ratio: {free_ratio:.2f} (Func post-op: {n_func_avail_post}, Free post-op: {free_space_post})")
        
        if free_ratio <= 0.2:
            print("  -> SYSTEMATIC VERIFICATION: MOSTLY FULL. Top candidates MUST have BaseDelta < 0 (Starving).")
        elif free_ratio >= 0.8:
            print("  -> SYSTEMATIC VERIFICATION: MOSTLY EMPTY. Top candidates MUST have BaseDelta > 0 (Congested).")
        else:
            print("  -> SYSTEMATIC VERIFICATION: BALANCED. Top candidates can have BaseDelta > 0 or < 0.")

        for s in all_stations[:5]: # Show top 5
            d = getattr(s, '_debug_score', {})
            # Verify delta direction matches fullness criteria (ignoring 0 score stations or maintenance driven)
            is_valid = False
            base_delta = d.get('base_delta', 0)
            if free_ratio <= 0.2 and base_delta < 0: is_valid = True
            elif free_ratio >= 0.8 and base_delta > 0: is_valid = True
            elif 0.2 < free_ratio < 0.8: is_valid = True
            elif d.get('maint', 0) > 0 and base_delta == 0: is_valid = True # Maintenance overrides
            
            valid_str = " OK " if is_valid or d.get('total', 0) == 0 else "FAIL"
            print(f"Station {s.id:<3} [{valid_str}]: Total={d.get('total', 0):.2f} | BaseDelta={base_delta:.0f} (Inv: {d.get('functional', 0)} / Tgt: {d.get('target', 0)}) | NetDem={d.get('net_demand', 0):.2f} TTV={d.get('ttv', 8):.1f}h/Cap={d.get('ttv_cap', 8):.0f}h Urg={d.get('urgency', 0):.2f} | Maint={d.get('maint', 0):.2f} | Time={d.get('travel_time', 0):.1f} (Penalty=/{d.get('penalty', 1):.2f})")
    
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
    op_profiles = _generate_operational_profiles(state, vehicle, maintenance_enabled)
    
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
        
        # GENERATE ROUTES SPECIFIC TO THIS OPERATION'S RESULTING INVENTORY!
        routing_targets = _generate_routing_candidates(state, vehicle, tabu_list, maintenance_enabled, n_func_avail_post=functional_after_op, free_space_post=free_space_post, debug_mode=True)

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

    return sim_actions


