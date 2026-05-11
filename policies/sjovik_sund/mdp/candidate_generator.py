import sim
from typing import List
from policies.sjovik_sund.mdp.mdp_formulation import MdpAction
from policies.sjovik_sund.mdp.action_bridge import mdp_action_to_sim_action
from settings import SERVICE_TIME_FROM, SYSTEM_CLOSE_HOUR, SYSTEM_OPEN_HOUR, TTV_MAX_OPERATIONAL_HOURS, LATE_SHIFT_HOURS, SERVICE_TIME_TO
from helpers import format_sim_time
 
DEPOT_RETURN_BUFFER_MINUTES = 10.0

candidate_debug_counts = {
    "decisions": 0,
    "swap_capacity_opportunities": 0,
    "total_extra_delivery_capacity": 0,
    "max_extra_delivery_capacity": 0,
}
candidate_debug_examples = []


def reset_candidate_debug_counts() -> None:
    for key in candidate_debug_counts:
        candidate_debug_counts[key] = 0
    candidate_debug_examples.clear()


def get_candidate_debug_summary() -> str:
    decisions = candidate_debug_counts["decisions"]
    opportunities = candidate_debug_counts["swap_capacity_opportunities"]
    if opportunities == 0:
        return f"swap-capacity opportunities=0/{decisions}"

    avg_extra = candidate_debug_counts["total_extra_delivery_capacity"] / opportunities
    summary = (
        f"swap-capacity opportunities={opportunities}/{decisions} "
        f"avg_extra_delivery={avg_extra:.2f} "
        f"max_extra_delivery={candidate_debug_counts['max_extra_delivery_capacity']}"
    )
    if candidate_debug_examples:
        rendered = "; ".join(
            f"{e['time']} {e['station']} spare={e['spare']} depot={e['depot_broken']} "
            f"truck_func={e['truck_func']} truck_free={e['truck_free']} "
            f"desired={e['desired_delivery']} needs_removal={e['needed_removal']}"
            for e in candidate_debug_examples
        )
        summary += f" examples=[{rendered}]"
    return summary


def _nearest_depot_id(state, vehicle):
    depots = state.get_depots()
    if not depots:
        return None
    return min(depots, key=lambda d: state.get_vehicle_travel_time(vehicle.location.id, d.id)).id


def _violates_depot_return_guard(state, vehicle, action, depot_id, buffer_minutes=DEPOT_RETURN_BUFFER_MINUTES):
    """True if a non-depot destination would make returning to depot infeasible before close."""
    if depot_id is None:
        return False

    next_station = getattr(action, "next_location", getattr(action, "next_station", None))
    if next_station == depot_id:
        return False

    close_min = SERVICE_TIME_TO * 60.0
    clock_min = state.time % 1440.0
    time_remaining = max(0.0, close_min - clock_min)

    try:
        travel_to_next = state.get_vehicle_travel_time(vehicle.location.id, next_station)
    except Exception:
        travel_to_next = 0.0
    try:
        travel_next_to_depot = state.get_vehicle_travel_time(next_station, depot_id)
    except Exception:
        travel_next_to_depot = 0.0
    try:
        service_time = action.get_action_time(0.0) if hasattr(action, "get_action_time") else 0.0
    except Exception:
        service_time = 0.0

    required_time = travel_to_next + service_time + travel_next_to_depot + buffer_minutes
    return time_remaining < required_time

 
def _rebalancing_options_range_based(functional_bikes, target, free_cap, n_vehicle_func, station_spare_cap):
    """Range-based options: greedy, max pickup/delivery, halves, no-op. Kept for reuse."""
    delta = len(functional_bikes) - target
    greedy = 0
    if delta > 0:
        greedy = -min(delta, free_cap)
    elif delta < 0:
        greedy = min(-delta, n_vehicle_func)
    max_pickup = -min(len(functional_bikes), free_cap)
    max_delivery = min(n_vehicle_func, station_spare_cap)
    options = {greedy, max_pickup, max_delivery, 0}
    if max_pickup < 0:
        options.add(int(max_pickup / 2))
    if max_delivery > 0:
        options.add(int(max_delivery / 2))
    return options
 
 
def _rebalancing_options_target_centered(functional_bikes, target, free_cap, n_vehicle_func, station_spare_cap):
    """Target-centered options: exact to target, ±25% of target, and no-op."""
    options = {0}
    delta = len(functional_bikes) - target
    if delta > 0:
        options.add(-min(delta, free_cap))
    elif delta < 0:
        options.add(min(-delta, n_vehicle_func))
    target_plus = round(target * 1.25)
    delta_plus = len(functional_bikes) - target_plus
    if delta_plus < 0:
        options.add(min(-delta_plus, n_vehicle_func, station_spare_cap))
    target_minus = round(target * 0.75)
    delta_minus = len(functional_bikes) - target_minus
    if delta_minus > 0:
        options.add(-min(delta_minus, free_cap))
    return options


def _record_swap_capacity_opportunity(
    state,
    vehicle,
    functional_count: int,
    target: int,
    station_spare_cap: int,
    num_depot_broken: int,
    n_vehicle_func: int,
    free_cap: int,
) -> None:
    """Track delivery options blocked only because removals cannot free docks in this model."""
    candidate_debug_counts["decisions"] += 1

    if num_depot_broken <= 0 or n_vehicle_func <= 0:
        return

    desired_deliveries = set()
    exact_to_target = target - functional_count
    if exact_to_target > 0:
        desired_deliveries.add(min(exact_to_target, n_vehicle_func))

    target_plus = round(target * 1.25)
    plus_to_target = target_plus - functional_count
    if plus_to_target > 0:
        desired_deliveries.add(min(plus_to_target, n_vehicle_func))

    feasible_blocked = []
    for desired in desired_deliveries:
        if desired <= 0 or desired <= station_spare_cap:
            continue
        needed_removal = desired - station_spare_cap
        if needed_removal <= num_depot_broken and needed_removal <= free_cap + desired:
            feasible_blocked.append((desired, needed_removal))

    if not feasible_blocked:
        return

    best_desired, needed_removal = max(feasible_blocked, key=lambda item: item[0] - station_spare_cap)
    extra_capacity = best_desired - station_spare_cap
    candidate_debug_counts["swap_capacity_opportunities"] += 1
    candidate_debug_counts["total_extra_delivery_capacity"] += extra_capacity
    candidate_debug_counts["max_extra_delivery_capacity"] = max(
        candidate_debug_counts["max_extra_delivery_capacity"],
        extra_capacity,
    )

    if len(candidate_debug_examples) < 5:
        candidate_debug_examples.append({
            "time": format_sim_time(state.time),
            "station": vehicle.location.id,
            "spare": station_spare_cap,
            "depot_broken": num_depot_broken,
            "truck_func": n_vehicle_func,
            "truck_free": free_cap,
            "desired_delivery": best_desired,
            "needed_removal": needed_removal,
        })
 
 
def _generate_operational_profiles(state, vehicle, maintenance_enabled: bool):
    """
    Returns discrete sets of operational actions for the current location.
    Each profile is a dict:
    {'rebalancing': x, 'onsite_repairs': y, 'depot_removals': z,
     'depot_dropoffs': u, 'load_from_queue': w}
    Each profile is a dict:
    {'rebalancing': x, 'onsite_repairs': y, 'depot_removals': z,
     'depot_dropoffs': u, 'load_from_queue': w}
    """
    profiles = []
 
    if vehicle.is_at_depot():
        n_vehicle = len(vehicle.get_bike_inventory())
        vehicle_capacity = int(getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", n_vehicle)))
        free_cap = max(0, vehicle_capacity - n_vehicle)
 
        repaired_available = len(getattr(vehicle.location, "fixed_queue", {}))
        in_repair_count = sum(len(bikes) for _, bikes in getattr(vehicle.location, "in_repair", []))
        # Depot-damaged cargo is dropped off first, so it frees up space for repaired bikes.
        n_depot_on_vehicle = sum(1 for b in vehicle.get_bike_inventory()
                                 if getattr(b, 'damage_status', None) == 'depot')
        n_func_on_vehicle = sum(1 for b in vehicle.get_bike_inventory()
                                if getattr(b, 'damage_status', None) not in ['depot', 'onsite'])
        effective_free_cap = free_cap + n_depot_on_vehicle
        will_load = min(repaired_available, effective_free_cap)

        print(
            f"[DEPOT VISIT] {format_sim_time(state.time)} "
            f"| fixed_queue={repaired_available} in_repair={in_repair_count} "
            f"| vehicle: {n_func_on_vehicle} func + {n_depot_on_vehicle} depot-broken / cap={vehicle_capacity} "
            f"| eff_free={effective_free_cap} → will_load={will_load}"
        )

        profiles.append({
            'rebalancing': 0, 'onsite_repairs': 0, 'depot_removals': 0,
            'depot_dropoffs': n_depot_on_vehicle,
            'load_from_queue': will_load
        })

        return profiles
 
    # --- STATION LOGIC ---
    functional_bikes = [
        b for b in vehicle.location.get_bikes()
        if getattr(b, "is_available", True)
        and getattr(b, "damage_status", None) not in ("onsite", "depot")
    ]
    target = round(vehicle.location.get_target_state(state.day(), state.hour()))
 
    n_vehicle = len(vehicle.get_bike_inventory())
    n_vehicle_func = sum(1 for b in vehicle.get_bike_inventory() if getattr(b, 'damage_status', None) not in ['depot', 'onsite'])
    vehicle_capacity = int(getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", n_vehicle)))
    free_cap = max(0, vehicle_capacity - n_vehicle)
    station_spare_cap = getattr(vehicle.location, 'spare_capacity', lambda: 999)()
 
    if maintenance_enabled:
        station_bikes = vehicle.location.get_bikes()
        num_depot_broken = len([b for b in station_bikes if getattr(b, 'damage_status', None) == 'depot'])
        num_onsite_broken = len([
            b for b in station_bikes
            if getattr(b, 'damage_status', None) == 'onsite'
            and not getattr(b, "onsite_repair_in_progress", False)
        ])
    else:
        num_depot_broken = 0
        num_onsite_broken = 0

    _record_swap_capacity_opportunity(
        state=state,
        vehicle=vehicle,
        functional_count=len(functional_bikes),
        target=target,
        station_spare_cap=station_spare_cap,
        num_depot_broken=num_depot_broken,
        n_vehicle_func=n_vehicle_func,
        free_cap=free_cap,
    )
 
    rebalancing_options = _rebalancing_options_target_centered(
        functional_bikes, target, free_cap, n_vehicle_func, station_spare_cap
    )
 
    fractions = [0.0, 0.25, 0.50, 0.75, 1.0]
    onsite_options = {round(num_onsite_broken * f) for f in fractions} if num_onsite_broken > 0 else {0}
    depot_options = {round(num_depot_broken * f) for f in fractions} if num_depot_broken > 0 else {0}
 
    seen = set()
    for reb in rebalancing_options:
        if reb > 0 and reb > station_spare_cap:
            continue
        for onsite in onsite_options:
            for depot in depot_options:
                # Delivery (reb > 0) frees truck space; pickup (reb < 0) consumes it
                space_available_for_broken = free_cap + max(0, reb) + min(0, reb)
                valid_depot_removal = min(depot, max(0, space_available_for_broken))
 
                tup = (reb, onsite, valid_depot_removal, 0)
                if tup not in seen:
                    seen.add(tup)
                    label = 'do_nothing' if reb == 0 else ('pickup' if reb < 0 else 'delivery')
                    if onsite > 0 and valid_depot_removal > 0:
                        label += '+maintenance'
                    elif onsite > 0:
                        label += '+onsite'
                    elif valid_depot_removal > 0:
                        label += '+depot_removal'
                    profiles.append({
                        'rebalancing': reb,
                        'onsite_repairs': onsite,
                        'depot_removals': valid_depot_removal,
                        'depot_dropoffs': 0,
                        'load_from_queue': 0,
                        'label': label
                    })
 
    return profiles
 
def _classify_op_profile(op: dict, is_at_depot: bool = False) -> str:
    """Return the pre-computed label from the profile dict."""
    if is_at_depot:
        return 'load_queue' if op.get('load_from_queue', 0) > 0 else 'depot_passthrough'
    return op.get('label', 'unknown')
 
 
def _generate_routing_candidates(state, vehicle, tabu_list, maintenance_enabled: bool, n_func_avail_post: int, free_space_post: int, n_candidates=10, debug_mode: bool = False):
    """
    Returns (list of target location IDs, dict of {station_id: heuristic_score})
    based on the ANTICIPATED post-operation inventory.
    """
    candidates = []
 
    # Criticality-based stations
    all_stations = [s for s in state.get_stations() if s.id != vehicle.location.id and s.id not in tabu_list]
 
    _cap = int(getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", 1)))
    _cap = max(1, _cap) # Prevent division by zero
    free_ratio = free_space_post / _cap
 
    def score_station(s):
        bikes = s.get_bikes()
        functional = len([
            b for b in bikes
            if getattr(b, "is_available", True)
            and getattr(b, "damage_status", None) not in ("onsite", "depot")
        ])
        occupied_docks = len(bikes)
        free_docks = max(0, s.capacity - occupied_docks)
        target = round(s.get_target_state(state.day(), state.hour()))
        delta = functional - target # > 0 means congested, < 0 means starving
 
        # Net demand over next hour (bikes/hour)
        net_demand = (s.get_arrive_intensity(state.day(), state.hour())
                      - s.get_leave_intensity(state.day(), state.hour()))
 
        # Time to violation: how many calendar hours until station overflows or empties
        if net_demand > 0:
            ttv = free_docks / net_demand
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
 
        # Maintenance need — count both depot and onsite broken bikes
        if maintenance_enabled:
            broken = len([
                b for b in s.get_bikes()
                if getattr(b, 'damage_status', None) == 'depot'
                or (
                    getattr(b, 'damage_status', None) == 'onsite'
                    and not getattr(b, "onsite_repair_in_progress", False)
                )
            ])
            maint_score = broken * 1.5
            score += maint_score
 
        # Distance penalty
        travel_time = state.get_vehicle_travel_time(vehicle.location.id, s.id)
        penalty_factor = 1.0
        if score > 0 and travel_time > 0:
            # Dampen criticality by root of travel time
            penalty_factor = travel_time ** 0.35
            score = score / penalty_factor

        # Soft penalty: if going to this station makes depot return infeasible, suppress the score.
        # travel_to_station + station_to_depot > time_left → can't get back before shift ends.
        if score > 0:
            _depots = state.get_depots()
            if _depots:
                _shift_end_min = SERVICE_TIME_TO * 60.0
                _clock_min_now = state.time % 1440.0
                _time_left = max(0.0, _shift_end_min - _clock_min_now)
                if 0.0 < _time_left < _shift_end_min:  # only apply during active shift
                    _station_to_depot = state.get_vehicle_travel_time(s.id, _depots[0].id)
                    if travel_time + _station_to_depot > _time_left:
                        score *= 0.15
 
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
 
    # Capture heuristic scores before extending candidates list
    station_scores = {s.id: getattr(s, '_debug_score', {}).get('total', 0.0) for s in all_stations}
 
    candidates.extend([s.id for s in all_stations[:n_candidates]])
 
    # Include nearest depot
    if maintenance_enabled:
        depots = state.get_depots()
        if depots:
            depots.sort(key=lambda d: state.get_vehicle_travel_time(vehicle.location.id, d.id))
            nearest_depot = depots[0].id
            if nearest_depot not in candidates and nearest_depot not in tabu_list and nearest_depot != vehicle.location.id:
                candidates.append(nearest_depot)
    return candidates, station_scores
 
def generate_candidates(state, vehicle, maintenance_enabled: bool, n_routing: int = 10, return_metadata: bool = False):
    """
    Generates operational profiles and routing candidates dynamically based on post-operation inventory.
 
    When return_metadata=True, returns (sim_actions, metadata) where metadata is a list of dicts
    parallel to sim_actions, each with keys 'profile_type' and 'heuristic_score'.
    """
    tabu_list = [v.location.id for v in state.get_vehicles() if v.id != vehicle.id]
    op_profiles = _generate_operational_profiles(state, vehicle, maintenance_enabled)
 
    sim_actions = []
    metadata = []
    cur_id = vehicle.location.id
    at_depot = vehicle.is_at_depot()

    n_vehicle = len(vehicle.get_bike_inventory())
    n_vehicle_func = sum(1 for b in vehicle.get_bike_inventory() if getattr(b, 'damage_status', None) not in ['depot', 'onsite'])
    n_vehicle_depot = sum(1 for b in vehicle.get_bike_inventory() if getattr(b, 'damage_status', None) == 'depot')
    n_vehicle_other = n_vehicle - n_vehicle_func - n_vehicle_depot
    vehicle_capacity = int(getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", n_vehicle)))
    depot_id = _nearest_depot_id(state, vehicle) if maintenance_enabled else None
 
    for op in op_profiles:
        depot_cargo_after_op = n_vehicle_depot + op['depot_removals'] - op['depot_dropoffs']
        # Note: rebalancing < 0 means pickup (vehicle gains bikes), rebalancing > 0 means delivery (vehicle loses bikes)
        functional_after_op = n_vehicle_func - op['rebalancing'] + op['load_from_queue']
        total_after_op = depot_cargo_after_op + n_vehicle_other + functional_after_op
        free_space_post = max(0, vehicle_capacity - total_after_op)
 
        if total_after_op > vehicle_capacity:
            continue
 
        profile_label = _classify_op_profile(op, at_depot)
 
        # GENERATE ROUTES SPECIFIC TO THIS OPERATION'S RESULTING INVENTORY!
        routing_targets, routing_scores = _generate_routing_candidates(
            state, vehicle, tabu_list, maintenance_enabled,
            n_func_avail_post=functional_after_op, free_space_post=free_space_post,
            n_candidates=n_routing, debug_mode=False,
        )
 
        for route in routing_targets:
            is_depot = any(d.id == route for d in state.get_depots())
 
            '''# Pruning 1: Only go to depot if we have broken bikes (skip if strictly moving functional bikes there)
            if is_depot and broken_after_op == 0 and op['load_from_queue'] == 0:
                continue
 
            # Pruning 2: If carrying enough broken bikes, ONLY route to depot
            depot_force_threshold = 0.8  # mirrors GreedyMaintenancePolicy.depot_load_threshold
            if broken_after_op >= depot_force_threshold * vehicle_capacity and not is_depot:
                continue'''
 
            mdp_action = MdpAction(
                current_station=cur_id,
                rebalancing=int(op['rebalancing']),
                onsite_repairs=int(op['onsite_repairs']),
                depot_removals=int(op['depot_removals']),
                load_from_queue=int(op['load_from_queue']),
                next_station=route,
                depot_dropoffs=int(op['depot_dropoffs']),
            )
            sim_action = mdp_action_to_sim_action(mdp_action, state, vehicle)
            if _violates_depot_return_guard(state, vehicle, sim_action, depot_id):
                continue
            sim_actions.append(sim_action)
            if return_metadata:
                metadata.append({
                    'profile_type': profile_label,
                    'heuristic_score': routing_scores.get(route, 0.0),
                })
 
 
    # Ensure at least one valid action if pruning removed everything
    # (very rare edge case, e.g. when completely full of broken bikes but no depot available)
    if not sim_actions:
        fallback_route = depot_id
        if fallback_route is None:
            fallback_route = sorted(state.get_stations(), key=lambda s: state.get_vehicle_travel_time(cur_id, s.id))[1].id

        # Preserve the depot operation on fallback. Near shift end, the return
        # guard may prune every outbound route while the vehicle is already at
        # the depot. A zero-operation fallback would leave depot cargo onboard.
        fallback_op = {
            'rebalancing': 0,
            'onsite_repairs': 0,
            'depot_removals': 0,
            'load_from_queue': 0,
            'depot_dropoffs': 0,
        }
        if at_depot and op_profiles:
            fallback_op = op_profiles[0]

        fallback_action = MdpAction(
            current_station=cur_id,
            rebalancing=int(fallback_op.get('rebalancing', 0)),
            onsite_repairs=int(fallback_op.get('onsite_repairs', 0)),
            depot_removals=int(fallback_op.get('depot_removals', 0)),
            load_from_queue=int(fallback_op.get('load_from_queue', 0)),
            next_station=fallback_route,
            depot_dropoffs=int(fallback_op.get('depot_dropoffs', 0)),
        )
        sim_actions.append(mdp_action_to_sim_action(fallback_action, state, vehicle))
        if return_metadata:
            metadata.append({'profile_type': 'do_nothing', 'heuristic_score': 0.0})
 
    if return_metadata:
        return sim_actions, metadata
    return sim_actions
